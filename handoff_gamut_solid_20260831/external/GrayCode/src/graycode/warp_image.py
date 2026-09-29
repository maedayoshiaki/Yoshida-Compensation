"""
Pixel Map Warper (PyTorch Version)

A GPU-accelerated image warping module using pixel correspondence maps.
Supports both forward warping (splatting) and backward warping (sampling).

Coordinate convention (pixel-is-point, single convention):
    Both source ("XY", camera) and destination ("UV", projector) images use the
    SAME convention: integer coordinate i is the CENTER of pixel i (pixel i covers
    [i-0.5, i+0.5)). This matches OpenCV / numpy. The "XY"/"UV" labels denote the
    source vs destination image spaces, NOT different pixel-center conventions.
    The only place the texel convention (center = i+0.5) appears is the boundary
    with ``F.grid_sample``, encapsulated in ``coords.point_to_normalized``.
    See COORDINATES.md.

Usage:
    >>> warper = PixelMapWarperTorch(pixel_map, device="cuda")
    >>> # Forward warp: source image -> destination image
    >>> uv_img = warper.forward_warp(xy_img, dst_size=(200, 200))
    >>> # Backward warp: destination image -> source image
    >>> xy_img = warper.backward_warp(uv_img, dst_size=(100, 100))

"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union
from enum import Enum

from . import coords
from .config import get_config

class AggregationMethod(Enum):
    """Aggregation method when multiple pixels map to the same location."""

    MEAN = "mean"  # Average of all values (weighted by splat weights)
    MAX = "max"  # Maximum value
    MIN = "min"  # Minimum value
    LAST = "last"  # Last value (overwrite)


class SplatMethod(Enum):
    """Splatting method for forward warping."""

    NEAREST = "nearest"  # Single pixel (original behavior, causes holes)
    BILINEAR = "bilinear"  # Bilinear splatting to 4 neighbors (recommended)


class InpaintMethod(Enum):
    """Inpainting method for filling holes."""

    NONE = "none"  # No inpainting
    CONV = "conv"  # Convolution-based inpainting (GPU compatible)


class PaddingMode(Enum):
    """Padding mode for out-of-bounds sampling."""

    ZEROS = "zeros"  # Fill with zeros
    BORDER = "border"  # Clamp to border values
    REFLECTION = "reflection"  # Reflect at boundaries


class PixelMapWarperTorch:
    """
    Warps images using pixel correspondence map (PyTorch version).

    The pixel map defines correspondences from the source image space ("XY") to
    the destination image space ("UV"):
        (x, y) -> (u, v)

    Coordinate convention (pixel-is-point, both spaces):
        Integer coordinate i is the CENTER of pixel i; pixel (i, j) covers
        [i-0.5, i+0.5) x [j-0.5, j+0.5). The same convention applies to both the
        source ("XY") and destination ("UV") spaces. "XY"/"UV" are space labels,
        not different centering rules. The texel convention (center = i+0.5)
        appears only at the grid_sample boundary (coords.point_to_normalized).

    Forward warp transforms a source image to destination space (splatting).
    Backward warp transforms a destination image to source space (sampling).

    Attributes:
        map_tensor: Pixel correspondence map as tensor (N, 4) [x, y, u, v]
        xy_bounds: Bounds in XY space (min_x, min_y, max_x, max_y)
        uv_bounds: Bounds in UV space (min_u, min_v, max_u, max_v)
        device: Computation device ('cpu' or 'cuda')
    """

    def __init__(
        self,
        pixel_map: Union[
            List[Tuple[Tuple[float, float], Tuple[float, float]]],
            np.ndarray,
            torch.Tensor,
        ],
        device: Optional[str] = None,
    ):
        """
        Initialize the warper with a pixel correspondence map.

        Args:
            pixel_map: Pixel correspondence map in one of the following formats:
                - List of tuples: [((src_x, src_y), (dst_x, dst_y)), ...]
                - NumPy array: shape (N, 4) with columns [src_x, src_y, dst_x, dst_y]
                - PyTorch tensor: shape (N, 4) with columns [src_x, src_y, dst_x, dst_y]
            device: Computation device. If None, uses CUDA if available.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Convert input to tensor
        if isinstance(pixel_map, list):
            flat_data = [[sx, sy, dx, dy] for (sx, sy), (dx, dy) in pixel_map]
            self.map_tensor = torch.tensor(
                flat_data, dtype=torch.float32, device=self.device
            )
        elif isinstance(pixel_map, np.ndarray):
            self.map_tensor = torch.from_numpy(pixel_map).float().to(self.device)
            if self.map_tensor.ndim == 3 and self.map_tensor.shape[1:] == (2, 2):
                self.map_tensor = self.map_tensor.view(-1, 4)
        elif isinstance(pixel_map, torch.Tensor):
            self.map_tensor = pixel_map.float().to(self.device)
        else:
            raise TypeError(f"Unsupported pixel_map type: {type(pixel_map)}")

        # Remove NaN entries
        mask = ~torch.isnan(self.map_tensor).any(dim=1)
        self.map_tensor = self.map_tensor[mask]

        # Load tunable numeric constants from config at instance creation time.
        # インポート時ではなく、インスタンス生成時に設定値を解決する。
        adv_cfg = get_config().warp.advanced
        self._eps = adv_cfg.eps
        self._large_pos = adv_cfg.large_pos
        self._large_neg = adv_cfg.large_neg
        self._invalid_coord = adv_cfg.invalid_coord

        self._cache_bounds()
        self._inpaint_kernels: dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _cache_bounds(self) -> None:
        """Cache the coordinate bounds of the map."""
        if self.map_tensor.numel() == 0:
            self.xy_bounds = (0.0, 0.0, 0.0, 0.0)
            self.uv_bounds = (0.0, 0.0, 0.0, 0.0)
            return

        x, y, u, v = self.map_tensor.T
        self.xy_bounds = (
            x.min().item(),
            y.min().item(),
            x.max().item(),
            y.max().item(),
        )
        self.uv_bounds = (
            u.min().item(),
            v.min().item(),
            u.max().item(),
            v.max().item(),
        )

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pixel(c: torch.Tensor) -> torch.Tensor:
        """Convert a coordinate to a pixel index (pixel-is-point: integer=center).

        Used for both source and destination spaces (single convention).
        Canonical definition: :func:`graycode.coords.to_pixel`.
        """
        return coords.to_pixel(c).long()

    # ------------------------------------------------------------------
    # Aggregation helper (shared by nearest / bilinear splatting)
    # ------------------------------------------------------------------

    @staticmethod
    def _scatter_aggregate(
        out: torch.Tensor,
        weights: torch.Tensor,
        dst_indices: torch.Tensor,
        values: torch.Tensor,
        w: Optional[torch.Tensor],
        aggregation: AggregationMethod,
    ) -> None:
        """
        Accumulate *values* into *out* at *dst_indices* using *aggregation*.

        For weighted splatting pass *w* (per-pixel weights); for nearest pass None.
        *weights* is a flat (M,) tensor that tracks per-destination total weight /
        count and is updated in-place.

        Args:
            out: (B, C, M) accumulation buffer
            weights: (M,) weight / count buffer
            dst_indices: (N,) destination flat indices
            values: (B, C, N) source pixel values (already weighted if bilinear)
            w: (N,) per-entry weights, or None for nearest (treated as 1)
            aggregation: aggregation strategy
        """
        B, C, _ = out.shape
        idx_exp = dst_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1)

        if w is None:
            w_add = torch.ones(len(dst_indices), device=out.device)
        else:
            w_add = w

        if aggregation == AggregationMethod.MEAN:
            out.index_add_(2, dst_indices, values)
            weights.index_add_(0, dst_indices, w_add)

        elif aggregation == AggregationMethod.MAX:
            out.scatter_reduce_(2, idx_exp, values, reduce="amax", include_self=True)
            weights.index_add_(0, dst_indices, w_add)

        elif aggregation == AggregationMethod.MIN:
            out.scatter_reduce_(2, idx_exp, values, reduce="amin", include_self=True)
            weights.index_add_(0, dst_indices, w_add)

        elif aggregation == AggregationMethod.LAST:
            out.scatter_(2, idx_exp, values)
            weights.index_add_(0, dst_indices, w_add)

    def _finalize_aggregation(
        self,
        out: torch.Tensor,
        weights: torch.Tensor,
        aggregation: AggregationMethod,
    ) -> None:
        """Normalize / clean up *out* after all scatter passes."""
        if aggregation == AggregationMethod.MEAN:
            w = weights.view(1, 1, -1)
            mask = w > 0
            out.copy_(torch.where(mask, out / (w + self._eps), out))
        elif aggregation == AggregationMethod.MIN:
            out[out == self._large_pos] = 0
        elif aggregation == AggregationMethod.MAX:
            out[out == self._large_neg] = 0

    def _init_fill(self, aggregation: AggregationMethod) -> float:
        """Return the initial fill value for the output buffer."""
        if aggregation == AggregationMethod.MIN:
            return self._large_pos
        if aggregation == AggregationMethod.MAX:
            return self._large_neg
        return 0.0

    # ------------------------------------------------------------------
    # Forward warp (splatting)
    # ------------------------------------------------------------------

    def forward_warp(
        self,
        src_img: torch.Tensor,
        dst_size: Optional[Tuple[int, int]] = None,
        src_offset: Tuple[int, int] = (0, 0),
        splat_method: Optional[SplatMethod] = None,
        aggregation: Optional[AggregationMethod] = None,
        inpaint: Optional[InpaintMethod] = None,
        inpaint_iter: Optional[int] = None,
        crop_rect: Optional[Tuple[int, int, int, int]] = None,
        output_dtype: Optional[torch.dtype] = None,
        keep_on_device: bool = False,
    ) -> torch.Tensor:
        """
        Forward warp (Splatting): XY image -> UV image.

        Args:
            src_img: Source image in XY coordinates. Shape: (C, H, W) or (B, C, H, W)
            dst_size: Output size as (width, height). If None, auto-calculated.
            src_offset: Offset for source coordinates (x_offset, y_offset).
            splat_method: Splatting method (NEAREST or BILINEAR). None uses config default.
            aggregation: Method for handling overlapping pixels. None uses config default.
            inpaint: Method for filling holes in the output. None uses config default.
            inpaint_iter: Number of inpainting iterations. None uses config default.
            crop_rect: Crop region (x, y, width, height) in UV space.
            output_dtype: Desired output dtype. If None, matches input.
            keep_on_device: If True, keep the result on the computation device.

        Returns:
            Warped image in UV coordinates. Shape matches input batch format.
        """
        wcfg = get_config().warp
        if splat_method is None:
            splat_method = SplatMethod(wcfg.default_splat_method)
        if aggregation is None:
            aggregation = AggregationMethod(wcfg.default_aggregation)
        if inpaint is None:
            inpaint = InpaintMethod(wcfg.default_inpaint)
        if inpaint_iter is None:
            inpaint_iter = wcfg.default_inpaint_iter_forward

        is_batch = src_img.ndim == 4
        if not is_batch:
            src_img = src_img.unsqueeze(0)

        input_dtype = src_img.dtype
        B, C, H, W = src_img.shape
        src_img = src_img.to(self.device).float()

        if dst_size is None:
            dst_w = int(self.uv_bounds[2]) + 1
            dst_h = int(self.uv_bounds[3]) + 1
        else:
            dst_w, dst_h = dst_size

        if splat_method == SplatMethod.BILINEAR:
            out_img, count_img = self._forward_warp_bilinear(
                src_img, dst_w, dst_h, src_offset, aggregation
            )
        else:
            out_img, count_img = self._forward_warp_nearest(
                src_img, dst_w, dst_h, src_offset, aggregation
            )

        if inpaint == InpaintMethod.CONV:
            out_img = self._apply_inpaint_conv(out_img, count_img, inpaint_iter)

        if crop_rect is not None:
            cx, cy, cw, ch = crop_rect
            out_img = out_img[:, :, cy : cy + ch, cx : cx + cw]

        target_dtype = output_dtype if output_dtype is not None else input_dtype
        result = out_img.to(target_dtype)
        if not keep_on_device:
            result = result.cpu()

        return result if is_batch else result.squeeze(0)

    def _forward_warp_bilinear(
        self,
        src_img: torch.Tensor,
        dst_w: int,
        dst_h: int,
        src_offset: Tuple[int, int],
        aggregation: AggregationMethod,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bilinear splatting: distribute each source pixel to 4 destination neighbors."""
        B, C, H, W = src_img.shape

        src_x = self._to_pixel(self.map_tensor[:, 0]) - src_offset[0]
        src_y = self._to_pixel(self.map_tensor[:, 1]) - src_offset[1]

        dst_x_f = self.map_tensor[:, 2]
        dst_y_f = self.map_tensor[:, 3]

        # 4-neighbor pixel indices and bilinear weights.
        # pixel-is-point convention: integer = pixel center = array index, so the
        # two neighbors of coordinate u are floor(u) and floor(u)+1, weighted by
        # the fractional distance. A point on a pixel center (integer u) thus gets
        # full weight on that single pixel (matching the nearest splat and the
        # backward-warp sampling).
        x0 = torch.floor(dst_x_f).long()
        y0 = torch.floor(dst_y_f).long()
        wx1 = dst_x_f - x0.float()
        wy1 = dst_y_f - y0.float()
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        neighbors = [
            (x0, y0, wx0 * wy0),
            (x0 + 1, y0, wx1 * wy0),
            (x0, y0 + 1, wx0 * wy1),
            (x0 + 1, y0 + 1, wx1 * wy1),
        ]

        src_valid = (src_x >= 0) & (src_x < W) & (src_y >= 0) & (src_y < H)

        M = dst_h * dst_w
        out_img = torch.full(
            (B, C, M), self._init_fill(aggregation),
            device=self.device, dtype=torch.float32,
        )
        weight_buf = torch.zeros(M, device=self.device, dtype=torch.float32)

        for dx, dy, w_nb in neighbors:
            dst_valid = (dx >= 0) & (dx < dst_w) & (dy >= 0) & (dy < dst_h)
            valid = src_valid & dst_valid
            if not valid.any():
                continue

            s_x, s_y = src_x[valid], src_y[valid]
            d_idx = dy[valid] * dst_w + dx[valid]
            w = w_nb[valid]
            vals = src_img[:, :, s_y, s_x] * w.view(1, 1, -1)

            self._scatter_aggregate(out_img, weight_buf, d_idx, vals, w, aggregation)

        self._finalize_aggregation(out_img, weight_buf, aggregation)

        out_img = out_img.view(B, C, dst_h, dst_w)
        count_img = (weight_buf > 0).float().view(1, 1, dst_h, dst_w)
        return out_img, count_img

    def _forward_warp_nearest(
        self,
        src_img: torch.Tensor,
        dst_w: int,
        dst_h: int,
        src_offset: Tuple[int, int],
        aggregation: AggregationMethod,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nearest-neighbor splatting: each source pixel maps to one destination pixel."""
        B, C, H, W = src_img.shape

        src_x = self._to_pixel(self.map_tensor[:, 0]) - src_offset[0]
        src_y = self._to_pixel(self.map_tensor[:, 1]) - src_offset[1]
        dst_x = self._to_pixel(self.map_tensor[:, 2])
        dst_y = self._to_pixel(self.map_tensor[:, 3])

        valid = (
            (src_x >= 0) & (src_x < W)
            & (src_y >= 0) & (src_y < H)
            & (dst_x >= 0) & (dst_x < dst_w)
            & (dst_y >= 0) & (dst_y < dst_h)
        )

        s_x, s_y = src_x[valid], src_y[valid]
        d_idx = dst_y[valid] * dst_w + dst_x[valid]
        vals = src_img[:, :, s_y, s_x]

        M = dst_h * dst_w
        out_img = torch.full(
            (B, C, M), self._init_fill(aggregation),
            device=self.device, dtype=torch.float32,
        )
        weight_buf = torch.zeros(M, device=self.device, dtype=torch.float32)

        self._scatter_aggregate(out_img, weight_buf, d_idx, vals, None, aggregation)
        self._finalize_aggregation(out_img, weight_buf, aggregation)

        out_img = out_img.view(B, C, dst_h, dst_w)
        count_img = (weight_buf > 0).float().view(1, 1, dst_h, dst_w)
        return out_img, count_img

    # ------------------------------------------------------------------
    # Backward warp (sampling)
    # ------------------------------------------------------------------

    def backward_warp(
        self,
        uv_img: torch.Tensor,
        dst_size: Optional[Tuple[int, int]] = None,
        src_rect: Optional[Tuple[int, int, int, int]] = None,
        mode: Optional[str] = None,
        padding_mode: Optional[PaddingMode] = None,
        inpaint: Optional[InpaintMethod] = None,
        inpaint_iter: Optional[int] = None,
        return_mask: bool = False,
        output_dtype: Optional[torch.dtype] = None,
        keep_on_device: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Backward warp (Sampling): UV image -> XY image.

        Args:
            uv_img: Source image in UV coordinates. Shape: (C, H, W) or (B, C, H, W)
            dst_size: Output size as (width, height) in XY space.
            src_rect: Region in UV space that uv_img covers, as (u, v, width, height).
            mode: Interpolation mode ('bilinear' or 'nearest'). None uses config default.
            padding_mode: How to handle pixels with no correspondence. None uses config default.
            inpaint: Inpainting method for grid holes. None uses config default.
            inpaint_iter: Number of inpainting iterations. None uses config default.
            return_mask: If True, also return a mask of valid pixels.
            output_dtype: Desired output dtype. If None, matches input.
            keep_on_device: If True, keep the result on the computation device.

        Returns:
            Warped image in XY coordinates.
            If return_mask=True, returns tuple (image, mask).
        """
        wcfg = get_config().warp
        if mode is None:
            mode = wcfg.default_backward_mode
        if padding_mode is None:
            padding_mode = PaddingMode(wcfg.default_padding_mode)
        if inpaint is None:
            inpaint = InpaintMethod(wcfg.default_inpaint)
        if inpaint_iter is None:
            inpaint_iter = wcfg.default_inpaint_iter_backward

        is_batch = uv_img.ndim == 4
        if not is_batch:
            uv_img = uv_img.unsqueeze(0)

        input_dtype = uv_img.dtype
        B, C, H_uv, W_uv = uv_img.shape
        uv_img = uv_img.to(self.device).float()

        uv_offset_x, uv_offset_y, uv_w, uv_h = self._parse_src_rect(
            src_rect, W_uv, H_uv
        )
        dst_w, dst_h, xy_offset_x, xy_offset_y = self._parse_dst_size(dst_size)

        # Build sampling grid
        grid_uv, valid_mask = self._build_sampling_grid(
            dst_w, dst_h, xy_offset_x, xy_offset_y,
            uv_offset_x, uv_offset_y, uv_w, uv_h,
            src_rect is not None,
        )

        # Inpaint grid holes
        if inpaint != InpaintMethod.NONE and inpaint_iter > 0:
            grid_uv, valid_mask = self._inpaint_grid(
                grid_uv, valid_mask, inpaint_iter
            )

        # Sample from UV image
        out_img = self._sample_from_grid(
            uv_img, grid_uv, valid_mask,
            uv_offset_x, uv_offset_y, W_uv, H_uv,
            mode, padding_mode, B, C,
        )

        # Output dtype and device
        target_dtype = output_dtype if output_dtype is not None else input_dtype
        result = out_img.to(target_dtype)
        mask_result = valid_mask
        if not keep_on_device:
            result = result.cpu()
            mask_result = mask_result.cpu()

        if not is_batch:
            result = result.squeeze(0)
            mask_result = mask_result.squeeze(0)

        if return_mask:
            if mask_result.shape[-3] == 1 and mask_result.ndim >= 3:
                mask_result = mask_result.squeeze(-3)
            return result, mask_result
        return result

    def _parse_src_rect(
        self,
        src_rect: Optional[Tuple[int, int, int, int]],
        W_uv: int,
        H_uv: int,
    ) -> Tuple[int, int, int, int]:
        if src_rect is not None:
            return src_rect
        return 0, 0, W_uv, H_uv

    def _parse_dst_size(
        self, dst_size: Optional[Tuple[int, int]]
    ) -> Tuple[int, int, int, int]:
        if dst_size is None:
            dst_w = int(self.xy_bounds[2] - self.xy_bounds[0]) + 1
            dst_h = int(self.xy_bounds[3] - self.xy_bounds[1]) + 1
            xy_off_x = int(self.xy_bounds[0])
            xy_off_y = int(self.xy_bounds[1])
        else:
            dst_w, dst_h = dst_size
            xy_off_x, xy_off_y = 0, 0
        return dst_w, dst_h, xy_off_x, xy_off_y

    def _build_sampling_grid(
        self,
        dst_w: int,
        dst_h: int,
        xy_off_x: int,
        xy_off_y: int,
        uv_off_x: int,
        uv_off_y: int,
        uv_w: int,
        uv_h: int,
        has_src_rect: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a (1, 2, dst_h, dst_w) UV sampling grid and a valid-pixel mask."""
        x_coords, y_coords, u_coords, v_coords = self.map_tensor.T

        x_int = self._to_pixel(x_coords) - xy_off_x
        y_int = self._to_pixel(y_coords) - xy_off_y

        valid = (x_int >= 0) & (x_int < dst_w) & (y_int >= 0) & (y_int < dst_h)
        if has_src_rect:
            uv_valid = (
                (u_coords >= uv_off_x) & (u_coords < uv_off_x + uv_w)
                & (v_coords >= uv_off_y) & (v_coords < uv_off_y + uv_h)
            )
            valid = valid & uv_valid

        x_v, y_v = x_int[valid], y_int[valid]
        u_v, v_v = u_coords[valid], v_coords[valid]

        grid_uv = torch.zeros(
            (1, 2, dst_h, dst_w), device=self.device, dtype=torch.float32
        )
        grid_count = torch.zeros(
            (1, 1, dst_h, dst_w), device=self.device, dtype=torch.float32
        )

        if len(x_v) > 0:
            flat_idx = y_v * dst_w + x_v
            uv_stack = torch.stack([u_v, v_v], dim=0)
            g_flat = grid_uv.view(1, 2, -1)
            c_flat = grid_count.view(1, 1, -1)

            g_flat.index_add_(2, flat_idx, uv_stack.unsqueeze(0))
            c_flat.index_add_(
                2, flat_idx,
                torch.ones(1, 1, len(flat_idx), device=self.device),
            )

            grid_uv = g_flat.view(1, 2, dst_h, dst_w)
            grid_count = c_flat.view(1, 1, dst_h, dst_w)

        valid_mask = grid_count > 0
        grid_uv = torch.where(valid_mask, grid_uv / (grid_count + self._eps), grid_uv)
        grid_uv = torch.where(
            valid_mask, grid_uv, torch.full_like(grid_uv, self._invalid_coord)
        )
        return grid_uv, valid_mask

    def _inpaint_grid(
        self,
        grid_uv: torch.Tensor,
        valid_mask: torch.Tensor,
        iterations: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inpaint holes in the sampling grid and return updated grid + mask."""
        original_valid = valid_mask.clone()
        grid_for_inpaint = torch.where(valid_mask, grid_uv, torch.zeros_like(grid_uv))
        count = valid_mask.float()
        grid_inpainted = self._apply_inpaint_conv(grid_for_inpaint, count, iterations)

        filled = grid_inpainted.abs().sum(dim=1, keepdim=True) > self._eps
        grid_uv = torch.where(filled, grid_inpainted, grid_uv)
        valid_mask = original_valid | filled
        grid_uv = torch.where(
            valid_mask, grid_uv, torch.full_like(grid_uv, self._invalid_coord)
        )
        return grid_uv, valid_mask

    def _sample_from_grid(
        self,
        uv_img: torch.Tensor,
        grid_uv: torch.Tensor,
        valid_mask: torch.Tensor,
        uv_off_x: int,
        uv_off_y: int,
        W_uv: int,
        H_uv: int,
        mode: str,
        padding_mode: PaddingMode,
        B: int,
        C: int,
    ) -> torch.Tensor:
        """Sample from *uv_img* using the prebuilt grid."""
        sample_x = grid_uv[:, 0:1, :, :] - uv_off_x
        sample_y = grid_uv[:, 1:2, :, :] - uv_off_y

        in_bounds = (
            (sample_x >= 0) & (sample_x < W_uv)
            & (sample_y >= 0) & (sample_y < H_uv)
        )
        valid_mask = valid_mask & in_bounds
        sample_x = torch.where(
            in_bounds, sample_x, torch.full_like(sample_x, self._invalid_coord)
        )
        sample_y = torch.where(
            in_bounds, sample_y, torch.full_like(sample_y, self._invalid_coord)
        )

        # pixel-is-point -> grid_sample(align_corners=False) 正規化座標。
        # ここが唯一の texel 規約への境界変換 (see coords.point_to_normalized)。
        norm_x = coords.point_to_normalized(sample_x, max(W_uv, 1))
        norm_y = coords.point_to_normalized(sample_y, max(H_uv, 1))
        grid = torch.cat([norm_x, norm_y], dim=1).permute(0, 2, 3, 1)
        grid_batch = grid.expand(B, -1, -1, -1)

        out_img = F.grid_sample(
            uv_img,
            grid_batch,
            mode=mode,
            padding_mode=padding_mode.value,
            align_corners=False,
        )

        if padding_mode == PaddingMode.ZEROS:
            out_img = out_img * valid_mask.expand(B, C, -1, -1).float()

        return out_img

    # ------------------------------------------------------------------
    # Inpainting
    # ------------------------------------------------------------------

    def _get_inpaint_kernels(
        self, channels: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached 3x3 inpainting kernels for the given channel count."""
        if channels in self._inpaint_kernels:
            return self._inpaint_kernels[channels]

        base = torch.tensor(
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
            device=self.device, dtype=torch.float32,
        )
        # Keep both convolutions in the same (unnormalized) units.  The colour
        # sum is divided by the number of valid neighbours below.
        color_kernel = base.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        weight_kernel = base.view(1, 1, 3, 3)
        self._inpaint_kernels[channels] = (color_kernel, weight_kernel)
        return color_kernel, weight_kernel

    def _apply_inpaint_conv(
        self,
        img: torch.Tensor,
        count_img: torch.Tensor,
        iterations: int = 3,
    ) -> torch.Tensor:
        """
        GPU-based inpainting using convolution.

        Fills empty regions with weighted average of neighboring valid pixels.
        """
        C = img.shape[1]
        kernel, weight_kernel = self._get_inpaint_kernels(C)

        current_img = img.clone()
        current_valid = (count_img > 0).float()

        for _ in range(iterations):
            is_hole = current_valid < 0.5

            neighbor_sum = F.conv2d(
                current_img * current_valid.expand(-1, C, -1, -1),
                kernel, padding=1, groups=C,
            )
            neighbor_weight = F.conv2d(
                current_valid, weight_kernel, padding=1
            )
            can_fill = is_hole & (neighbor_weight > self._eps)
            neighbor_avg = neighbor_sum / neighbor_weight.clamp(min=self._eps)

            current_img = torch.where(can_fill, neighbor_avg, current_img)
            current_valid = torch.where(
                can_fill,
                torch.ones_like(current_valid),
                current_valid,
            )

        return current_img

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_bounds(self) -> dict:
        """Get the coordinate bounds of the pixel map."""
        return {"xy": self.xy_bounds, "uv": self.uv_bounds}

    def __len__(self) -> int:
        return len(self.map_tensor)

    def __repr__(self) -> str:
        return (
            f"PixelMapWarperTorch("
            f"n_points={len(self)}, "
            f"xy_bounds={self.xy_bounds}, "
            f"uv_bounds={self.uv_bounds}, "
            f"device='{self.device}')"
        )
