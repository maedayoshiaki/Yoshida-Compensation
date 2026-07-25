# EDSDK trigger capture hardware smoke test (2026-07-25)

## Environment

- Branch: `feature/triggered-batch-capture`
- Camera backend: Canon EDSDK, `edsdk-python` 0.2.0
- Trigger path: `transfer="memory"`
- Projection window: user-visible window 5
  - Win32 device: `\\.\DISPLAY4`
  - Bounds: `(3440, 0)`, `1280 x 720`
- Camera RAW quality: `LR`
- Smoke-test RAW development: `half_size=true`

The OpenCV window was moved and resized before switching to fullscreen.
The camera session and the triggered-capture API were both verified against
the connected camera.

## Trigger/development ordering

Three patterns (black, gray 128, white) were projected and triggered first.
Only after all three trigger calls completed did RAW development start:

```text
triggered=black
triggered=gray128
triggered=white
all_triggers_complete; starting_raw_development
developed=black
developed=gray128
developed=white
```

This confirms that RAW development is no longer interleaved with the
timing-critical projection loop.

Artifacts:

- `data/hardware_trigger_smoke_20260725_121719/`

## Exposure and ROI diagnosis

The original repository settings (`Av=7.1`, `Tv=1/30`, `ISO=200`) were too
bright for non-HDR color-matrix capture. In the derived projection region,
the white pattern saturated 73.8% of red pixels and 70.4% of blue pixels.

The test was repeated with:

- `Av=13`
- `Tv=1/15` (four complete refresh periods at 60 Hz)
- `ISO=100`
- full-resolution interior ROI: `(2778, 826, 1837, 2324)`

The validated white ROI had zero saturated pixels. Its linear-light,
8-bit-scaled statistics were:

| channel | mean | p01 | p99 | saturation |
| --- | ---: | ---: | ---: | ---: |
| R | 115.06 | 72.38 | 155.66 | 0% |
| G | 47.86 | 28.62 | 73.19 | 0% |
| B | 107.07 | 64.61 | 152.77 | 0% |

Black, red, green, blue, and white were visibly correct. Each primary had
the expected dominant camera channel:

| projection | measured ROI mean R/G/B (8-bit-scaled linear light) |
| --- | --- |
| black | 0.10 / 0.03 / 0.09 |
| red | 87.18 / 0.31 / 1.90 |
| green | 15.47 / 57.23 / 7.34 |
| blue | 0.04 / 0.00 / 88.77 |
| white | 115.06 / 47.86 / 107.07 |

Artifacts:

- `data/hardware_color_smoke_20260725_121856/`

The validated exposure and ROI were applied to `config.toml`.

## Actual sample.py eight-pattern path

`sample.capture_projection_patterns()` was run with the eight RGB cube
corners (`num_divisions=2`). All eight frames were returned in projection
order. A global four-term color-mixing fit using the measured ROI produced:

- frame count: 8 projected / 8 captured
- matrix rank: 4 (full rank)
- normalized RMSE: `0.0432`
- maximum absolute error: `0.0810`

All smoke-test thresholds passed (`RMSE < 0.05`, maximum error `< 0.10`).

Artifacts:

- `data/hardware_sample8_smoke_20260725_122114/`

## Timing and transfer-mode decision

For eight patterns with a 300 ms display wait and half-size RAW development:

| mode | trigger/projection loop | later USB transfer | later RAW development | total |
| --- | ---: | ---: | ---: | ---: |
| memory trigger | 21.72 s | included | 3.68 s | about 25.40 s |
| deferred card | 18.36 s | 9.98 s | 3.33 s | 31.67 s |

`deferred_card` shortened the projection loop but increased total time and
retained originals on the camera card. The application therefore keeps
`transfer="memory"`: it is faster end-to-end on this hardware, avoids card
accumulation, and still removes RAW development from the projection loop.

The deferred-card benchmark intentionally left eight RAW originals on the
camera card. Host-side temporary files were removed automatically.
