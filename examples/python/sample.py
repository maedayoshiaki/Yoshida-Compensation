import os
import numpy as np
from numpy import ndarray
import PIL.Image as Image
from PIL import ImageTk
from tkinter import Tk, Label


from src.python.color_mixing_matrix import (
    generate_projection_patterns,
    apply_inverse_gamma_correction,
    calculate_color_mixing_matrix,
)

from src.python.photometric_compensation import (
    calculate_compensation_image,
)


def Capture() -> ndarray:
    # Placeholder function to simulate image capture
    # In practice, this would interface with camera hardware
    dummy_image = np.random.rand(800, 1280, 3).astype(np.float32)
    return dummy_image


def main():
    # Generate Projection patterns
    linear_proj_patterns = generate_projection_patterns(
        1280, 800, num_divisions=4, dtype=np.uint8
    )
    inv_gamma_patterns = []
    # Save patterns to disk
    os.makedirs("data/linear_proj_patterns", exist_ok=True)
    os.makedirs("data/inv_gamma_proj_patterns", exist_ok=True)

    for i, pattern in enumerate(linear_proj_patterns):
        img = Image.fromarray(pattern)
        img.save(os.path.join("data/linear_proj_patterns", f"pattern_{i:02d}.png"))
        # Apply inverse gamma correction to an example image
        inv_gamma_pattern = apply_inverse_gamma_correction(pattern, gamma=2.2)
        inv_gamma_patterns.append(inv_gamma_pattern)
        inv_gamma_img = Image.fromarray(inv_gamma_pattern)
        inv_gamma_img.save(
            os.path.join(
                "data/inv_gamma_proj_patterns", f"inv_gamma_pattern_{i:02d}.png"
            )
        )
        print(f"Saved pattern_{i:02d}.png and inv_gamma_pattern_{i:02d}.png")

    # Display patterns and capture images
    # show pattern on projector's screen and capture with camera
    captured_images = []
    root = Tk()
    w = 1280
    h = 800
    x = root.winfo_screenwidth() - w  # Projector on the right
    y = 0
    # ウィンドウを指定モニター位置＆サイズに設定
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.overrideredirect(True)  # 枠なし
    root.attributes("-topmost", True)

    label = Label(root)
    label.pack(expand=True, fill="both")
    for pattern in inv_gamma_patterns:
        img = Image.fromarray(pattern)
        tk_img = ImageTk.PhotoImage(img)
        label.config(image=tk_img)
        root.update()
        # Simulate a delay for projection stabilization
        root.after(1)
        # Capture image
        captured_image = Capture()
        captured_images.append(captured_image)
        print("Captured an image.")
    root.destroy()

    # After capturing all images, calculate color mixing matrices
    color_mixing_matrices = calculate_color_mixing_matrix(
        proj_images=linear_proj_patterns, captured_images=captured_images
    )

    # Load target image
    target_images = []
    target_image_folder_path = "data/target_images"
    for img_name in os.listdir(target_image_folder_path):
        if img_name.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(target_image_folder_path, img_name)
            target_img = Image.open(img_path).convert("RGB")
            target_img_array = np.array(target_img)
            target_images.append(target_img_array)

    print(f"Loaded {len(target_images)} target images.")
    # Calculate and save compensation images
    for idx, target_image in enumerate(target_images):
        # Calculate compensation image
        compensation_image = calculate_compensation_image(
            target_image=target_image,
            color_mixing_matrices=color_mixing_matrices,
            dtype=np.uint8,
        )

        # Save compensation image
        compensation_img = Image.fromarray(compensation_image)
        compensation_img.save(
            f"data/compensation_images/compensation_image_{idx:02d}.png"
        )
        print(f"Saved compensation_image_{idx:02d}.png")


if __name__ == "__main__":
    main()
