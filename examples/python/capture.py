from examples.python.sample import capture_image, warp_image
import cv2
import os

captured_image = capture_image()
if captured_image is None:
    print("Failed to capture image")
    exit(1)
os.makedirs("data/manual_capture/wo_warp", exist_ok=True)
cv2.imwrite(
    "data/manual_capture/wo_warp/captured_image.png",
    cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR),
)

warped_image = warp_image(
    captured_image,
    "C:\\py_scripts\\CompenNeSt-plusplus\\data\\light10\\pos1\\normal\\cam\\raw\\sl\\correspondence_map.npy",
    1920,
    1080,
    512,
    512,
)

os.makedirs("data/manual_capture/with_warp", exist_ok=True)
cv2.imwrite(
    "data/manual_capture/with_warp/warped_image.png",
    cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR),
)
print("Captured and warped images saved.")
