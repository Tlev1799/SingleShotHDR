import cv2
import numpy as np

# 1. Load the original 1920x1080 EXR
img = cv2.imread("./datasets/test_dataset/streetlights.exr", cv2.IMREAD_UNCHANGED)

# 2. Target dimensions
target_w, target_h = 320, 320

# 3. Calculate proportional size
# For 1920x1080 to a 320x320 box, this will scale the image to 320x180
scale = min(target_w / 1920, target_h / 1080)
new_w, new_h = int(1920 * scale), int(1080 * scale)

# Resize using INTER_AREA (essential for downscaling high-res HDR images smoothly)
resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# 4. Create a blank 320x320 black canvas
# We detect the number of channels (RGB or RGBA) from the original file
channels = img.shape[2] if len(img.shape) > 2 else 1
canvas = np.zeros((target_h, target_w, channels), dtype=np.float32)

# 5. Center the 320x180 image vertically onto the 320x320 canvas
# This leaves 70 pixels of black padding on the top and bottom
top = (target_h - new_h) // 2
left = (target_w - new_w) // 2
canvas[top:top+new_h, left:left+new_w] = resized_img

# 6. Save the perfectly square EXR without any distortion
cv2.imwrite("streetlights.exr", canvas)
print("Resized successfully! New shape:", canvas.shape)