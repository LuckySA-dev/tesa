# !pip install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# step 0
# # โหลดภาพ
# bgr = cv2.imread('images/drones.jpg')

# # แปลงเป็น RGB
# rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# # แสดงผล
# plt.imshow(rgb); plt.axis('off')
# plt.title("Original")
# plt.show()

# step 1
bgr = cv2.imread('images/drones.jpg')
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)

blue_sky = (H>=90) & (H<=135) & (S>=0) & (V>=80)
clouds   = (S<=35) & (V>=180)
sky_mask = (blue_sky | clouds).astype(np.uint8) * 255

# plt.imshow(sky_mask, cmap='gray'); plt.axis('off'); plt.title("Step 1: Sky Mask"); plt.show()

kernel = np.ones((5,5), np.uint8)

sky_mask = cv2.erode(sky_mask, kernel, iterations=8)

plt.imshow(sky_mask, cmap='gray'); plt.axis('off'); plt.title("Sky Mask"); plt.show()