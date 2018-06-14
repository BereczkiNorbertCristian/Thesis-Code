

import cv2

img = cv2.imread('raw_dataset/A/a/color_0_0111.png',cv2.IMREAD_COLOR)
img = cv2.resize(img,(150,150),interpolation=cv2.INTER_LINEAR)

print(img.shape)

