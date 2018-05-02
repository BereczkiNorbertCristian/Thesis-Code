
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import random

def get_names():
	ret = []
	with open("dataset_1/bboxes.csv","r") as f:
		for line in f:
			ret.append(line.split(",")[0])
	return ret

def get_box(img_name):
	with open("dataset_1/bboxes.csv","r") as f:
		for line in f:
			lst = line.split(",")
			if lst[0] == img_name:
				return np.array([lst[1],lst[2],lst[3],lst[4]])

IMG_NAMES = get_names()
IMG_NAME = random.choice(IMG_NAMES)
IMG_PATH = os.path.join('dataset_1',IMG_NAME)
image = read_image_bgr(IMG_PATH)
draw = image.copy()
draw = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)

box_params = get_box(IMG_NAME).astype(int)
color = label_color(3)
draw_box(draw,box_params,color=color)

plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()

