import tkinter as tk
import cv2
from PIL import Image, ImageTk

import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

model = models.load_model('resnet50_sign.h5', backbone_name='resnet50')

labels_to_names = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}


width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
lmain = tk.Label(root)
lmain.pack()

def detect(image):
	draw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# preprocess image for network
	image = preprocess_image(image)
	image, scale = resize_image(image)

	# process image
	start = time.time()
	boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
	print("processing time: ", time.time() - start)

	# correct for image scale
	boxes /= scale

	# visualize detections
	for box, score, label in zip(boxes[0], scores[0], labels[0]):
	# scores are sorted so we can break
		if score < 0.5:
			break

		color = label_color(label)

		b = box.astype(int)
		draw_box(draw, b, color=color)

		caption = "{} {:.3f}".format(labels_to_names[label], score)
		draw_caption(draw, b, caption)
	return draw

def show_frame():
	_, frame = cap.read()
	frame = cv2.flip(frame, 1)
	detected_frame = detect(frame)
	img = Image.fromarray(detected_frame)
	imgtk = ImageTk.PhotoImage(image=img)
	lmain.imgtk = imgtk
	lmain.configure(image=imgtk)
	lmain.after(10, show_frame)

show_frame()
root.mainloop()
