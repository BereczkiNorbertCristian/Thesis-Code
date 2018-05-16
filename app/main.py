

import tkinter as tk
import cv2
import PIL.Image
import PIL.ImageTk
import time
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

from tkinter import font as tfont
from image_streamer import ImageStreamer


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())


class App:

    def __init__(self, root, title, video_source=0):

        self.init_retinanet()

        default_font = tfont.Font(family='Times New Roman', size=12)

        WIDTH_PERC = 0.8
        HEIGHT_PERC = 0.8

        self.root = root
        self.root.title(title)
        self.video_source = video_source

        self.vid = ImageStreamer(self.video_source)
        self.vid.width = int(self.vid.width * WIDTH_PERC)
        self.vid.height = int(self.vid.height * HEIGHT_PERC)

        self.canvas = tk.Canvas(
            root, width=self.vid.width, height=self.vid.height)
        # self.canvas.pack()
        self.canvas.grid(row=0, column=1)

        SCREEN_W = root.winfo_screenwidth()
        SCREEN_H = root.winfo_screenheight()

        root.geometry('{0}x{1}+{2}+{3}'.format(int(SCREEN_W),
                                               int(SCREEN_H), 0, 0))

        buttons = self.init_buttons(root, default_font)
        self.button_detect_image = buttons[0]

        self.output_canvas = tk.Canvas(
            root, width=self.vid.width, height=self.vid.height)
        # self.output_canvas.pack()
        self.output_canvas.grid(row=1, column=1)

        self.delay = 15
        self.update()

        root.mainloop()

    def init_retinanet(self):
        self.model = models.load_model(
            'models/retinanet_mobile128_sign.h5', backbone_name='mobilenet128')
        self.labels_to_names = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L',
                                11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

    def run_detection(self, image):
        draw = image

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        boxes, scores, labels = self.model.predict_on_batch(
            np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale

        # visualize detections
        itr = 1
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if itr > 1:
                break
            itr += 1

            color = label_color(label)

            b = box.astype(int)
            draw_box(draw, b, color=color)

            caption = "{} {:.3f}".format(self.labels_to_names[label], score)
            draw_caption(draw, b, caption)
        return draw

    def take_snapshot(self):
        ret, frame = self.vid.get_frame()

        if ret:
            detection_image = self.run_detection(frame)
            resized_image = cv2.resize(
                detection_image, (self.vid.width, self.vid.height))
            self.output_photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(resized_image))
            self.output_canvas.create_image(
                0, 0, image=self.output_photo, anchor=tk.NW)
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") +
                        ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        ret, frame = self.vid.get_frame()

        if ret:
            frame = cv2.resize(frame, (self.vid.width, self.vid.height))
            self.photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.root.after(self.delay, self.update)

    def detect_video_func(self):
        pass

    def init_buttons(self, root, default_font):

        button_detect_image = tk.Button(
            root, text="Detect Image", command=self.take_snapshot, fg='white', bg='#44475A')
        button_detect_video = tk.Button(
            root, text="Detect Video", command=self.detect_video_func, fg='white', bg='#45C264')

        button_detect_image.config(
            height=3, width=10, font=default_font, padx=4, pady=4)
        button_detect_video.config(
            height=3, width=10, font=default_font, padx=4, pady=4)

        button_detect_image.grid(row=0, column=0)
        # button_detect_video.grid(row=1, column=0)

        return [button_detect_image, button_detect_video]


if __name__ == '__main__':

    App(tk.Tk(), "Sign Language Detection", 0)
