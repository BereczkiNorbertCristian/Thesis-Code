

import tkinter as tk
import cv2
import PIL.Image
import PIL.ImageTk
import time

from tkinter import font as tfont
from image_streamer import ImageStreamer

class App:

    def __init__(self, root, title, video_source=0):

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

    def take_snapshot(self):
        ret, frame = self.vid.get_frame()

        if ret:
            frame = cv2.resize(frame, (self.vid.width, self.vid.height))
            self.output_photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(frame))
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
