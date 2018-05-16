
import keras
import matplotlib.pyplot as plt
import cv2
import os
import sys
import numpy as np
import time
import tensorflow as tf
import cfg

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

prediction_times = []


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())


def process_line(line):
    elems = line.strip().split(',')
    image_fname = elems[0]
    bbox = tuple(map(int, [elems[1], elems[2], elems[3], elems[4]]))
    ground_label = elems[5]

    return (image_fname, bbox, ground_label)


def predict_with_topk(image_fname, k):
    global model
    global prediction_times

    image = read_image_bgr(os.path.join(cfg.VALIDATION_PATH, image_fname))

    image = preprocess_image(image)
    image, scale = resize_image(image)

    start = time.time()
    boxes, scores, labels = model.predict_on_batch(
        np.expand_dims(image, axis=0))
    prediction_times.append(time.time() - start)

    if labels[0][0] == -1:
        print("NOTHING TO PREDICT AT" + image_fname)
    return labels[0][:k],scores[0][:k]


model = models.load_model(cfg.MODEL_PATH, backbone_name=cfg.BACKBONE)

N = len(cfg.labels_to_names.keys())
inv_map = {v: k for k, v in cfg.labels_to_names.items()}


all_predictions = 0
good_predictions = 0
K = 5
label_to_scores = {k:[] for k in cfg.labels_to_names.keys()}
with open(cfg.VALIDATION_PATH_ANNOTATIONS, 'r') as f:
    for line in f:

        all_predictions += 1
        print(all_predictions)

        image_fname, bbox, ground_label = process_line(line)
        labels_k,scores_k = predict_with_topk(image_fname,K) 

        ground_label = inv_map[ground_label]

        if ground_label in labels_k:
            good_predictions += 1

print("Stats:")
print("Good predictions: " + str(good_predictions))
print("All predictions: " + str(all_predictions))
print("Top 5 accuracy: " + str(good_predictions / all_predictions))
print("End...")

