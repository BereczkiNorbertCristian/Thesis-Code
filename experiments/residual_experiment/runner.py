import numpy as np
import scipy.misc
import keras.backend as K
import tensorflow as tf
import sys

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_normal
from matplotlib.pyplot import imshow
from blocks import *
from res_model import *
from data_generator import DataGenerator

# ------------------- FUNCTIONS --------------------


def get_class_to_id():
    H = {}
    with open('class_to_id.csv', 'r') as f:
        for line in f:
            v, k = line.strip().split(',')
            H[k] = int(v)
    return H


def get_train_val():
    train_lst = []
    with open('train.csv', 'r') as f:
        for line in f:
            elems = line.strip().split(',')
            train_lst.append((elems[0], elems[1], elems[2], elems[3]))
    val_lst = []
    with open('validation.csv', 'r') as f:
        for line in f:
            elems = line.strip().split(',')
            val_lst.append((elems[0], elems[1], elems[2], elems[3]))
    return train_lst, val_lst


def get_labels(class_to_id, train_lst, val_lst):
    train_labels = [0] * len(train_lst)
    val_labels = [0] * len(val_lst)
    for el in train_lst:
        ID = int(el[0])
        sign = el[2]
        train_labels[ID] = class_to_id[sign]
    for el in val_lst:
        ID = int(el[0])
        sign = el[2]
        val_labels[ID] = class_to_id[sign]
    return train_labels, val_labels

# ----------------- MAIN --------------------------

class_to_id = get_class_to_id()
X_train, X_val = get_train_val()
Y_train, Y_val = get_labels(class_to_id, X_train, X_val)

params = {	'dim': (150, 150),
           'batch_size': 32,
           'n_classes': 24,
           'n_channels': 3,
           'shuffle': True}

train_generator = DataGenerator(X_train, Y_train, **params)
val_generator = DataGenerator(X_val, Y_val, **params)

model = ResNet32(input_shape=(150, 150, 3), classes=params['n_classes'])
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

# Normalize image vectors

# Convert training and test labels to one hot matrices
print("number of training examples = " + str(len(X_train)))
print("number of test examples = " + str(len(X_val)))
print("X_train shape: " + str(len(X_train)))
print("Y_train shape: " + str(len(Y_train)))
print("X_test shape: " + str(len(X_val)))
print("Y_test shape: " + str(len(Y_val)))

model.fit_generator(
    generator=train_generator,
    validation_data=val_generator,
    use_multiprocessing=True,
    workers=6, epochs=2, verbose=True)

preds = model.evaluate_generator(
    val_generator, workers=6, use_multiprocessing=True, verbose=True)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
