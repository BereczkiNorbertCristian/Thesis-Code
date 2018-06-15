import numpy as np
import scipy.misc
import keras.backend as K
import tensorflow as tf
import sys

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils, plot_model
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.initializers import glorot_normal
from matplotlib.pyplot import imshow
from blocks import *
from res_model import *
from data_generator import DataGenerator
from keras.backend.tensorflow_backend import set_session
from dataset_utils import *

'''
Compatible with tensorflow backend
'''
# ----------------- CONFIG ------------------------

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

# ----------------- MAIN --------------------------

class_to_id = get_class_to_id()
X_train, X_val = get_train_val()
Y_train, Y_val = get_labels(class_to_id, X_train, X_val)

params = {'dim': (150, 150),
          'batch_size': 16,
          'n_classes': 24,
          'n_channels': 3,
          'shuffle': True}

train_generator = DataGenerator(X_train, Y_train, **params)
val_generator = DataGenerator(X_val, Y_val, **params)

model = ResNet32(input_shape=(150, 150, 3), classes=params['n_classes'])
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

lr_reducer = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=0.000001)
checkpoint_saver = ModelCheckpoint('resnet32.h5', verbose=True)
tensorboard_logger = TensorBoard('logs',
                                 histogram_freq=0,
                                 write_graph=False,
                                 write_grads=False,
                                 batch_size=16,
                                 write_images=False)

callback_list = [lr_reducer, checkpoint_saver, tensorboard_logger]

# Normalize image vectors

# Convert training and test labels to one hot matrices
print("number of training examples = " + str(len(X_train)))
print("number of test examples = " + str(len(X_val)))
print("X_train shape: " + str(len(X_train)))
print("Y_train shape: " + str(len(Y_train)))
print("X_test shape: " + str(len(X_val)))
print("Y_test shape: " + str(len(Y_val)))

model.summary()

hist = model.fit_generator(
    generator=train_generator,
    validation_data=val_generator,
    callbacks=callback_list,
    use_multiprocessing=True,
    workers=3, epochs=50, verbose=True)


preds = model.evaluate_generator(
    val_generator, workers=6, use_multiprocessing=True)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
