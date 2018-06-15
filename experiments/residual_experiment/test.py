import numpy as np
import scipy.misc
import keras.backend as K
import tensorflow as tf
import cv2

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
from keras.initializers import glorot_uniform
from matplotlib.pyplot import imshow
from blocks import *

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

# ______________ START HERE ______________

tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [1, 2, 2, 1])
    X = np.random.randn(1, 2, 2, 1)
    A = convolutional_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block='a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = " + str(out[0].shape))
    print("out_len = " + str(len(out)))


model = load_model('resnet32.h5')

img = cv2.imread('raw_dataset/A/a/color_0_0111.png',cv2.IMREAD_COLOR)
img = cv2.resize(img,(150,150),interpolation=cv2.INTER_LINEAR)
lst = np.empty((1,150,150,3))
lst[0] = img
res = model.predict(lst, verbose=True)
print(res)