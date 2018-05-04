
import numpy as np
import argparse
import os

from scipy import ndimage,misc
from skimage import io,exposure
from random import random,randint

def horizontal_flip(image):
    return image[:,::-1]

def vertical_flip(image):
    return image[::-1,:]

# if a horizontal flip was done then axis = 0
# if a vertical flip was done then axis = 1

def recompute_box(h,w,box,axis):
    x1,y1,x2,y2 = box
    if axis == 0:
        x1 = w - x1
        x2 = w - x2
    else:
        y1 = h - y1
        y2 = h - y2
    return np.array([x1,y1,x2,y2])

def process_line(DATASET_DIR,line):

    VERTICAL_THRESHOLD = 0.5
    HORIZONTAL_THRESHOLD = 0.5
    ADJUST_GAMMA = 0.3
    BLUR = 0.3
    SHARPEN = 0.2

    if line[5] == '':
        return line

    elems = line.split(",")
    fname = elems[0]
    box = np.array([elems[1],elems[2],elems[3],elems[4]]).astype(int)
    label = elems[5]

    IMAGE_PATH = os.path.join(DATASET_DIR,fname)
    image = io.imread(IMAGE_PATH)
    h,w,channels = image.shape

    if VERTICAL_THRESHOLD > random():
        image = vertical_flip(image)
        box = recompute_box(h,w,box,0)
    if HORIZONTAL_THRESHOLD > random():
        image = horizontal_flip(image)
        box = recompute_box(h,w,box,1)
    if ADJUST_GAMMA > random():
        GAMMA_RATE = 0.6 - randint(0,4) / 10
        GAIN_RATE = 0.9 - randint(0,4) / 10
        image = exposure.adjust_gamma(image,GAMMA_RATE,GAIN_RATE)
    if BLUR > random():
        FILTER_H = randint(1,6)
        FILTER_W = randint(1,6)
        image = ndimage.uniform_filter(image, size=(FILTER_H,FILTER_W,1))
    if SHARPEN > random():
        image = misc.imfilter(image,'sharpen')

    aug_fname = 'aug_' + fname
    io.imsave(os.path.join(DATASET_DIR, aug_fname),image)
    return (aug_fname,box,label)

def render_line(fname,box,label):
    return fname + ',' + str(box[0]) + \
            str(box[1]) + str(box[2]) + str(box[3]) + label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augmenter per image')
    parser.add_argument('dataset_dir',type=str,help='introduce the dataset directory containing the images you want to augment')

    args = parser.parse_args()
    DATASET_DIR = args.dataset_dir
    BOXES_PATH = os.path.join(DATASET_DIR,'bboxes.csv') 
    AUG_BOXES_PATH = os.path.join(DATASET_DIR,'aug_bboxes.csv')

    aug_boxes = []
    with open(BOXES_PATH,'r') as f:
        for line in f:
                fname,box,label = process_line(DATASET_DIR,line)
                aug_boxes.append(render_line(fname,box,label))

    with open(AUG_BOXES_PATH,'w') as f:
        for line in aug_boxes:
            f.write(line + '\n')
