
import shutil
import os
import sys
import random

# ---------------- FUNCTIONS ---------------------

def split_train_test(label_dict,SAMPLE_NR):

    training_images = []
    validation_images = []
    for label,images in label_dict.items():
        validation_images_per_label = random.sample(images,SAMPLE_NR)
        training_images_per_label = [image for image in images if image not in validation_images_per_label]
        training_images.extend(training_images_per_label)
        validation_images.extend(validation_images_per_label)
    return (training_images, validation_images)


# ------------------------------------------------

datasets = sys.argv[1:]

SUPER_DATASET_DIR = 'super_dataset'
TRAIN_DATASET = os.path.join(SUPER_DATASET_DIR,'train')
VALIDATION_DATASET = os.path.join(SUPER_DATASET_DIR,'validation')
SAMPLE_NR = 10

for dataset in datasets:
    label_dict = {}
    BBOXES_PATH = os.path.join(dataset,'bboxes.csv')
    
    with open(BBOXES_PATH,'r') as f:
        for line in f:
            label = line.strip().split(",")[5]
            if label not in label_dict.keys():
                label_dict[label] = []
            label_dict[label].append(line)
    
    training_images, validation_images = split_train_test(label_dict,SAMPLE_NR)

    # images are string lines
    with open(os.path.join(TRAIN_DATASET,'bboxes.csv'),'a') as f:
        for line in training_images:
            f.write(line.strip() + '\n')
            img_fname = line.strip().split(",")[0]
            IMG_PATH_FROM = os.path.join(dataset,img_fname)
            IMG_PATH_TO = os.path.join(TRAIN_DATASET,img_fname)
            shutil.copyfile(IMG_PATH_FROM,IMG_PATH_TO)

    with open(os.path.join(VALIDATION_DATASET,'bboxes.csv'),'a') as f:
        for line in validation_images:
            f.write(line.strip() + '\n')
            img_fname = line.strip().split(",")[0]
            IMG_PATH_FROM = os.path.join(dataset,img_fname)
            IMG_PATH_TO = os.path.join(VALIDATION_DATASET,img_fname)
            shutil.copyfile(IMG_PATH_FROM,IMG_PATH_TO)



