
import os
import numpy as np
import cv2
import random

# Please run this above the raw_dataset dir


def precompute_classes():

	class_to_id = {}
	lst = os.listdir('raw_dataset/A')
	lst.sort()
	for i in range(len(lst)):
		class_to_id[lst[i]] = i
	return class_to_id


def split_train_test(X, Y):
	print("Splitting data...")
	RESIZE_HEIGHT = 50
	RESIZE_WEIGHT = 50
	NO_TEST_IMAGES = 10000
	NO_TRAIN_IMAGES = X.shape[0] - NO_TEST_IMAGES

	indices = random.sample(range(X.shape[0]), NO_TEST_IMAGES)

	X_train = np.ndarray((NO_TRAIN_IMAGES, RESIZE_HEIGHT, RESIZE_WEIGHT, 3), dtype=float)
	Y_train = np.ndarray(NO_TRAIN_IMAGES, dtype=int)
	X_test = np.ndarray((NO_TEST_IMAGES, RESIZE_HEIGHT, RESIZE_WEIGHT, 3), dtype=float)
	Y_test = np.ndarray(NO_TEST_IMAGES, dtype=int)

	cnt_train = 0
	cnt_test = 0
	for i in range(X.shape[0]):
		if i in indices:
			X_test[cnt_test] = X[i]
			Y_test[cnt_test] = Y[i]
			cnt_test += 1
		else:
			X_train[cnt_train] = X[i]
			Y_train[cnt_train] = Y[i]
			cnt_train += 1

	print("End splitting data...")
	return (X_train, Y_train, X_test, Y_test)


def load_dataset():
	print("Begin loading dataset...")
	EXPECTED_CLASSES = 24
	DATASET_FOLDER = 'raw_dataset'
	IMAGE_NUMBER = 65774
	RESIZE_HEIGHT = 50
	RESIZE_WEIGHT = 50
	cnt = 0

	class_to_id = precompute_classes()

	signs = []
	X = np.ndarray(shape=(IMAGE_NUMBER, RESIZE_HEIGHT,
						RESIZE_WEIGHT, 3), dtype=float)
	Y = np.ndarray(IMAGE_NUMBER, dtype=int)

# Used relative paths
	for sign_maker in os.listdir(DATASET_FOLDER):
		SIGN_MAKER_FOLDER = os.path.join(DATASET_FOLDER, sign_maker)
		for sign in os.listdir(SIGN_MAKER_FOLDER):
			signs.append(sign)
			SIGN_SUBFOLDER = os.path.join(SIGN_MAKER_FOLDER, sign)
			for image_fname in os.listdir(SIGN_SUBFOLDER):
				FNAME_PATH = os.path.join(SIGN_SUBFOLDER, image_fname)
				img = cv2.imread(FNAME_PATH, cv2.IMREAD_COLOR)
				img = cv2.resize(img, (RESIZE_HEIGHT, RESIZE_HEIGHT),
								interpolation=cv2.INTER_LINEAR)
				X[cnt] = img
				Y[cnt] = class_to_id[sign]
				cnt += 1

		signs = list(set(signs))
		signs.sort()
		assert(len(signs) == 24)
		assert(len(signs) == len(class_to_id.keys()))

	print("Class to id:")
	print(class_to_id)
	print("End loading dataset...")
	return split_train_test(X, Y)
