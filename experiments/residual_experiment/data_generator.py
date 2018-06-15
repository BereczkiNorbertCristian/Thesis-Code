
import numpy as np
import keras
import os
import cv2


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_metadata, labels, batch_size=32, dim=(150, 150), n_channels=3,
                 n_classes=24, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_metadata = list_metadata
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.DATASET_FOLDER = 'raw_dataset'
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_metadata) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_image_metadata = self.list_metadata[
            index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(batch_image_metadata)

        return X, y

    def on_epoch_end(self):
        'Suffle data if needed after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.list_metadata)

    def load_image(self, image_fname):
        img = cv2.imread(image_fname, cv2.IMREAD_COLOR)
        img = cv2.resize(
            img, (self.dim[0], self.dim[1]), interpolation=cv2.INTER_LINEAR)
        img = img / 255
        return img

    def make_fname(self, el):
        return os.path.join(self.DATASET_FOLDER,el[1], el[2], el[3])

    def __data_generation(self, batch_image_metadata):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i,el in enumerate(batch_image_metadata):
            # Store sample
            X[i, ] = self.load_image(self.make_fname(el))

            # Store class
            ID = int(el[0])
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
