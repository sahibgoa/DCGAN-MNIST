"""
Creates the dataset of images for the discriminator to train on.
"""

import os
import random
import numpy as np
import scipy
import config


class Dataset:

    def __init__(self):
        # keeps track of which instance in the train and test sets to return next
        self.image_class_indices = [0, 0]

        # photos in the first list, sketches in the second
        all_images = [[], []]

        # count of number of images read in
        self.total_count = 0

        # not a one liner because keeping track of count :(
        for directory in config.DIRECTORIES:
            for class_name in os.listdir(directory):
                # keep track of how many images of this type are saved
                count = 0
                for file_name in os.listdir(os.path.join(directory, class_name)):
                    # limit the number of images of this type that are kept
                    if count < config.NUM_TO_KEEP:
                        # read in image, flatten
                        all_images[directory == config.DIRECTORIES[1]].append(
                            self.__get_flatten_image(os.path.join(os.path.join(directory, class_name), file_name)))
                        # increment counts
                        self.total_count += 1
                        count += 1

        # keep track of how many training and test images we create
        self.train_count = 0
        self.test_count = 0

        # train and test, each with photos and sketches
        self.train_images = [[], []]
        self.test_images = [[], []]

        # partition into train and test
        for i in range(config.NUM_CLASSES):
            # number of images to keep in train
            num_train = len(all_images[i]) * config.TRAIN_PROPORTION

            # randomly select images for train
            while len(self.train_images[i]) < num_train:
                self.train_images[i].append(all_images[i].pop(random.randint(0, len(all_images[i]) - 1)))
                self.train_count += 1

            # insert the remaining into test
            while len(all_images[i]) > 0:
                self.test_images[i].append(all_images[i].pop(0))
                self.test_count += 1

        print("%d total images loaded, %d in train and %d in test" % (
            self.total_count, self.train_count, self.test_count))

    def __get_flatten_image(self, file_name):
        # read in image, resize, flatten
        if config.USE_RGB:
            return np.ndarray.flatten(
                np.rollaxis(
                    scipy.misc.imresize(
                        scipy.misc.imread(
                            file_name, False), (config.IM_SIZE_X, config.IM_SIZE_Y)), 2, 0))
        else:
            return np.ndarray.flatten(
                scipy.misc.imresize(
                    scipy.misc.imread(
                        file_name, True), (config.IM_SIZE_X, config.IM_SIZE_Y)))

    def next_batch(self, n):
        # returns
        batch_images = []
        batch_labels = []

        # add n images and labels to the return lists
        for i in range(n):
            # 0 for photo, 1 for sketch
            index = random.randint(0, 1)

            # shuffle train when end reached
            if self.image_class_indices[index] >= len(self.train_images[index]):
                random.shuffle(self.train_images[index])
                self.image_class_indices[index] = 0

            batch_images.append(self.train_images[index][self.image_class_indices[index]])
            batch_labels.append([index == 0, index == 1])

            # increase index
            self.image_class_indices[index] += 1

        return batch_images, batch_labels

    def get_test(self):
        # returns
        test_images = []
        test_labels = []

        # add all the test image and labels to the return lists
        for i in range(config.NUM_CLASSES):
            for j in range(len(self.test_images[i])):
                test_images.append(self.test_images[i][j])
                test_labels.append([i == 0, i == 1])

        return test_images, test_labels
