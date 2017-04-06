# somewhat based on https://www.tensorflow.org/get_started/mnist/pros
# discriminates between photographs and sketches with high accuracy
# author: David Liang (dliangsta)

import os
import numpy as np
import scipy.io as sio
import scipy
import tensorflow as tf
import random

# parameters, could probably use an arg parser in the future
DIRECTORIES = ['photo', 'sketch'] # directories of photos and sketches
USE_RGB = True # results are better when using grey only (set USE_RGB to false)
NUM_CHANNELS = 3 if USE_RGB else 1 # number of color channels, 3 for RGB, 1 for grayscale
NUM_TO_KEEP = 15 # keep NUM_TO_KEEP images from each type (max 25), decrease if python complains about RAM/memory
INITIAL_LEARNING_RATE = 1e-4 # learning rate for optimizer
DROPOUT_RATE = 0.5 # drop out rate for learning
TRAIN_PROPORTION = .9 # proportion of iamges to keep as train, 1 - TRAIN_PROPORTION kept for test
NUM_CLASSES = 2 # photos and sketches
MAX_EPOCHS = 100000 # maximal number of epochs to run
ACCURACY_FREQUENCY = 10 # how often to print accuracy
BATCH_SIZE = 50 # how many instances per batch
IM_SIZE_X = 128 # width of image to downscale to
IM_SIZE_Y = 128 # height of image to downscale to

def main():
    # load train and test sets
    dataset = Dataset()
    # create discriminator
    discriminator = Discriminator()
    # train
    discriminator.train_network(dataset)

class Discriminator:

    def __init__(self):
        # placeholders for input, output
        self.x = tf.placeholder(tf.float32, shape=[None, np.prod(IM_SIZE_X * IM_SIZE_Y * NUM_CHANNELS)])
        self.y = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
        self.x_image = tf.reshape(self.x, [-1, IM_SIZE_Y, IM_SIZE_X, NUM_CHANNELS])

        # convolutions, biases, activations, and pool of 1st layer
        self.W_conv1 = self.__weight_variable([5, 5, NUM_CHANNELS, 32]) 
        self.b_conv1 = self.__bias_variable([32])
        self.h_conv1 = tf.nn.relu(self.__conv2d(self.x_image, self.W_conv1) + self.b_conv1) # 128 x 128
        self.h_pool1 = self.__max_pool_2x2(self.h_conv1) # 64 x 64

        # convolutions, biases, activations, and pool of 2nd layer
        self.W_conv2 = self.__weight_variable([5, 5, 32, 64]) 
        self.b_conv2 = self.__bias_variable([64])
        self.h_conv2 = tf.nn.relu(self.__conv2d(self.h_pool1, self.W_conv2) + self.b_conv2) # 64 x 64
        self.h_pool2 = self.__max_pool_2x2(self.h_conv2) # 32 x 32

        # convolutions, biases, activations, and pool of 3rd layer
        self.W_conv3 = self.__weight_variable([5, 5, 64, 64])
        self.b_conv3 = self.__bias_variable([64])
        self.h_conv3 = tf.nn.relu(self.__conv2d(self.h_pool2, self.W_conv3) + self.b_conv3) # 32 x 32
        self.h_pool3 = self.__max_pool_2x2(self.h_conv3) # 16 x 16

        # convolutions, biases, activations, and pool of 3rd layer
        self.W_conv4 = self.__weight_variable([5, 5, 64, 64])
        self.b_conv4 = self.__bias_variable([64])
        self.h_conv4 = tf.nn.relu(self.__conv2d(self.h_pool3, self.W_conv4) + self.b_conv4) # 16 x 16
        self.h_pool4 = self.__max_pool_2x2(self.h_conv4) # 8 x 8

        # flattening of h_pool4's output
        self.h_pool4_flat = tf.reshape(self.h_pool4, [-1, 8 * 8 * 64])

        # first fully-connected layer, biases, and activation
        self.W_fc1 = self.__weight_variable([8 * 8 * 64, IM_SIZE_X * IM_SIZE_Y])
        self.b_fc1 = self.__bias_variable([IM_SIZE_X * IM_SIZE_Y])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool4_flat, self.W_fc1) + self.b_fc1)

        # dropout rate
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        # second fully-connected layer and biases
        self.W_fc2 = self.__weight_variable([IM_SIZE_X * IM_SIZE_Y,NUM_CLASSES])
        self.b_fc2 = self.__bias_variable([NUM_CLASSES])

        # final output layer
        self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

        # cross entropy
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_conv))

        # make a step using the adam optimizer
        self.train_step = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(self.cross_entropy)

        # correct predictions
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y, 1))

        # accuracy
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train_network(self, dataset):
        self.dataset = dataset

        # launch tensorflow session
        with tf.Session() as sess:
            # intialize tensorflwo variables
            sess.run(tf.global_variables_initializer())

            # run training epochs
            for i in range(MAX_EPOCHS):
                # get next training batch
                batch = dataset.next_batch(BATCH_SIZE)

                # run training step
                self.train_step.run(feed_dict={self.x: batch[0], self.y: batch[1], self.keep_prob: DROPOUT_RATE})
                
                # print accuracy every so often
                if i % ACCURACY_FREQUENCY == 0:
                    # get test data
                    test_images, test_labels = dataset.get_test()

                    # print accuracy
                    print("step %d: test accuracy %g" % (i, self.accuracy.eval(feed_dict={
                        self.x: test_images, 
                        self.y: test_labels, 
                        self.keep_prob: 1.0})))

    def __weight_variable(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def __conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def __max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def __bias_variable(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

class Dataset:

    def __init__(self):
        self.image_class_indices = [0, 0]

        all_images = [[],[]]
        
        self.total_count = 0

        # not a one liner because keeping track of count :(
        for directory in DIRECTORIES:
            for class_name in os.listdir(directory):
                count = 0
                for file_name in os.listdir(os.path.join(directory,class_name)):
                    if count < NUM_TO_KEEP:
                        all_images[directory == DIRECTORIES[1]].append(self.__get_flatten_image(os.path.join(os.path.join(directory, class_name), file_name)))
                        self.total_count += 1
                    count += 1

        self.train_count = 0
        self.test_count = 0
        # train and test, each with photos and sketches
        self.train_images = [[],[]]
        self.test_images = [[],[]]

        for i in range(NUM_CLASSES):
            num_train = len(all_images[i]) * TRAIN_PROPORTION

            # randomly select images for train
            while len(self.train_images[i]) < num_train:
                self.train_images[i].append(all_images[i].pop(random.randint(0, len(all_images[i]) - 1)))
                self.train_count += 1

            # insert the remaining into test
            while len(all_images[i]) > 0:
                self.test_images[i].append(all_images[i].pop(0))
                self.test_count += 1

        print("%d total images loaded, %d in train and %d in test" % (self.total_count, self.train_count, self.test_count))

    def __get_flatten_image(self, file_name):
        # read in image and flatten
        if USE_RGB:
            return np.ndarray.flatten(
                np.rollaxis(
                    scipy.misc.imresize(
                        scipy.misc.imread(
                            file_name, False), (IM_SIZE_X, IM_SIZE_Y)), 2, 0))
        else:
             return np.ndarray.flatten(
                    scipy.misc.imresize(
                        scipy.misc.imread(
                            file_name, True), (IM_SIZE_X, IM_SIZE_Y)))

    def next_batch(self, n):
        imgs = []
        labels = []

        for i in range(n):
            # 0 for photo, 1 for sketch
            index = random.randint(0,1) 
            
            # shuffle train when end reached
            if self.image_class_indices[index] >= len(self.train_images[index]):
                random.shuffle(self.train_images[index])
                self.image_class_indices[index] = 0

            imgs.append(self.train_images[index][self.image_class_indices[index]])
            labels.append([index == 0, index == 1])
            
            self.image_class_indices[index] = self.image_class_indices[index] + 1

        return (imgs, labels)

    def get_test(self):
        imgs = []
        labels = []
        for i in range(NUM_CLASSES):
            for j in range(len(self.test_images[i])):
                imgs.append(self.test_images[i][j])
                labels.append([i == 0, i == 1])
        return (imgs, labels)


if __name__ == "__main__":
    main()
