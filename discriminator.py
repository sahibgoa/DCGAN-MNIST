"""
somewhat based on https://www.tensorflow.org/get_started/mnist/pros
discriminates between photographs and sketches
"""

from config import *
from dataset import *
from util import *
import numpy as np
import tensorflow as tf


class Discriminator(object):

    def __init__(self):
        # placeholders for input, output
        self.x = tf.placeholder(tf.float32, shape=[None, np.prod(IM_SIZE_X * IM_SIZE_Y * NUM_CHANNELS)])
        self.y = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
        self.x_image = tf.reshape(self.x, [-1, IM_SIZE_Y, IM_SIZE_X, NUM_CHANNELS])

        # convolutions, biases, activations, and pool of 1st layer
        self.W_conv1 = weight_variable([5, 5, NUM_CHANNELS, 32])
        self.b_conv1 = bias_variable([32])
        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)  # 64 x 64
        self.h_pool1 = max_pool_2x2(self.h_conv1)  # 32 x 32

        # convolutions, biases, activations, and pool of 2nd layer
        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)  # 32 x 32
        self.h_pool2 = max_pool_2x2(self.h_conv2)  # 16 x 16

        # convolutions, biases, activations, and pool of 3rd layer
        self.W_conv3 = weight_variable([5, 5, 64, 64])
        self.b_conv3 = bias_variable([64])
        self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self.W_conv3) + self.b_conv3)  # 16 x 16
        self.h_pool3 = max_pool_2x2(self.h_conv3)  # 8 x 8

        # flattening of h_pool3's output
        self.h_pool3_flat = tf.reshape(self.h_pool3, [-1, 8 * 8 * 64])

        # first fully-connected layer, biases, and activation
        self.W_fc1 = weight_variable([8 * 8 * 64, IM_SIZE_X * IM_SIZE_Y])
        self.b_fc1 = bias_variable([IM_SIZE_X * IM_SIZE_Y])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool3_flat, self.W_fc1) + self.b_fc1)

        # dropout rate
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        # second fully-connected layer and biases
        self.W_fc2 = weight_variable([IM_SIZE_X * IM_SIZE_Y, NUM_CLASSES])
        self.b_fc2 = bias_variable([NUM_CLASSES])

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

    def train_network(self, data):
        # launch tensorflow session
        with tf.Session() as sess:
            # initialize tensorflow variables
            sess.run(tf.global_variables_initializer())

            # run training epochs
            for i in range(MAX_EPOCHS):
                # get next training batch
                batch = data.next_batch(BATCH_SIZE)

                # run training step
                self.train_step.run(feed_dict={self.x: batch[0], self.y: batch[1], self.keep_prob: DROPOUT_RATE})

                # print accuracy every so often
                if i % ACCURACY_FREQUENCY == 0:
                    # get test data
                    test_images, test_labels = data.get_test()

                    # print accuracy
                    print("step %d: test accuracy %g" % (i, self.accuracy.eval(feed_dict={
                        self.x: test_images,
                        self.y: test_labels,
                        self.keep_prob: 1.0
                    })))

    # Return 1 if it thinks that the image is from the real dataset and 0 if it thinks its generated
    def discriminate(self, image) -> int:
        pass