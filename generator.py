"""
The generator for the GAN. It attempts to create images that the discriminator thinks are real images and 
not generated ones.
"""

import config
import discriminator
import numpy as np
import scipy
import scipy.io as sio
import tensorflow as tf
import Util


class Generator(object):

    def __init__(self, discrim: discriminator.Discriminator):
        # The discriminator used to train the generator
        self.discriminator = discrim

        # placeholders for input, output
        self.x = tf.placeholder(tf.float32,
                                shape=[None, np.prod(config.IM_SIZE_X * config.IM_SIZE_Y * config.NUM_CHANNELS)])
        self.y = tf.placeholder(tf.float32, shape=[None, config.NUM_CLASSES])
        self.x_image = tf.reshape(self.x, [-1, config.IM_SIZE_Y, config.IM_SIZE_X, config.NUM_CHANNELS])

        # convolutions, biases, activations, and pool of 1st layer
        self.W_conv1 = Util.weight_variable([5, 5, config.NUM_CHANNELS, 32])
        self.b_conv1 = Util.bias_variable([32])
        self.h_conv1 = tf.nn.relu(Util.conv2d(self.x_image, self.W_conv1) + self.b_conv1)  # 128 x 128
        self.h_pool1 = Util.max_pool_2x2(self.h_conv1)  # 64 x 64

        # convolutions, biases, activations, and pool of 2nd layer
        self.W_conv2 = Util.weight_variable([5, 5, 32, 64])
        self.b_conv2 = Util.bias_variable([64])
        self.h_conv2 = tf.nn.relu(Util.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)  # 64 x 64
        self.h_pool2 = Util.max_pool_2x2(self.h_conv2)  # 32 x 32

        # convolutions, biases, activations, and pool of 3rd layer
        self.W_conv3 = Util.weight_variable([5, 5, 64, 64])
        self.b_conv3 = Util.bias_variable([64])
        self.h_conv3 = tf.nn.relu(Util.conv2d(self.h_pool2, self.W_conv3) + self.b_conv3)  # 32 x 32
        self.h_pool3 = Util.max_pool_2x2(self.h_conv3)  # 16 x 16

        # convolutions, biases, activations, and pool of 4th layer
        self.W_conv4 = Util.weight_variable([5, 5, 64, 64])
        self.b_conv4 = Util.bias_variable([64])
        self.h_conv4 = tf.nn.relu(Util.conv2d(self.h_pool3, self.W_conv4) + self.b_conv4)  # 16 x 16
        self.h_pool4 = Util.max_pool_2x2(self.h_conv4)  # 8 x 8

        # flattening of h_pool4's output
        self.h_pool4_flat = tf.reshape(self.h_pool4, [-1, 8 * 8 * 64])

        # first fully-connected layer, biases, and activation
        self.W_fc1 = Util.weight_variable([8 * 8 * 64, config.IM_SIZE_X * config.IM_SIZE_Y])
        self.b_fc1 = Util.bias_variable([config.IM_SIZE_X * config.IM_SIZE_Y])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool4_flat, self.W_fc1) + self.b_fc1)

        # dropout rate
        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        # second fully-connected layer and biases
        self.W_fc2 = Util.weight_variable([config.IM_SIZE_X * config.IM_SIZE_Y, config.NUM_CLASSES])
        self.b_fc2 = Util.bias_variable([config.NUM_CLASSES])

        # final output layer
        self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

        # cross entropy
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_conv))

        # make a step using the adam optimizer
        self.train_step = tf.train.AdamOptimizer(config.INITIAL_LEARNING_RATE).minimize(self.cross_entropy)

        # correct predictions
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y, 1))

        # accuracy
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    # Uses the discriminator to get the outputs
    def train_network(self):
        pass

    def generate(self) -> object:
        pass
