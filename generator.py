"""
Generates images like the training data.
"""

from util import *

class Generator(object):

    def __init__(self):
        # input
        self.z = tf.placeholder(tf.float32, [None, DIM_Z])

        # weights of each layer
        self.w1 = tf_gaussian([DIM_Z, DIM_H1])
        self.w2 = tf_gaussian([DIM_H1, DIM_H2])
        self.w3 = tf_gaussian([DIM_H2, DIM_IM])

        # weights of biases
        self.b1 = tf_zeros([DIM_H1])
        self.b2 = tf_zeros([DIM_H2])
        self.b3 = tf_zeros([DIM_IM])

        # layers
        self.h1 = tf_relu(self.z, self.w1, self.b1)
        self.h2 = tf_relu(self.h1, self.w2, self.b2)
        self.h3 = tf.matmul(self.h2, self.w3) + self.b3

        # output
        self.x_fake = tf.nn.tanh(self.h3)

        # trainable parameters
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
