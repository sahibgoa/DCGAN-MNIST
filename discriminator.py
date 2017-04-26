"""
Discriminates between generated and real images.
"""

from util import *

class Discriminator(object):

    def __init__(self, x_fake):
        # inputs
        self.x_real = tf.placeholder(tf.float32, [None, DIM_IM])
        self.x_fake = x_fake
        
        # combine input for one pass
        self.x = tf.concat([self.x_real, self.x_fake], axis=0)

        # weights of each layer
        self.w1 = tf_gaussian([DIM_IM, DIM_H2])
        self.w2 = tf_gaussian([DIM_H2, DIM_H1])
        self.w3 = tf_gaussian([DIM_H1, 1])

        # weights of biases
        self.b1 = tf_zeros([DIM_H2])
        self.b2 = tf_zeros([DIM_H1])
        self.b3 = tf_zeros([1])

        # layers
        self.h1 = tf.nn.dropout(tf_relu(self.x,  self.w1, self.b1, leaky=True), KEEP_PROB)
        self.h2 = tf.nn.dropout(tf_relu(self.h1, self.w2, self.b2, leaky=True), KEEP_PROB)
        self.h3 = tf.nn.sigmoid(tf.matmul(self.h2, self.w3) + self.b3)

        # separate discrimination of real and fake data
        self.y_real = tf.slice(self.h3, [0, 0], [BATCH_SIZE, -1])
        self.y_fake = tf.slice(self.h3, [BATCH_SIZE, 0], [-1, -1])

        # trainable parameters
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
