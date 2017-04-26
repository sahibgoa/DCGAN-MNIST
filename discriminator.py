"""
Discriminates between generated and real images.
"""

from util import *

class Discriminator(object):

    def __init__(self, fake):
        # inputs
        self.x_real = tf.placeholder(tf.float32, [None, DIM_IM])
        self.x_fake = fake
        
        # combine input for one pass
        self.x = tf.concat([self.x_real, self.x_fake], axis=0)

        # weights of each layer
        self.w1 = tf_gaussian([DIM_IM, DIM_H3])
        self.w2 = tf_gaussian([DIM_H3, DIM_H2])
        self.w3 = tf_gaussian([DIM_H2, DIM_H1])
        self.w4 = tf_gaussian([DIM_H1, 1])

        # weights of biases
        self.b1 = tf_zeros([DIM_H3])
        self.b2 = tf_zeros([DIM_H2])
        self.b3 = tf_zeros([DIM_H1])
        self.b4 = tf_zeros([1])

        # layers
        self.h1 = tf.nn.dropout(tf_relu(self.x, self.w1, self.b1), KEEP_PROB)
        self.h2 = tf.nn.dropout(tf_relu(self.h1, self.w2, self.b2), KEEP_PROB)
        self.h3 = tf.nn.dropout(tf_relu(self.h2, self.w3, self.b3), KEEP_PROB)
        self.h4 = tf.matmul(self.h3, self.w4) + self.b4

        # separate discrimination of real and fake data
        self.y_real = tf.nn.sigmoid(tf.slice(self.h4, [0, 0], [BATCH_SIZE, -1]))
        self.y_fake = tf.nn.sigmoid(tf.slice(self.h4, [BATCH_SIZE, 0], [-1, -1]))

        # trainable parameters
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4]

