"""
Generates images like the training data.
"""

from util import *

class Generator(object):

    def __init__(self):
        # input
        self.z = tf.placeholder(tf.float32, [BATCH_SIZE, DIM_Z])

        # weights of each layer
        self.w1 = tf_gaussian([DIM_Z, DIM_H1], name='g_w1')
        self.b1 = tf_zeros([DIM_H1], name='g_b1')
        self.w2 = tf_gaussian([DIM_H1, DIM_H2], name='g_w2')
        self.b2 = tf_zeros([DIM_H2], name='g_b2')
        self.w3 = tf_gaussian([DIM_H2, DIM_IM], name='g_w3')
        self.b3 = tf_zeros([DIM_IM], name='g_b3')

        # trainable parameters
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

        # layers
        self.h1 = tf_relu(self.z, self.w1, self.b1)
        self.h2 = tf_relu(self.h1, self.w2, self.b2)
        self.h3 = tf_relu(self.h2, self.w3, self.b3)

        # output
        self.y_fake = tf.nn.tanh(self.h3)

        # loss
        self.loss = tf.reduce_mean(-tf.log(self.y_fake))

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(ETA).minimize(self.loss, var_list=self.params)
        