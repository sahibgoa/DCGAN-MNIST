"""
Generates images like the training data.
"""

from util import *

class Generator(object):

    def __init__(self, z):
        # input
        self.z = z

        # weights of each layer
        self.w1 = tf_gaussian([DIM_Z, DIM_H1], name='g_w1')
        self.w2 = tf_gaussian([DIM_H1, DIM_H2], name='g_w2')
        self.w3 = tf_gaussian([DIM_H2, DIM_IM], name='g_w3')

        # trainable parameters
        self.params = [self.w1, self.w2, self.w3]

        # layers
        self.h1 = tf.nn.relu(tf.matmul(z, self.w1))
        self.h2 = tf.nn.relu(tf.matmul(self.h1, self.w2))
        self.h3 = tf.nn.relu(tf.matmul(self.h2, self.w3))

        # output
        self.y_fake = tf.nn.tanh(self.h3)

        # loss
        self.loss = -tf.log(self.y_fake)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(ETA).minimize(self.loss, var_list=self.params)
        
    def generate(self):
        pass
