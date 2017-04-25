"""
Discriminates between generated and real images.
"""

from util import *

class Discriminator(object):

    def __init__(self, x_real, x_fake):
        # input
        self.x = tf.concat([x_real, x_fake], 0)

        # weights of each layer
        self.w1 = tf_gaussian([DIM_IM, DIM_H2], name='d_w1')
        self.w2 = tf_gaussian([DIM_H2, DIM_H1], name='d_w2')
        self.w3 = tf_gaussian([DIM_H1, 1], name='d_w3')

        # trainable parameters
        self.params = [self.w1, self.w2, self.w3]

        # layers
        self.h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(self.x, self.w1)), KEEP_PROB)
        self.h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(self.h1, self.w2)), KEEP_PROB)
        self.h3 = tf.matmul(self.h2, self.w3)

        # separate discrimination of real and fake data
        self.y_real = tf.nn.sigmoid(tf.slice(self.h3, [0, 0], [BATCH_SIZE, -1]))
        self.y_fake = tf.nn.sigmoid(tf.slice(self.h3, [BATCH_SIZE, 0], [-1, -1]))

        # loss
        self.loss = tf.reduce_mean(-(tf.log(self.y_real) + tf.log(1 - self.y_fake)))

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(ETA).minimize(self.loss, var_list=self.params)


    def discriminate(self):
        pass