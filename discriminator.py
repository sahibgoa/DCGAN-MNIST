from util import *

class Discriminator(object):

    def __init__(self, x_real, x_fake, keep_prob):
        # input
        self.x = tf.concat([x_real, x_fake], 0)

        # weights of each layer
        self.w1 = tf_gaussian([DIM_IM, DIM_H2], name='d_w1')
        self.w2 = tf_gaussian([DIM_H2, DIM_H1], name='d_w2')
        self.w3 = tf_gaussian([DIM_H1, 1], name='d_w3')

        # layers
        self.h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(self.x, self.w1)), keep_prob)
        self.h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(self.h1, self.w2)), keep_prob)
        self.h3 = tf.matmul(self.h2, self.w3)

        # separate discrimination of real and fake data
        self.y_real = tf.nn.sigmoid(tf.slice(self.h3, [0, 0], [BATCH_SIZE, -1]))
        self.y_fake = tf.nn.sigmoid(tf.slice(self.h3, [BATCH_SIZE, 0], [-1, -1]))

    def discriminate(self, x, y):
        pass