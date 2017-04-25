"""
The generator for the GAN. It attempts to create images that the discriminator thinks are real images and 
not generated ones.
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

        # layers
        self.h1 = tf.nn.relu(tf.matmul(z, self.w1))
        self.h2 = tf.nn.relu(tf.matmul(self.h1, self.w2))
        self.h3 = tf.nn.relu(tf.matmul(self.h2, self.w3))
        self.x = tf.nn.tanh(self.h3)
        
    def generate(self, Z, Y):
        pass
