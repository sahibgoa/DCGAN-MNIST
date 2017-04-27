'''
Generates images like the training data.
'''

from util import *

class Generator(object):

    def __init__(self):
        '''
        Creates the weights and biases for each hidden layer as well as the output of the generator.
        '''
        
        # input
        self.z = tf.placeholder(tf.float32, [None, DIM_Z])

        # weights of each layer
        self.w1 = tf_gaussian([DIM_Z, DIM_H1], name='g_w1')
        self.w2 = tf_gaussian([DIM_H1, DIM_H2], name='g_w2')
        self.w3 = tf_gaussian([DIM_H2, DIM_IM], name='g_w3')

        # weights of biases
        self.b1 = tf_zeros([DIM_H1], name='g_b1')
        self.b2 = tf_zeros([DIM_H2], name='g_b2')
        self.b3 = tf_zeros([DIM_IM], name='g_b3')

        # layers
        self.h1 = tf_relu(tf.matmul(self.z, self.w1) + self.b1)
        self.h2 = tf_relu(tf.matmul(self.h1, self.w2) + self.b2)
        self.h3 = tf.nn.tanh(tf.matmul(self.h2, self.w3) + self.b3)

        # output
        self.x_fake = self.h3

        # trainable parameters
        self.params = [self.w1, self.w2, self.w3]

        # if using bias
        if USE_BIAS:
            self.params = np.concatenate([self.params, [self.b1, self.b2, self.b3]])
