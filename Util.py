"""
All utility functions present here since both generator and discriminator require them
"""

import tensorflow as tf


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
