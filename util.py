"""
Utility functions.
"""

import numpy as np
import tensorflow as tf
import os
from config import *
import scipy.misc

def tf_gaussian(shape, name=None, mean=0., stddev=STDDEV):
    return tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=stddev), name=name, dtype=tf.float32)

def tf_zeros(shape, name=None):
    return tf.Variable(tf.zeros(shape), name=name, dtype=tf.float32)

def tf_relu(A, B, C, leaky=False):
    if not leaky:
        return tf.nn.relu(tf.matmul(A,B) + C)
    else:
        X = tf.matmul(A,B)
        l1 = 0.5 * (1 + LEAK)
        l2 = 0.5 * (1 - LEAK)
        return l1 * X + l2 * tf.abs(X)

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def save_sample(batch, path):
    scipy.misc.imsave(path, np.reshape(batch/2 + 0.5, IMAGE_SHAPE))