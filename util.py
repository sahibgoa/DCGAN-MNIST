"""
Utility functions.
"""

import numpy as np
import tensorflow as tf
import os
from config import *
import scipy.misc

def tf_gaussian(shape, name=None, mean=0., stddev=STDDEV, xavier=USE_XAVIER):
    if xavier:
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    else:
        return tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=stddev), name=name)

def tf_zeros(shape, name=None):
    return tf.Variable(tf.zeros(shape), name=name)

def tf_relu(X, leaky=False):
    if leaky:
        X = ((1 + LEAK) / 2 * X) + ((1 - LEAK) / 2 * tf.abs(X))
    else:
        X = tf.nn.relu(X)
        
    return X

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def save_sample(batch, path):
    # make output directory
    if not os.path.exists(os.path.dirname(SAVE_PATH)):
        os.makedirs(os.path.dirname(SAVE_PATH))
    scipy.misc.imsave(path, np.reshape(batch/2 + 0.5, IMAGE_SHAPE))