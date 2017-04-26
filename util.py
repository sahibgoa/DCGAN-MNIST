"""
Utility functions.
"""

import numpy as np
import tensorflow as tf
import os
import math
import scipy.misc
from config import *


def tf_gaussian(shape, name, mean=0., stddev=STDDEV):
    return tf.Variable(tf.random_normal(shape=shape, mean=mean, stddev=stddev), name=name)

def tf_zeros(shape, name):
    return tf.Variable(tf.zeros(shape), name=name)

def tf_relu(A, B, C):
    return tf.nn.relu(tf.matmul(A,B) + C)

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def save_sample(X, path):
    pass