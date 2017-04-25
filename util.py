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

def tf_relu(A, B):
    return tf.nn.relu(tf.matmul(A,B))

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')
