"""
Utility functions
"""

import numpy as np
import tensorflow as tf
import os
from config import *


def tf_gaussian(shape, name, mean=0., stddev=STDDEV):
    return tf.Variable(tf.random_normal(shape=shape, mean=mean, stddev=stddev), name=name)

def save_sample(X, save_path):
    sample_x = sample_y = math.ceil(math.sqrt(BATCH_SIZE))
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h * sample_x, w * sample_y, 3))

    for n,x in enumerate(X):
        j = n // sample_y
        i = n % sample_y
        img[j*h:j*h+h, i*w:i*w+w, :] = x

    scipy.misc.imsave(save_path, img)

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')
