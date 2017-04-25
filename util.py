"""
Utility functions
"""

import numpy as np
import tensorflow as tf
import os
import math
import scipy.misc
from config import *


def tf_gaussian(shape, name, mean=0., stddev=STDDEV):
    return tf.Variable(tf.random_normal(shape=shape, mean=mean, stddev=stddev), name=name)

def save_sample(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], IMAGE_SHAPE[0], IMAGE_SHAPE[1])) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    scipy.misc.imsave(fname, img_grid)

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')
