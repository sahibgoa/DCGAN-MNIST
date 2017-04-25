"""
All utility functions present here since both generator and discriminator require them
"""

import numpy as np
import tensorflow as tf
from config import *


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def tf_gaussian(shape, name, mean=0., stddev=STDDEV):
    return tf.Variable(tf.random_normal(shape=shape, mean=mean, stddev=stddev), name=name)

def encode_one_hot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh

def save_sample(X, save_path):
    sample_x = sample_y = math.ceil(math.sqrt(BATCH_SIZE))
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h * sample_x, w * sample_y, 3))

    for n,x in enumerate(X):
        j = n // sample_y
        i = n % sample_y
        img[j*h:j*h+h, i*w:i*w+w, :] = x

    scipy.misc.imsave(save_path, img)

def batchnormalize(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0, 1, 2])
        std = tf.reduce_mean(tf.square(X - mean), [0 , 1 ,2])
        X = (X - mean) / tf.sqrt(std + eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1, 1, 1, -1])
            b = tf.reshape(b, [1, 1, 1, -1])
            X = X*g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1, -1])
            b = tf.reshape(b, [1, -1])
            X = X*g + b

    return X

def cross_entropy(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=o, labels=t))
