"""
The generator for the GAN. It attempts to create images that the discriminator thinks are real images and 
not generated ones.
"""

import config
import discriminator
import numpy as np
import scipy
import scipy.io as sio
import tensorflow as tf
import Util


class Generator(object):

    def __init__(self, discrim: discriminator.Discriminator):
        pass
    # Uses the discriminator to get the outputs
    def train_network(self):
        pass

    def generate(self) -> object:
        pass
