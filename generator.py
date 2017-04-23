"""
The generator for the GAN. It attempts to create images that the discriminator thinks are real images and 
not generated ones.
"""

import numpy as np
import scipy
import scipy.io as sio
import tensorflow as tf
from config import *
from dataset import *
from discriminator import *
from generator import *
from util import *



class Generator(object):

    def __init__(self, discrim):
        pass
    # Uses the discriminator to get the outputs
    def train_network(self):
        pass

    def generate(self):
        pass
