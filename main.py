import os
import numpy as np
import math
from gan import *
from util import *
from tensorflow.examples.tutorials.mnist import input_data

def main():
    # load data
    mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)

    # make model
    gan = GAN()

    # train model
    gan.train(mnist)

if __name__ == '__main__':
    main()
