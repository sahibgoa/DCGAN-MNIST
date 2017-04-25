import os
import numpy as np
from gan import *
from util import *
import math

def main():
    # load data
    mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True))

    # make model
    gan = GAN()

    # train model
    gan.train(mnist)

if __name__ == '__main__':
    main()
