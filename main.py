'''
Loads the MNIST data, builds the GAN, and trains it.
'''

from tensorflow.examples.tutorials.mnist import input_data
from gan import *

def main():
    # load data
    data = input_data.read_data_sets('data/mnist/', one_hot=True)

    # make model
    gan = GAN()

    # train model
    gan.train(data)

if __name__ == '__main__':
    main()
