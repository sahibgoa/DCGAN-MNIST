import os
import numpy as np
from gan import *
from util import *
import math

def main():
    # load data
    train_data, train_labels = load()

    # make model
    gan = GAN()

    # train model
    gan.train(train_data, train_labels)

def load():
    # read training data
    loaded = np.fromfile(file=open(os.path.join(DATA_PATH,'train-images-idx3-ubyte')),dtype=np.uint8)
    train_data = loaded[16:].reshape((60000,IMAGE_SHAPE[0]*IMAGE_SHAPE[1])).astype(float)

    loaded = np.fromfile(file=open(os.path.join(DATA_PATH,'train-labels-idx1-ubyte')),dtype=np.uint8)
    train_labels = loaded[8:].reshape((60000))
    train_labels = np.asarray(train_labels)

    return train_data, train_labels

if __name__ == '__main__':
    main()
