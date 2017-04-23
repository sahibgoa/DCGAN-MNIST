"""
Creates the GAN using the generator and discriminator and trains it until the
discriminator cannot distinguish between the generated and real data.
"""

from config import *
from dataset import *
from discriminator import *
from generator import *

class GAN(object):

    def __init__(self):
        # Create the discriminator
        self.discriminator = Discriminator()
        # Create the generator
        self.generator = Generator(self.discriminator)

    def train(self, data):
        # Train the discriminator initially with the given training data
        self.discriminator.train_network(data)
        i = 0
        # Train the GAN till the discriminator cannot distinguish generated images from real
        # images (in MAX_EPOCHS_NO_IMPROVEMENT tries)
        while self.discriminator.discriminate(self.generator.generate()) != 1 and i < config.MAX_EPOCHS_NO_IMPROVEMENT:
            self.generator.train_network()
            i += 1
