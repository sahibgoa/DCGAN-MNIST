"""
Creates the GAN using the generator and discriminator and trains it until the
discriminator cannot distinguish between the generated and real data.
"""

import config
import dataset
import discriminator
import generator


def main():
    # Create a dataset
    data = dataset.Dataset()
    # Create a GAN
    gan = GAN()
    # Train on the GAN
    gan.train(data)


class GAN(object):

    def __init__(self):
        # Create the discriminator
        self.discriminator = discriminator.Discriminator()
        # Create the generator
        self.generator = generator.Generator(self.discriminator)

    def train(self, data: dataset.Dataset):
        # Train the discriminator initially with the given training data
        self.discriminator.train_network(data)
        i = 0
        # Train the GAN till the discriminator cannot distinguish generated images from real
        # images (in MAX_EPOCHS_NO_IMPROVEMENT tries)
        while self.discriminator.discriminate(self.generator.generate()) != 1 and i < config.MAX_EPOCHS_NO_IMPROVEMENT:
            self.generator.train_network()
            i += 1


if __name__ == "__main__":
    main()
