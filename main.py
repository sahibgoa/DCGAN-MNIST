from config import *
from dataset import *
from discriminator import *
from generator import *
from gan import *

def main():
    # Create a dataset
    data = Dataset()
    # Create a GAN
    gan = GAN()
    # Train on the GAN
    gan.train(data)

if __name__ == "__main__":
    main()
    