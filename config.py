"""
 Configuration file for the GAN
"""

DIRECTORIES = ['photo', 'sketch']  # directories of photos and sketches
USE_RGB = True  # results are better when using grey only (set USE_RGB to false)
NUM_CHANNELS = 3 if USE_RGB else 1  # number of color channels, 3 for RGB, 1 for grayscale
NUM_TO_KEEP = 20  # keep NUM_TO_KEEP images from each type (max 25), decrease if python complains about RAM/memory
INITIAL_LEARNING_RATE = 1e-4  # learning rate for optimizer
DROPOUT_RATE = 0.5  # drop out rate for learning
TRAIN_PROPORTION = .9  # proportion of images to keep as train, 1 - TRAIN_PROPORTION kept for test
NUM_CLASSES = 2  # photos and sketches
MAX_EPOCHS = 100000  # maximal number of epochs to run
MAX_EPOCHS_NO_IMPROVEMENT = 1000  # maximal number of epochs to run while generator is not improving
ACCURACY_FREQUENCY = 10  # how often to print accuracy
BATCH_SIZE = 25  # how many instances per batch
IM_SIZE_X = 64  # width of image to downscale to
IM_SIZE_Y = 64  # height of image to downscale to