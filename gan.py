from util import *
from discriminator import *
from generator import *

class GAN():
    def __init__(self):
        self.keep_prob = KEEP_PROB
        self.z = tf.placeholder(tf.float32, [BATCH_SIZE, DIM_Z])
        self.x_real = tf.placeholder(tf.float32, [BATCH_SIZE, DIM_IM])
        self.x_fake = tf.placeholder(tf.float32, [BATCH_SIZE, DIM_IM])
        self.d = Discriminator(self.x_real, self.x_fake, self.keep_prob)
        self.g = Generator(self.z)

    def generate_batch(self, batch_size):
        pass

    def train(self, data):
        pass