"""
GAN network that handles the training of the generator and discriminator.
"""

from util import *
from discriminator import *
from generator import *

class GAN():
    def __init__(self):
        self.g = Generator()
        self.d = Discriminator()

    def train(self, data):
        with tf.Session() as sess:
            # initialize globals
            sess.run(tf.global_variables_initializer())
            clear()
            z_original = np.random.normal(0, 1, size=[BATCH_SIZE, DIM_Z])

            for epoch in range(MAX_EPOCHS):
                for iteration in range(TRAIN_SIZE // BATCH_SIZE):
                    # next MNIST batch
                    x_real, _= data.train.next_batch(BATCH_SIZE)
                    # random input to generator
                    z = np.random.normal(0, 1, size=[BATCH_SIZE, DIM_Z])

                    # get generator output
                    x_fake = sess.run(self.g.y_fake, feed_dict={self.g.z:z})
                    # run optimizer ops
                    _, d_loss = sess.run([self.d.optimizer, self.d.loss], feed_dict={self.d.x_real:x_real, self.d.x_fake:x_fake})
                    _, g_loss = sess.run([self.g.optimizer, self.g.loss], feed_dict={self.g.z:z})

                    print(d_loss, g_loss)
