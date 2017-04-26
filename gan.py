"""
GAN network that handles the training of the generator and discriminator.
"""

from util import *
from discriminator import *
from generator import *

class GAN():
    def __init__(self):
        self.g = Generator()
        self.d = Discriminator(self.g.y_fake)

        self.d_loss = tf.reduce_mean(-(tf.log(self.d.y_real) + tf.log(1 - self.d.y_fake)))
        self.d_optimizer = tf.train.AdamOptimizer(ETA).minimize(self.d_loss, var_list=self.d.params)

        self.g_loss = tf.reduce_mean(-tf.log(self.d.y_fake))
        self.g_optimizer = tf.train.AdamOptimizer(ETA).minimize(self.g_loss, var_list=self.g.params)


    def train(self, data):
        with tf.Session() as sess:
            # initialize globals
            sess.run(tf.global_variables_initializer())
            clear()

            # same random numbers for all saved samples
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
                    _, d_loss = sess.run([self.d_optimizer, self.d_loss], feed_dict={self.d.x_real:x_real, self.g.z:z})
                    _, g_loss = sess.run([self.g_optimizer, self.g_loss], feed_dict={self.d.x_real:x_real, self.g.z:z})

                    print(d_loss, g_loss)
