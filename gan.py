"""
GAN network that handles the training of the generator and discriminator.
"""

from util import *
from discriminator import *
from generator import *

class GAN():
    def __init__(self):
        # generator and discriminator
        self.g = Generator()
        self.d = Discriminator(self.g.x_fake)

        # losses
        self.d_loss = tf.reduce_mean(-(tf.log(self.d.y_real) + tf.log(1 - self.d.y_fake)))
        self.g_loss = tf.reduce_mean(-tf.log(self.d.y_fake))

        # optimizers
        self.d_optimizer = tf.train.AdamOptimizer(ETA, beta1=BETA1).minimize(self.d_loss, var_list=self.d.params)
        self.g_optimizer = tf.train.AdamOptimizer(ETA, beta1=BETA1).minimize(self.g_loss, var_list=self.g.params)


    def train(self, data):
        with tf.Session() as sess:
            # initialize globals
            sess.run(tf.global_variables_initializer())
            clear()

            # same random numbers for all saved samples
            z_sample = np.random.normal(0, 1, size=[1, DIM_Z])

            # train
            for epoch in range(MAX_EPOCHS):
                # average losses
                avg_d_loss = avg_g_loss = 0

                # minibatch
                for iteration in range(TRAIN_SIZE // BATCH_SIZE):
                    # next MNIST batch
                    x_real, _= data.train.next_batch(BATCH_SIZE)
                    # because most values are 0
                    x_real = 2 * (x_real - 0.5)
                    
                    # random input to generator
                    z = np.random.normal(0, 1, size=[BATCH_SIZE, DIM_Z])

                    # run optimizer ops
                    _, _, d_loss_curr, g_loss_curr = sess.run([self.d_optimizer, self.g_optimizer, self.d_loss, self.g_loss], feed_dict={self.d.x_real:x_real, self.g.z:z})

                    avg_d_loss += d_loss_curr
                    avg_g_loss += g_loss_curr


                print(epoch, ": ", avg_d_loss / (TRAIN_SIZE // BATCH_SIZE + 1), avg_g_loss / (TRAIN_SIZE // BATCH_SIZE + 1))
                sample = sess.run(self.g.x_fake, feed_dict={self.g.z:z_sample})
                save_sample(sample, SAVE_PATH % epoch)
