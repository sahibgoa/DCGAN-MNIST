'''
GAN network that handles the training of the generator and discriminator.
'''

from util import *
from discriminator import *
from generator import *

class GAN():

    def __init__(self):
        '''
        Initializes the GAN by setting up the generator, discriminator, loss values, and optimizers.
        '''

        # generator and discriminator
        self.g = Generator()
        self.d = Discriminator(self.g.x_fake)

        # losses
        self.d_loss = tf.reduce_mean(-tf.log(self.d.y_real) - tf.log(1 - self.d.y_fake))
        self.g_loss = tf.reduce_mean(-tf.log(self.d.y_fake))
            
        if LAMBDA_L1:
            # L1 regularization
            self.d_loss += LAMBDA_L1 * tf.reduce_mean([tf.reduce_mean(tf.abs(x)) for x in self.d.params])
            self.g_loss += LAMBDA_L1 * tf.reduce_mean([tf.reduce_mean(tf.abs(x)) for x in self.g.params])

        if LAMBDA_L2:
            # L2 regularization
            self.d_loss += LAMBDA_L2 * tf.reduce_mean([tf.reduce_mean(tf.square(tf.abs(x))) for x in self.d.params])
            self.g_loss += LAMBDA_L2 * tf.reduce_mean([tf.reduce_mean(tf.square(tf.abs(x))) for x in self.g.params])

        # optimizers
        self.d_optimizer = tf.train.AdamOptimizer(ETA, beta1=BETA1).minimize(self.d_loss, var_list=self.d.params)
        self.g_optimizer = tf.train.AdamOptimizer(ETA, beta1=BETA1).minimize(self.g_loss, var_list=self.g.params)


    def train(self, data):
        '''
        Trains the GAN on the given dataset.
        '''

        # launch tf graph
        with tf.Session() as sess:
            # initialize globals
            sess.run(tf.global_variables_initializer())

            # same random numbers for all saved samples
            z_sample = np.random.normal(RANDOM_MEAN, RANDOM_STDDEV, size=[SAMPLE_SIZE, DIM_Z])

            clear()
            print("training beginning!\n")

            # train
            for epoch in range(MAX_EPOCHS):
                # average losses
                avg_d_loss = avg_g_loss = 0

                iterations = TRAIN_SIZE // BATCH_SIZE
                
                # minibatch
                for iteration in range(iterations):
                    # next MNIST batch, adjust for most values being 0
                    x_real, _= data.train.next_batch(BATCH_SIZE)
                    x_real = 2 * (x_real - 0.5)

                    # random input to generator
                    z = np.random.normal(RANDOM_MEAN, RANDOM_STDDEV, size=[BATCH_SIZE, DIM_Z])
                    
                    # feed dict for all ops
                    feed_dict = {self.d.x_real:x_real, self.g.z:z}
                    
                    # train discriminator and then generator
                    sess.run(self.d_optimizer, feed_dict)
                    sess.run(self.g_optimizer, feed_dict)
                    
                    # get losses and increment totals
                    d_loss_curr, g_loss_curr = sess.run([self.d_loss, self.g_loss], feed_dict)
                    avg_d_loss += d_loss_curr
                    avg_g_loss += g_loss_curr

                # print out average losses and epoch
                print('epoch: %03d: d=%01.9f g=%01.9f' % (epoch, avg_d_loss / iterations, avg_g_loss / iterations))

                # save sample
                sample = sess.run(self.g.x_fake, feed_dict={self.g.z:z_sample})
                save_sample(sample, SAVE_PATH + 'sample_%04d.jpg' % epoch)
