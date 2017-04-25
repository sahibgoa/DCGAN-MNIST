from util import *
from discriminator import *
from generator import *

class GAN():
    def __init__(self):
        self.z = tf.placeholder(tf.float32, [BATCH_SIZE, DIM_Z])
        self.x_real = tf.placeholder(tf.float32, [BATCH_SIZE, DIM_IM])
        self.g = Generator(self.z)
        self.d = Discriminator(self.x_real, self.g.y_fake)

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

                    # run optimizer ops
                    _, d_loss = sess.run([self.d.optimizer, self.d.loss], feed_dict={self.x_real:x_real, self.z:z})
                    _, g_loss = sess.run([self.g.optimizer, self.g.loss], feed_dict={self.x_real:x_real, self.z:z})
                    print(d_loss, g_loss)
