import argparse

import numpy as np
import tensorflow as tf
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
from six.moves import xrange

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SHAPE = [28, 28]

# Hyper-Parameters
LEAK = 0.2
KEEP_PROB = 0.5
DIM_Z = 100
DIM_Y = 10
DIM_IMAGE = np.prod(IMAGE_SHAPE)


class DataDistribution(object):
    """
    Defines the real data distribution, which is the MNIST dataset
    """

    def __init__(self, data_dir):
        self.data = input_data.read_data_sets(data_dir, one_hot=True)

    def images(self, batch_size):
        # TODO apply this tweak => 2 * (x_real - 0.5)
        images, _ = self.images_and_labels(batch_size)
        return images

    def images_and_labels(self, batch_size):
        """
        :return: batch of samples of specified size from the MNIST training data
        """
        return self.data.train.next_batch(batch_size)

    def num_training_examples(self):
        return self.data.train.num_examples


# TODO try MNIST fake data
class GeneratorDistribution(object):
    """
    Define the generator's input noise distribution using stratified sampling - the samples are first generated
    uniformly over a specified range, and then randomly perturbed.

    This better aligns the input space with the target space and makes the transformation as smooth as possible
    and easier to learn. Stratified sampling also increases the representativeness the entire training space.
    """

    def __init__(self):
        self.sample_dim = DIM_Z

    def image_samples(self, batch_size):
        """
        :return: batch of samples of specified size, each with a noise vector
        """
        return np.random.normal(0, 1, size=[batch_size, self.sample_dim])


def weight_variable(shape, mean=0.0, std_dev=0.1):
    return tf.get_variable(
        'w',
        dtype=tf.float32,
        initializer=tf.truncated_normal(shape=shape, mean=mean, stddev=std_dev),
        trainable=True)


def bias_variable(shape, ):
    return tf.get_variable(
        'b',
        dtype=tf.float32,
        initializer=tf.zeros(shape),
        trainable=True)


# TODO remove in-dimension
def linear(x, dim_in, dim_out, scope):
    with tf.variable_scope(scope or 'linear'):
        w = weight_variable([dim_in, dim_out])
        b = bias_variable([dim_out])
    return tf.matmul(x, w) + b


# TODO remove in-dimension
def relu(x, dim_in, dim_out, scope, leaky=False):
    with tf.variable_scope(scope or 'relu'):
        w = weight_variable([dim_in, dim_out])
        b = bias_variable([dim_out])
    if leaky:
        return tf.nn.relu(tf.matmul(x, w) + b)
    else:
        leak = LEAK
        x_ = tf.matmul(x, w)
        l1 = 0.5 * (1 + leak)
        l2 = 0.5 * (1 - leak)
        return l1 * x_ + l2 * tf.abs(x_)


def generator(z, y, batch_size, dim_con=64, dim_fc=1024):
    """
    Args:
        z: input noise tensor, float - [batch_size, DIM_Z=100]
        y: input label tensor, float - [batch_size, DIM_Y=10]
        batch_size:
        dim_con:
        dim_fc:
    Returns:
        x': the generated image tensor, float - [batch_size, DIM_IMAGE=784]
    """
    # TODO add nn.dropout layer
    with tf.variable_scope('generator'):
        s_h, s_w = 28, 28
        s_h2, s_h4 = 14, 7
        s_w2, s_w4 = 14, 7

        y_4d = tf.reshape(y, [batch_size, 1, 1, DIM_Y])
        z_ = tf.concat([z, y], 1)

        h1_ = relu(z_, DIM_Z + DIM_Y, dim_fc, 'h1_fc')
        h1 = tf.concat([h1_, y], 1)

        h2_ = relu(h1, h1.get_shape().as_list()[-1], dim_con * 2 * s_h4 * s_w4, 'h2_fc')
        h2_4d = tf.reshape(h2_, [batch_size, s_h4, s_w4, dim_con * 2])
        h1 = concat(h2_4d, y_4d)

        h2_4d = tf.nn.relu(deconv2d(h1, [batch_size, s_h2, s_w2, dim_con * 2], 'h3_con'))
        h2 = concat(h2_4d, y_4d)

        return tf.nn.sigmoid(deconv2d(h2, [batch_size, s_h, s_w, 1], 'x'))

def sampler(z, y, batch_size=10, dim_con=64, dim_fc=1024):
    """
    Args:
        z: input noise tensor, float - [batch_size, DIM_Z=100]
        y: input label tensor, float - [batch_size, DIM_Y=10]
        batch_size:
        dim_con:
        dim_fc:
    Returns:
        x': the generated image tensor, float - [batch_size, DIM_IMAGE=784]
    """
    # TODO add nn.dropout layer
    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()

        s_h, s_w = 28, 28
        s_h2, s_h4 = 14, 7
        s_w2, s_w4 = 14, 7

        y_4d = tf.reshape(y, [batch_size, 1, 1, DIM_Y])
        z_ = tf.concat([z, y], 1)

        h1_ = relu(z_, DIM_Z + DIM_Y, dim_fc, 'h1_fc')
        h1 = tf.concat([h1_, y], 1)

        h2_ = relu(h1, h1.get_shape().as_list()[-1], dim_con * 2 * s_h4 * s_w4, 'h2_fc')
        h2_4d = tf.reshape(h2_, [batch_size, s_h4, s_w4, dim_con * 2])
        h1 = concat(h2_4d, y_4d)

        h2_4d = tf.nn.relu(deconv2d(h1, [batch_size, s_h2, s_w2, dim_con * 2], 'h3_con'))
        h2 = concat(h2_4d, y_4d)

        return tf.nn.sigmoid(deconv2d(h2, [batch_size, s_h, s_w, 1], 'x'))

# TODO implement minibatch
def discriminator(x, y, batch_size, dim_con=64, dim_fc=1024, reuse=False):
    """
    """
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        y_4d = tf.reshape(y, [batch_size, 1, 1, DIM_Y])
        x_4d = tf.reshape(x, [batch_size, 28, 28, 1])
        x_ = concat(x_4d, y_4d)

        h1_ = lrelu(conv2d(x_, 1 + DIM_Y, 'h1_conv2'))
        h1 = concat(h1_, y_4d)

        # TODO add batch-normalization
        h2_4d = lrelu(conv2d(h1, dim_con + DIM_Y, 'h2_conv2'))
        h2_ = tf.reshape(h2_4d, [batch_size, -1])
        h2 = tf.concat([h2_, y], 1)

        h3_ = tf.nn.dropout(relu(h2, h2.get_shape().as_list()[-1], dim_fc, 'h3_fc1', leaky=True), KEEP_PROB)
        h3 = tf.concat([h3_, y], 1)
        # h4 = tf.nn.dropout(relu(h3, dim_fc, dim_fc, 'h4_fc', leaky=True), KEEP_PROB)

        logits = linear(h3, dim_fc + DIM_Y, 1, 'y')

        return tf.nn.sigmoid(logits), logits

def concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, scope=None, filter_height=5, filter_width=5, stride=2):
    with tf.variable_scope(scope or 'conv2'):
        w = weight_variable([filter_height, filter_width, input_.get_shape().as_list()[-1], output_dim])
        b = bias_variable([output_dim])
        strides = [1, stride, stride, 1]

        conv2 = tf.nn.conv2d(input_, w, strides, padding='SAME')

        return tf.reshape(tf.nn.bias_add(conv2, b), conv2.get_shape())

def deconv2d(input_, output_dim, scope=None, filter_height=5, filter_width=5, stride=2):
    with tf.variable_scope(scope or 'deconv2'):
        w = weight_variable([filter_height, filter_width, output_dim[-1], input_.get_shape().as_list()[-1]])
        b = bias_variable([output_dim[-1]])
        strides = [1, stride, stride, 1]

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_dim, strides=strides)

        return tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class GAN(object):
    def __init__(self, data, gen, batch_size, epoch_size, learning_rate, decay_rate, num_pre_train_steps, out):
        self.data = data
        self.gen = gen
        self.num_epoch = epoch_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.num_pre_train_steps = num_pre_train_steps
        self.out = out
        self._create_model()

    def _create_model(self):
        """
        Build a GAN model that include two competing neural network models G and D (combination of D1 and D2)
        """

        self.y = tf.placeholder(tf.float32, [None, DIM_Y])

        # placeholder for samples from a noise distribution
        self.z = tf.placeholder(tf.float32, [None, DIM_Z])
        # the generator network takes noise and target label
        self.G = generator(self.z, self.y, self.batch_size)
        # a sampler network that generates demo samples for logging
        self.S = sampler(self.z, self.y)
        # placeholder for samples from the true data distribution
        self.x = tf.placeholder(tf.float32, [None, DIM_IMAGE])
        # the discriminator network predicting the likelihood of true data distribution
        self.D_real, self.logits_D_real = discriminator(self.x, self.y, self.batch_size)
        # the discriminator network predicting the likelihood of generated (fake) data distribution
        self.D_fake, self.logits_D_fake = discriminator(self.G, self.y, self.batch_size, reuse=True)

        # When optimizing D, we want to define it's loss function such that it
        # maximizes the quantity D1 (which maps the distribution of true data) and
        # minimizes D2 (which maps the distribution of fake data)
        self.loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits_D_real, labels=tf.ones_like(self.D_real)))
        self.loss_D_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits_D_fake, labels=tf.zeros_like(self.D_fake)))
        # TODO try tf.reduce_mean(d1 + (1 - d2))
        self.loss_d = self.loss_D_real + self.loss_D_fake
        self.params_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.opt_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.decay_rate) \
            .minimize(self.loss_d, var_list=self.params_d)

        # When optimizing G, we want to define it's loss function such that it
        # maximizes the quantity D2 (in order to successfully fool D)
        self.loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits_D_fake, labels=tf.ones_like(self.D_fake)))
        self.params_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.opt_g = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.decay_rate) \
            .minimize(self.loss_g, var_list=self.params_g)

    def train(self):
        """
        To train the model, we draw samples from the data distribution and the noise distribution,
        and alternate between optimizing the parameters of D and G.
        """
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())


            for epoch in xrange(self.num_epoch):
                loss_g_sum = loss_d_sum = 0
                num_steps = self.data.num_training_examples() // self.batch_size

                for step in xrange(num_steps):
                    # update discriminator
                    x, y = self.data.images_and_labels(self.batch_size)
                    z = self.gen.image_samples(self.batch_size)
                    loss_d, _ = session.run([self.loss_d, self.opt_d], feed_dict={
                        self.x: x,
                        self.y: y,
                        self.z: z
                    })
                    loss_d_sum += loss_d

                    # update generator
                    z = self.gen.image_samples(self.batch_size)
                    loss_g, _ = session.run([self.loss_g, self.opt_g], feed_dict={
                        self.y: y,
                        self.z: z
                    })
                    loss_g_sum += loss_g

                print('{}: avg_d {}\tavg_g {}'.format(epoch, loss_d_sum / num_steps, loss_g_sum / num_steps))
                samples = session.run(self.S, feed_dict={
                    self.y: [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                    ],
                    self.z: self.gen.image_samples(10)
                })
                for index in range(10):
                    imsave(self.out % (epoch, index), np.reshape(samples[index], IMAGE_SHAPE))


def main(args):
    model = GAN(
        DataDistribution(args.data_dir),
        GeneratorDistribution(),
        args.batch_size,
        args.epoch_size,
        args.learning_rate,
        args.decay_rate,
        args.num_pre_train_steps,
        args.out
    )
    model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--learning-rate', type=int, default=0.0002,
                        help='the learning rate for training')
    parser.add_argument('--decay-rate', type=int, default=0.5,
                        help='the learning rate for training')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--minibatch', type=bool, default=False,
                        help='use minibatch discrimination')
    parser.add_argument('--epoch-size', type=int, default=100,
                        help='size of each epoch')
    parser.add_argument('--num-pre-train-steps', type=int, default=1000,
                        help='number of pre-training steps')
    parser.add_argument('--out', type=str,
                        default='data/out%04d_%04d.jpg',
                        help='output location for writing samples from G')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())