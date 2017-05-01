import argparse

import numpy as np
import tensorflow as tf
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
from six.moves import xrange
import os

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_SHAPE = [28, 28]

# Hyper-Parameters
LEAK = 0.2
KEEP_PROB = 0.5
DIM_Z = 100
DIM_Y = NUM_CLASSES
DIM_IMAGE = np.prod(IMAGE_SHAPE)

SAVE_DIR = 'out/'


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



def linear(x_input, dim_in, dim_out, name='linear'):
    """
    Builds a fully connected layer of neurons and returns their activations as computed by a
    matrix multiplication followed by a bias offset.

    Args:
        x_input:
        dim_in:
        dim_out:
        name:

    Returns:

    """
    with tf.variable_scope(name):
        w = weight_variable([dim_in, dim_out])
        b = bias_variable([dim_out])

        return tf.matmul(x_input, w) + b


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(x, .2*x)


def concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def weight_variable(shape, name='w', mean=0.0, std_dev=0.1):
    """
    Returns a trainable weight variable that is randomly initialized from a normal distribution.

    Args:
        shape: shape of the weight variable
        name: optional name for the variable as a string
        mean: mean of the random values to generate, a python scalar or a scalar tensor
        std_dev: standard deviation of the random values to generate, a python scalar or a scalar tensor

    Returns:
        A newly created or existing variable.
    """
    return tf.get_variable(
        name=name,
        shape=shape,
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(mean=mean, stddev=std_dev),
        trainable=True)


def bias_variable(shape, name='b'):
    """
    Returns a trainable bias variable that is initialized with constant values of 0.

    Args:
        shape: shape of the bias variable
        name: optional name for the variable as a string

    Returns:
        A newly created or existing variable.
    """
    return tf.get_variable(
        name=name,
        shape=shape,
        dtype=tf.float32,
        initializer=tf.constant_initializer(0),
        trainable=True)


def deconv2d(input_, output_dim,
             name='deconv2',
             filter_height=5,
             filter_width=5,
             strides=[1, 2, 2, 1]):
    """
    Transpose (gradient) of the tf.nn.conv2d operation.

    Args:
         input_:
         output_dim:
         name:
         filter_height:
         filter_width:
         strides:

    Returns:
        A tensor ...
    """
    bias_shape = [output_dim[-1]]
    in_channels = output_dim[-1]
    out_channels = input_.get_shape().as_list()[-1]
    filter_shape = [filter_height, filter_width, in_channels, out_channels]

    with tf.variable_scope(name):
        f_w = weight_variable(filter_shape)
        b = bias_variable(bias_shape)

        return tf.nn.bias_add(
            tf.nn.conv2d_transpose(input_, filter=f_w, output_shape=output_dim, strides=strides), b)


def cnn_block(x_image, out_channels,
              name='cnn_block',
              filter_height=5,
              filter_width=5,
              conv_stride=[1, 1, 1, 1],
              ksize=[1, 2, 2, 1],
              pool_stride=[1, 2, 2, 1]):
    """
    Block of three key operations that form the basic building blocks of every
    Convolutional Neural Network (CNN)
        1. Convolution (conv2d)
        2. Non Linearity (ReLU)
        3. Pooling or Sub-Sampling (avg_pool)

    Args:
        x_image: input images as a matrix of pixel values, float-32 - [batch, in_height, in_width, in_channels]
        out_channels:
        name:
        filter_height:
        filter_width:
        conv_stride:
        ksize:
        pool_stride:

    Returns:
        A Tensor with the same type as value. The convoluted-rectified-average_pooled output tensor.
    """
    bias_shape = [out_channels]
    in_channels = x_image.get_shape().as_list()[-1]
    filter_shape = [filter_height, filter_width, in_channels, out_channels]

    with tf.variable_scope(name):
        f_w = weight_variable(filter_shape)
        b = bias_variable(bias_shape)
        # slide the filter over the image to build a feature map
        feat_map = tf.nn.bias_add(
            tf.nn.conv2d(input=x_image, filter=f_w, strides=conv_stride, padding='SAME'), b)
        # ReLU is applied on each pixel to introduce non-linearity
        rect_feat_map = lrelu(feat_map)
        # average pooling used to reduce dimensionality
        sub_sample = tf.nn.avg_pool(rect_feat_map, ksize=ksize, strides=pool_stride, padding='SAME')

        return sub_sample


def generator(z_input, y_label,
              batch_size=10,
              dim_con=64,
              dim_fc=1024,
              reuse=False):
    """
    Args:
        z_input: input noise tensor, float - [batch_size, DIM_Z=100]
        y_label: input label tensor, float - [batch_size, DIM_Y=10]
        batch_size:
        dim_con:
        dim_fc:
        reuse:
    Returns:
        x': the generated image tensor, float - [batch_size, DIM_IMAGE=784]
    """
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()

        # create z as the joint representation of the input noise and the label
        z = tf.concat([z_input, y_label], 1)

        # first fully-connected layer
        g1 = tf.nn.relu(tf.contrib.layers.batch_norm(linear(
            x_input=z,
            dim_in=DIM_Z + DIM_Y,
            dim_out=dim_fc,
            name='g1'), epsilon=1e-5, scope='g1_bn'))

        # join the output of the previous layer with the labels vector
        g1 = tf.concat([g1, y_label], 1)

        # second fully-connected layer
        g2 = tf.nn.relu(tf.contrib.layers.batch_norm(linear(
            x_input=g1,
            dim_in=g1.get_shape().as_list()[-1],
            dim_out=dim_con * 2 * IMAGE_SIZE / 4 * IMAGE_SIZE / 4,
            name='g2'), epsilon=1e-5, scope='g2_bn'))

        # create a joint 4-D feature representation of the output of the previous layer and the label
        # to serve as a 7x7 input image for the next de-convolution layer
        y_ = tf.reshape(y_label, [batch_size, 1, 1, DIM_Y])
        g2 = tf.reshape(g2, [batch_size, IMAGE_SIZE // 4, IMAGE_SIZE // 4, dim_con * 2])
        g2 = concat(g2, y_)

        # first layer of deconvolution produces a larger 14x14 image
        g3 = deconv2d(g2, [batch_size, IMAGE_SIZE // 2, IMAGE_SIZE // 2, dim_con * 2], 'g3')

        # apply batch normalization to ___
        # apply ReLU to stabilize the output of this layer
        g3 = tf.nn.relu(tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='g3_bn'))

        # join the output of the previous layer with the labels vector
        g3 = concat(g3, y_)

        # second layer of deconvolution produces the final sized 28x28 image
        g4 = deconv2d(g3, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1], 'x')

        # no batch normalization in the final layer but a sigmoid activation function is used to
        # generate a sharp and crisp image vector; dimension - [28, 28, 1]
        return tf.nn.sigmoid(g4)


def discriminator(x_image, y_label, batch_size,
                  dim_con=64,
                  dim_fc=1024,
                  reuse=False):
    """
    Returns the discriminator network. It takes an image and returns a real/fake classification across each label.
    The discriminator network is structured as a Convolution Neural Net with two layers of convolution and pooling,
    followed by two fully-connected layers.

    Args:
        x_image:
        y_label:
        batch_size:
        dim_con:
        dim_fc:
        reuse:

    Returns:
        The discriminator network.
    """
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        # create x as the joint 4-D feature representation of the image and the label
        y_4d = tf.reshape(y_label, [batch_size, 1, 1, DIM_Y])
        x_4d = tf.reshape(x_image, [batch_size, 28, 28, 1])
        x = concat(x_4d, y_4d)

        # first convolution-activation-pooling layer
        d1 = cnn_block(x, 1 + DIM_Y, 'd1')

        # join the output of the previous layer with the labels vector
        d1 = concat(d1, y_4d)

        # second convolution-activation-pooling layer
        d2 = cnn_block(d1, dim_con + DIM_Y, 'd2')

        # flatten the output of the second layer to a 2-D matrix with shape - [batch, ?]
        d2 = tf.reshape(d2, [batch_size, -1])

        # join the flattened output with the labels vector and apply this as input to
        # a series of fully connected layers.
        d2 = tf.concat([d2, y_label], 1)

        # first fully connected layer
        d3 = tf.nn.dropout(lrelu(linear(
            x_input=d2,
            dim_in=d2.get_shape().as_list()[-1],
            dim_out=dim_fc,
            name='d3')), KEEP_PROB)

        # join the output of the previous layer with the labels vector
        d3 = tf.concat([d3, y_label], 1)

        # second and last fully connected layer
        d4 = linear(d3, dim_fc + DIM_Y, 1, 'd4')

        # return the activation values, dimension - [batch, 1]; and
        # the un-normalized log probability of each label
        return tf.nn.sigmoid(d4), d4



class DCGAN(object):
    def __init__(self, data, gen, batch_size, epoch_size, learning_rate, decay_rate, out):
        self.data = data
        self.gen = gen
        self.num_epoch = epoch_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
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
        # the sample generator network that generates demo samples for logging
        self.S = generator(self.z, self.y, reuse=True)
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
                    self.y: np.identity(NUM_CLASSES),
                    self.z: self.gen.image_samples(NUM_CLASSES)
                })
                save_samples(self.out, samples, epoch)



def main(args):
    # set params for dataset
    global NUM_CLASSES
    global IMAGE_SIZE
    global IMAGE_SHAPE
    global DIM_Y
    global DIM_IMAGE

    if ~args.use_mnist:
        # 125 types of images
        NUM_CLASSES = 125

        # downsized images are 128 x 128
        IMAGE_SIZE = 128
        IMAGE_SHAPE = [128, 128]

        DIM_Y = NUM_CLASSES
        DIM_IMAGE = np.prod(IMAGE_SHAPE)

    # build dcgan
    model = DCGAN(
        DataDistribution(args.data_dir),
        GeneratorDistribution(),
        args.batch_size,
        args.epoch_size,
        args.learning_rate,
        args.decay_rate,
        args.out
    )

    # train dcgan
    model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--use_mnist', action="store_true",
                        help='true if using MNIST dataset, false if using sketchy dataset')
    parser.add_argument('--learning-rate', type=int, default=0.0002,
                        help='the learning rate for training')
    parser.add_argument('--decay-rate', type=int, default=0.5,
                        help='the learning rate for training')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--epoch-size', type=int, default=100,
                        help='size of each epoch')
    parser.add_argument('--out', type=str,
                        default=SAVE_DIR,
                        help='output location for writing samples from G')
    return parser.parse_args()


def save_samples(directory, samples, epoch):
    # make save directories if needed
    if not os.path.exists(directory + 'combined/'):
        os.makedirs(directory + 'combined/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    for index in range(NUM_CLASSES):
        digit_directory = directory + str(index) + '/'
        if not os.path.exists(digit_directory):
            os.makedirs(digit_directory)

    # save an image for each digit
    for index in range(NUM_CLASSES):
        imsave(directory + '%d/digit_%d_%03d.jpg' % (index, index, epoch), np.reshape(samples[index], IMAGE_SHAPE))

    # save one image with all ten samples, stacked vertically
    imsave(directory + 'combined/combined_%03d.jpg' % epoch, np.reshape(samples, np.multiply(IMAGE_SHAPE,[NUM_CLASSES,1])))


if __name__ == '__main__':
    main(parse_args())
