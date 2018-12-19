# ---------------------------------------------------------
# Tensorflow PixelRNN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import logging
import numpy as np
import tensorflow as tf

import utils as utils

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


def _init_logger(flags, log_path):
    if flags.is_train:
        formatter = logging.Formatter('%(asctime)s:%(name)s: %(message)s')
        # file handler
        file_handler = logging.FileHandler(os.path.join(log_path, 'dataset.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        # add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)


class MnistDataset(object):
    def __init__(self, sess, flags, dataset_name):
        self.sess = sess
        self.flags = flags
        self.dataset_name = dataset_name
        self.img_size = (28, 28, 1)
        self.img_buffle = 100000  # image buffer for image shuffling
        self.num_trains, self.num_tests = 0, 0

        self.mnist_path = os.path.join('../../Data', self.dataset_name)
        self._load_mnist()

    def _load_mnist(self):
        logger.info('Load {} dataset...'.format(self.dataset_name))
        self.train_data, self.test_data = tf.keras.datasets.mnist.load_data()
        self.num_trains = self.train_data[0].shape[0]
        self.num_tests = self.test_data[0].shape[0]

        # TensorFlow Dataset API
        train_x, train_y = self.train_data
        test_x, test_y = self.test_data

        # training data
        train_dataset = tf.data.Dataset.from_tensor_slices(({'image': train_x}, train_y))
        train_dataset = train_dataset.shuffle(self.img_buffle).repeat().batch(self.flags.batch_size)
        train_dataset = train_dataset.prefetch(10)  # prefetch
        train_iterator = train_dataset.make_one_shot_iterator()
        self.next_batch_train = train_iterator.get_next()

        # test data
        test_dataset = tf.data.Dataset.from_tensor_slices(({'image': test_x}, test_y))
        test_dataset = test_dataset.shuffle(self.img_buffle).repeat(1).batch(self.flags.batch_size)
        test_dataset = test_dataset.prefetch(10)  # prefetch
        test_iterator = test_dataset.make_one_shot_iterator()
        self.next_batch_test = test_iterator.get_next()

        self.train_step_per_epoch = int(self.num_trains / self.flags.batch_size)
        self.test_step_per_epoch = int(self.num_tests / self.flags.batch_size)

        logger.info('Load {} datast SUCCESS!'.format(self.dataset_name))
        logger.info('Image size: {}'.format(self.img_size))
        logger.info('Num. of training data: {}'.format(self.num_trains))

    def train_next_batch(self):
        batch_data = self.sess.run(self.next_batch_train)
        batch_imgs = batch_data[0]["image"]
        batch_labels = batch_data[1]

        imgs_array = np.reshape(batch_imgs, [self.flags.batch_size, *self.img_size])
        imgs_array = imgs_array / 255.
        imgs_array = utils.binarize(imgs_array)  # input of the pixelrnn for mnist should be binarized data

        # one-hot representations
        labels_array = np.zeros((batch_labels.shape[0], 10))
        labels_array[range(batch_labels.shape[0]), batch_labels] = 1

        return imgs_array, labels_array

    def test_next_batch(self):
        # idxs = np.random.randint(low=0, high=self.num_tests, size=self.flags.batch_size)
        # batch_imgs, batch_labels = self.test_x[idxs], self.test_y[idxs]
        batch_data = self.sess.run(self.next_batch_test)
        batch_imgs = batch_data[0]["image"]
        batch_labels = batch_data[1]

        imgs_array = np.reshape(batch_imgs, [self.flags.batch_size, *self.img_size])
        imgs_array = imgs_array / 255.
        imgs_array = utils.binarize(imgs_array)  # input of the pixelrnn for mnist should be binarized data

        # one-hot representations
        labels_array = np.zeros((batch_labels.shape[0], 10))
        labels_array[range(batch_labels.shape[0]), batch_labels] = 1

        return imgs_array, labels_array


class Cifar10(object):
    def __init__(self, flags, dataset_name):
        self.flags = flags
        self.dataset_name = dataset_name
        self.img_size = (32, 32, 3)
        self.num_trains = 0

        self.cifar10_path = os.path.join('../../Data', self.dataset_name)
        self._load_cifar10()

    def _load_cifar10(self):
        import cifar10

        cifar10.data_path = self.cifar10_path
        logger.info('Load {} dataset...'.format(self.dataset_name))

        # The CIFAR-10 dataset is about 13MB and will be downloaded automatically if it is not
        # located in teh given path
        cifar10.maybe_download_and_extract()

        self.train_data, _, _ = cifar10.load_training_data()
        self.num_trains = self.train_data.shape[0]

        logger.info('Load {} dataset SUCCESS!'.format(self.dataset_name))
        logger.info('Img size: {}'.format(self.img_size))
        logger.info('Num. of training data: {}'.format(self.num_trains))

    def train_next_batch(self):
        batch_imgs = self.train_data[np.random.choice(self.num_trains, self.flags.batch_size, replace=False)]
        imgs_array = np.reshape(batch_imgs, (self.flags.batch_size, *self.img_size))
        imgs_array = imgs_array * 2. - 1.  # from [0. 1.] to [-1., 1.]
        labels_array = np.asarray([])

        return imgs_array, labels_array


# noinspection PyPep8Naming
def Dataset(sess, flags, dataset_name, log_path=None):
    if flags.is_train:
        _init_logger(flags, log_path)  # init logger

    if dataset_name == 'mnist':
        return MnistDataset(sess, flags, dataset_name)
    elif dataset_name == 'cifar10':
        return Cifar10(flags, dataset_name)
    else:
        raise NotImplementedError
