# ---------------------------------------------------------
# TensorFlow PixelRNN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import logging
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from scipy.misc import imsave

import tensorflow_utils as tf_utils
import utils as utils

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


class PixelRNN(object):
    def __init__(self, sess, flags, dataset, log_path=None):
        self.sess = sess
        self.flags = flags
        self.dataset = dataset
        self.img_size = dataset.img_size
        self.log_path = log_path

        self.hidden_dims = 0
        if flags.dataset == 'mnist':
            self.hidden_dims = 32

        self._init_logger()  # init logger
        self._build_net()    # init graph
        self._tensorboard()  # init tensorboard

    def _init_logger(self):
        if self.flags.is_train:
            tf_utils._init_logger(self.log_path)

    def _build_net(self):
        self.input_tf = tf.placeholder(tf.float32, shape=[None, *self.img_size], name='input_img')
        self.output = self.network(self.input_tf, name='pixelcnn')

        # conv2d(self.l['normalized_inputs'], conf.hidden_dims * 2, [7, 7], "A", scope=scope)


    def network(self, inputs, name='pixelrnn'):
        with tf.variable_scope(name):
            tf_utils.print_activations(inputs)

            output = self.conv2d_mask(inputs, self.hidden_dims, [7, 7], "A", name='conv1')

    def conv2d_mask(self, inputs, output_dim, kernel_shape,  mask_type, strides=[1, 1], padding="SAME",
                    name="conv2d", is_print=True):
        # kernel_shape: [kernel_height, kernel_width]
        # mask_type: None, "A", "B"
        # strides: [1, 1] [column_wise_stride, row_wise_stride]
        with tf.variable_scope(name):
            batch, height, width, channel = inputs.get_shape().as_list()

            k_h, k_w = kernel_shape
            assert k_h % 2 == 1 and k_w % 2 == 1, "kernel height and width should be odd number"
            # Kaiming He weight initialization
            weights = tf.get_variable('weights', [k_h, k_w, inputs.get_shape()[-1], output_dim],
                                      initializer=tf.contrib.layers.variance_scaling_initializer())
                                    # initializer=tf.truncated_normal_initializer(stddev=stddev))

            mask = np.ones((k_h, k_w, channel, output_dim), dtype=np.float32)
            mask[int(k_h // 2), int(k_w // 2)+1:, :, :] = 0.
            mask[int(k_h // 2):, :, :, :] = 0.

            mask_type = mask_type.lower()
            if mask_type == 'a':
                mask[int(k_h // 2), int(k_w // 2), :, :] = 0.

            weights *= tf.constant(mask, dtype=tf.float32)
            outputs = tf.nn.conv2d(inputs, weights, [1, strides[0], strides[1], 1], padding=padding, name='outputs')

            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            outputs = tf.nn.bias_add(outputs, biases)

            if is_print:
                tf_utils.print_activations(outputs)

            return outputs


