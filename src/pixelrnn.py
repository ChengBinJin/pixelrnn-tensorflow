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

        self.grad_clip = 1.
        if flags.dataset == 'mnist':
            self.hidden_dims = 16
            self.recurrent_length = 7
            self.out_recurrent_length = 2

        self._init_logger()  # init logger
        self._build_net()    # init graph
        self._tensorboard()  # init tensorboard

    def _init_logger(self):
        if self.flags.is_train:
            tf_utils._init_logger(self.log_path)

    def _build_net(self):
        self.input_tf = tf.placeholder(tf.float32, shape=[None, *self.img_size], name='input_img')

        # pixelrnn network
        self.output, self.output_logits = self.network(self.input_tf, name='pixelcnn')

        # loss
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_logits,
                                                                           labels=self.input_tf))
        optimizer = tf.train.RMSPropOptimizer(self.flags.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        new_grads_and_vars = [(tf.clip_by_value(gv[0], -self.grad_clip, self.grad_clip), gv[1])
                              for gv in grads_and_vars]
        self.train_op = optimizer.apply_gradients(new_grads_and_vars)

    def _tensorboard(self):
        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()

    def network(self, inputs, name=None):
        with tf.variable_scope(name):
            tf_utils.print_activations(inputs)

            # input of main reccurent layers
            output = self.conv2d_mask(inputs, 2*self.hidden_dims, [7, 7], mask_type="A", name='inputConv1')

            # main recurrent layers
            if self.flags.model == 'pixelcnn':
                for idx in range(self.recurrent_length):
                    output = self.conv2d_mask(output, self.hidden_dims, [3, 3], mask_type="B",
                                              name='mainConv{}'.format(idx+2))
                    output = tf_utils.relu(output, name='mainRelu{}'.format(idx+2))

            elif self.flags.model == 'diagonal_bilstm':
                for idx in range(self.recurrent_length):
                    output = self.diagonal_bilstm(output, name='BiLSTM{}'.format(idx+2))

            else:
                raise NotImplementedError

            # output recurrent layers
            for idx in range(self.out_recurrent_length):
                output = self.conv2d_mask(output, self.hidden_dims, [1, 1], mask_type="B",
                                          name='outputConv{}'.format(idx + 1))
                output = tf_utils.relu(output, name='outputRelu{}'.format(idx + 1))

            # TODO: for color images, implement a 256-way softmax for each RGB channel here
            output = self.conv2d_mask(output, self.img_size[2], [1, 1], mask_type="B", name='outputConv3')
            # output = tf_utils.sigmoid(output_logits, name='output_sigmoid')

            return tf_utils.sigmoid(output), output

    def train_step(self, imgs):
        _, loss, summary = self.sess.run([self.train_op, self.loss, self.summary_op],
                                               feed_dict={self.input_tf: imgs})
        return [loss], summary

    def test_step(self, imgs):
        test_summary = self.sess.run(self.summary_op, feed_dict={self.input_tf: imgs})
        return test_summary

    def sample_imgs(self):
        samples = np.zeros((self.flags.sample_batch, *self.img_size), dtype=np.float32)

        for i in range(self.img_size[0]):
            for j in range(self.img_size[1]):
                for k in range(self.img_size[2]):
                    next_sample = utils.binarize(self.sess.run(self.output, feed_dict={self.input_tf: samples}))
                    samples[:, i, j, k] = next_sample[:, i, j, k]

        return samples

    def print_info(self, loss, iter_batch, iter_epoch, tar_batch):
        if np.mod(iter_batch, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('cur_batch', iter_batch), ('tar_batch', tar_batch),
                                                  ('cur_epoch', iter_epoch), ('tar_epochs', self.flags.epochs),
                                                  ('batch_size', self.flags.batch_size),
                                                  ('total_loss', loss[0]),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_batch, ord_output)

    def plots(self, imgs, iter_time, save_file):
        # reshape image from vector to (N, H, W, C)
        if self.img_size[2] == 1:
            imgs_fake = np.reshape(imgs, (-1, self.img_size[0], self.img_size[1]))
        else:
            imgs_fake = np.reshape(imgs, (-1, *self.img_size))

        h_imgs, w_imgs = int(np.sqrt(imgs_fake.shape[0])), int(np.sqrt(imgs_fake.shape[0]))
        imsave(os.path.join(save_file, '{}.png'.format(str(iter_time).zfill(3))),
               utils._merge(imgs_fake, size=[h_imgs, w_imgs], resize_ratio=1.))

    @staticmethod
    def conv2d_mask(inputs, output_dim, kernel_shape, mask_type, strides=[1, 1], padding="SAME",
                    name="conv2d", is_print=True):
        # kernel_shape: [kernel_height, kernel_width]
        # mask_type: None, "A", "
        # B"
        # strides: [1, 1] [column_wise_stride, row_wise_stride]
        with tf.variable_scope(name):
            batch, height, width, channel = inputs.get_shape().as_list()

            k_h, k_w = kernel_shape
            assert k_h % 2 == 1 and k_w % 2 == 1, "kernel height and width should be odd number"
            # Kaiming He weight initialization
            weights = tf.get_variable('weights', [k_h, k_w, inputs.get_shape()[-1], output_dim],
                                      initializer=tf.contrib.layers.xavier_initializer())

            mask = np.ones((k_h, k_w, channel, output_dim), dtype=np.float32)
            mask[int(k_h // 2), int(k_w // 2)+1:, :, :] = 0.
            mask[int(k_h // 2)+1:, :, :, :] = 0.

            mask_type = mask_type.lower()
            if mask_type == 'a':
                mask[int(k_h // 2), int(k_w // 2), :, :] = 0.

            # weights.assign(weights * tf.constant(mask, dtype=tf.float32))
            weights *= tf.constant(mask, dtype=tf.float32)
            outputs = tf.nn.conv2d(inputs, weights, [1, strides[0], strides[1], 1], padding=padding, name='outputs')

            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            outputs = tf.nn.bias_add(outputs, biases)

            if is_print:
                tf_utils.print_activations(outputs)

            return outputs

    def diagonal_bilstm(self, inputs, name='BiLSTM'):
        with tf.variable_scope(name):
            output_state_fw = self.diagonal_lstm(inputs, name='output_state_fw')
            output_state_bw = self.reverse(self.diagonal_lstm(self.reverse(inputs), name='output_state_bw'))

            # output = self.conv2d_mask(inputs, self.hidden_dims, [7, 7], mask_type="A", name='inputConv1')
            # Residual connection part
            residual_state_fw = self.conv2d_mask(output_state_fw, 2*self.hidden_dims, [1, 1], mask_type="B",
                                                 name='residual_fw')
            output_state_fw = residual_state_fw + inputs

            residual_state_bw = self.conv2d_mask(output_state_bw, 2*self.hidden_dims, [1, 1], mask_type="B",
                                                 name='residual_bw')
            output_state_bw = residual_state_bw + inputs



            return 0

    def diagonal_lstm(self, inputs, name='LSTM'):
        print('Hello diagonal_lstm!')

        return 0

    @staticmethod
    def reverse(inputs, name='Reverse'):
        with tf.variable_scope(name):
            reverse_inputs = tf.reverse(inputs, axis=[2])  # [False, False, True, False]
            return reverse_inputs
