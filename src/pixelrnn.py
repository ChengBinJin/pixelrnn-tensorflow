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
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
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
        self.output, self.output_logits = self.network(self.input_tf, name=self.flags.model)

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
            output = tf_utils.conv2d_mask(inputs, 2*self.hidden_dims, [7, 7], mask_type="A", name='inputConv1')

            # main recurrent layers
            if self.flags.model == 'pixelcnn':
                for idx in range(self.recurrent_length):
                    output = tf_utils.conv2d_mask(output, self.hidden_dims, [3, 3], mask_type="B",
                                                  name='mainConv{}'.format(idx+2))
                    output = tf_utils.relu(output, name='mainRelu{}'.format(idx+2))

            elif self.flags.model == 'diagonal_bilstm':
                for idx in range(self.recurrent_length):
                    output = self.diagonal_bilstm(output, name='BiLSTM{}'.format(idx+2))

            else:
                raise NotImplementedError

            # output recurrent layers
            for idx in range(self.out_recurrent_length):
                output = tf_utils.conv2d_mask(output, self.hidden_dims, [1, 1], mask_type="B",
                                          name='outputConv{}'.format(idx + 1))
                output = tf_utils.relu(output, name='outputRelu{}'.format(idx + 1))

            # TODO: for color images, implement a 256-way softmax for each RGB channel here
            output = tf_utils.conv2d_mask(output, self.img_size[2], [1, 1], mask_type="B", name='outputConv3')
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

    def diagonal_bilstm(self, inputs, name='diagonal_bilstm'):
        with tf.variable_scope(name):
            output_state_fw = self.diagonal_lstm(inputs, name='output_state_fw')
            output_state_bw = tf_utils.reverse(self.diagonal_lstm(tf_utils.reverse(inputs), name='output_state_bw'))

            # Residual connection part
            residual_state_fw = tf_utils.conv2d_mask(output_state_fw, 2*self.hidden_dims, [1, 1], mask_type="B",
                                                     name='residual_fw')
            output_state_fw = residual_state_fw + inputs

            residual_state_bw = tf_utils.conv2d_mask(output_state_bw, 2*self.hidden_dims, [1, 1], mask_type="B",
                                                     name='residual_bw')
            output_state_bw = residual_state_bw + inputs

            batch, height, width, channel = output_state_bw.get_shape().as_list()
            output_state_bw_except_last = tf.slice(output_state_bw, [0, 0, 0, 0], [-1, height-1, -1, -1])
            output_state_bw_only_last = tf.slice(output_state_bw, [0, height-1, 0, 0], [-1, -1, -1, -1])
            dummy_zeros = tf.zeros_like(output_state_bw_only_last)

            output_state_bw_with_last_zeros = tf.concat([output_state_bw_except_last, dummy_zeros], axis=1)

            return output_state_fw + output_state_bw_with_last_zeros

    def diagonal_lstm(self, inputs, name='diagonal_lstm'):
        with tf.variable_scope(name):
            skewed_inputs = tf_utils.skew(inputs, name='skewed_i')

            # input-to-state (K_is * x_i): 1x1 convolution. generate 4h x n x n tensor
            input_to_state = tf_utils.conv2d_mask(skewed_inputs, 4*self.hidden_dims, [1, 1], mask_type="B",
                                                  name="i_to_s")
            # [batch, width, height, hidden_dims*4]
            column_wise_inputs = tf.transpose(input_to_state, [0, 2, 1, 3])

            batch, width, height, channel = tf_utils.get_shape(column_wise_inputs)
            # [batch, max_time, height*hidden_dims*4]
            rnn_inputs = tf.reshape(column_wise_inputs, [-1, width, height*channel])
            # rnn_input_list = [tf.squeeze(rnn_input, axis=[1]) for rnn_input in tf.split(rnn_inputs, width, axis=1)]
            cell = DiagonalLSTMCell(self.hidden_dims, height, channel)

            # [batch, width, height * hidden_dims]
            outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=rnn_inputs, dtype=tf.float32)
            packed_outputs = outputs

            # [batch, width, height, hidden_dims]
            width_first_output = tf.reshape(packed_outputs, [-1, width, height, self.hidden_dims])

            skewed_outputs = tf.transpose(width_first_output, [0, 2, 1, 3])  # [batch, height, width, hidden_dims]
            outputs = tf_utils.unskew(skewed_outputs)

        return outputs


class DiagonalLSTMCell(core_rnn_cell.RNNCell):
    def __init__(self, hidden_dims, height, channel):
        self._num_unit_shards = 1
        self._forget_bias = 1.

        self._height = height
        self._channel = channel

        self._hidden_dims = hidden_dims
        self._num_units = self._hidden_dims * self._height
        self._state_size = self._num_units * 2
        self._output_size = self._num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, i_to_s, state, name='DiagonalBiLSTMCell'):
        c_prev = tf.slice(state, begin=[0, 0], size=[-1, self._num_units])
        # [batch, height * hidden_dims]
        h_prev = tf.slice(state, begin=[0, self._num_units], size=[-1, self._num_units])

        # i_to_s: [batch, 4 * height * hidden_dims]
        input_size = i_to_s.get_shape().with_rank(2)[1]

        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        with tf.variable_scope(name):
            # input-to-state (K_ss * h_{i-1}) : 2x1 convolution. generate 4h x n x n ternsor.
            # [batch, height, 1, hidden_dims]
            conv1d_inputs = tf.reshape(h_prev, [-1, self._height, 1, self._hidden_dims], name='conv1d_inputs')

            # [batch, height, 1, hidden_dims * 4]
            conv_s_to_s = tf_utils.conv1d(conv1d_inputs, 4*self._hidden_dims, kernel_size=2, name='s_to_s')
            # [batch, height * hidden_dims * 4]
            s_to_s = tf.reshape(conv_s_to_s, [-1, self._height * self._hidden_dims * 4])
            lstm_matrix = tf_utils.sigmoid(s_to_s + i_to_s)

            # i=input_gate, g=new_input, f=forget_gate, o=output_gate
            o, f, i, g = tf.split(lstm_matrix, 4, axis=1)
            c = f * c_prev + i * g
            h = o * tf_utils.tanh(c)

        new_state = tf.concat([c, h], axis=1)
        return h, new_state

