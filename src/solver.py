# ---------------------------------------------------------
# Tensorflow PixelRNN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import sys
import logging
import cv2
import tensorflow as tf
from datetime import datetime

from dataset_ import Dataset

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.iter_time = 0
        self._make_folders()
        self._init_logger()

        self.dataset = Dataset(self.sess, self.flags, self.flags.dataset, log_path=self.log_out_dir)
        # self.model = PixelRNN(self.sess, self.flags, self.dataset, log_path=self.log_out_dir)

        # self.saver = tf.train.Saver()
        # self.sess.run(tf.global_variables_initializer())

    def _make_folders(self):
        if self.flags.is_train:  # train stage
            if self.flags.load_model is None:
                cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                self.model_out_dir = "{}/model/{}".format(self.flags.dataset, cur_time)

                if not os.path.isdir(self.model_out_dir):
                    os.makedirs(self.model_out_dir)
            else:
                cur_time = self.flags.load_model
                self.model_out_dir = "{}/model/{}".format(self.flags.dataset, cur_time)

            self.sample_out_dir = "{}/sample/{}".format(self.flags.dataset, cur_time)

            if not os.path.isdir(self.sample_out_dir):
                os.makedirs(self.sample_out_dir)

            self.log_out_dir = "{}/logs/{}".format(self.flags.dataset, cur_time)
            self.train_writer = tf.summary.FileWriter("{}/logs/{}".format(self.flags.dataset, cur_time))

        elif not self.flags.is_train:  # test stage
            self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)
            self.test_out_dir = "{}/test/{}".format(self.flags.dataset, self.flags.load_model)
            self.log_out_dir = "{}/logs/{}".format(self.flags.dataset, self.flags.load_model)

            if not os.path.isdir(self.test_out_dir):
                os.makedirs(self.test_out_dir)

    def _init_logger(self):
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
        # file handler
        file_handler = logging.FileHandler(os.path.join(self.log_out_dir, 'solver.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        # add Handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        if self.flags.is_train:
            logger.info('gpu_index: {}'.format(self.flags.gpu_index))
            logger.info('model: {}'.format(self.flags.model))
            logger.info('batch_size: {}'.format(self.flags.batch_size))
            logger.info('dataset: {}'.format(self.flags.dataset))

            logger.info('is_train: {}'.format(self.flags.is_train))
            logger.info('learning_rate: {}'.format(self.flags.learning_rate))

            logger.info('iters: {}'.format(self.flags.iters))
            logger.info('print_freq: {}'.format(self.flags.print_freq))
            logger.info('save_freq: {}'.format(self.flags.save_freq))
            logger.info('sample_freq: {}'.format(self.flags.sample_freq))
            logger.info('load_model: {}'.format(self.flags.load_model))

    def train(self):
        print(' [*] Hello train function!')

        imgs, _ = self.dataset.train_next_batch()
        for idx in range(imgs.shape[0]):
            # print('Label: {}'.format(labels[idx]))
            img = imgs[idx]
            img = img[:, :, ::-1]  # RGB to BGR
            cv2.imshow('Show', img)

            if cv2.waitKey(0) & 0xFF == 27:
                sys.exit(' [!] Esc clicked!')

        print('imgs shape: {}'.format(imgs.shape))
        # print('labels shape: {}'.format(labels.shape))

    def test(self):
        print(' [*] Hello test function!')
