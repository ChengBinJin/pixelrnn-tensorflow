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
from datetime import datetime

from pixelrnn import PixelRNN
from dataset_ import Dataset

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.iter_epoch = 0
        self._make_folders()
        self._init_logger()

        self.dataset = Dataset(self.sess, self.flags, self.flags.dataset, log_path=self.log_out_dir)
        self.model = PixelRNN(self.sess, self.flags, self.dataset, log_path=self.log_out_dir)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def _make_folders(self):
        if self.flags.is_train:  # train stage
            if self.flags.load_model is None:
                cur_time = datetime.now().strftime("%Y%m%d-%H%M")
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
            self.writer = tf.summary.FileWriter("{}/logs/{}".format(self.flags.dataset, cur_time))

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

            logger.info('epochs: {}'.format(self.flags.epochs))
            logger.info('print_freq: {}'.format(self.flags.print_freq))
            logger.info('save_freq: {}'.format(self.flags.save_freq))
            logger.info('sample_batch: {}'.format(self.flags.sample_batch))
            logger.info('load_model: {}'.format(self.flags.load_model))

    def train(self):
        # load initialized checkpoint that provided
        if self.flags.load_model is not None:
            if self.load_model():
                logger.info(' [*] Load SUCCESS!\n')
            else:
                logger.info(' [!] Load failed...\n')

        for self.iter_epoch in range(self.flags.epochs):
            for iter_batch in range(self.dataset.train_step_per_epoch):
                # next batch
                batch_imgs, batch_labels = self.dataset.train_next_batch()

                # train_step
                loss, summary = self.model.train_step(batch_imgs)
                self.model.print_info(loss, iter_batch, self.iter_epoch, self.dataset.train_step_per_epoch)
                self.writer.add_summary(summary, self.iter_epoch * self.dataset.train_step_per_epoch + iter_batch)
                self.writer.flush()

            # samppling images and save them
            self.sample(self.iter_epoch, self.sample_out_dir)

            # save model
            self.save_model(self.iter_epoch)

        # last save
        self.save_model(self.flags.epochs)

    def test(self):
        # load initialized checkpoint that provided
        if self.flags.load_model is not None:
            if self.load_model():
                logger.info(' [*] Load SUCCESS!\n')
            else:
                logger.info(' [!] Load failed...\n')

        num_iters = 20
        for iter_time in range(num_iters):
            print('iter_time: {}...'.format(iter_time))
            self.sample(iter_time, self.test_out_dir)

    def sample(self, iter_epoch, save_folder):
        samples = self.model.sample_imgs()
        self.model.plots(samples, iter_epoch, save_folder)

    def save_model(self, iter_epoch):
        if np.mod(iter_epoch, self.flags.save_freq) == 0:
            model_name = 'model'
            self.saver.save(self.sess, os.path.join(self.model_out_dir, model_name), global_step=iter_epoch)
            logger.info('[*] Model saved! Iter: {}'.format(iter_epoch))

    def load_model(self):
        logger.info(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))

            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            self.iter_epoch = int(meta_graph_path.split('-')[-1].split('.')[0])

            logger.info('[*] Load epoch_time: {}'.format(self.iter_epoch))
            return True
        else:
            return False
