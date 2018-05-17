"""
License: Apache-2.0
Author: Huadong Liao
E-mail: naturomics.liao@gmail.com
"""

import tensorflow as tf

from config import cfg
from utils import get_batch_data
from utils import softmax
from utils import reduce_sum
from capsLayer import CapsLayer
from capsLayer import cap_dropout

epsilon = 1e-9


class CapsNet(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.build_arch()
        self.loss = self._compute_loss()
        self._summary()
        # t_vars = tf.trainable_variables()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)  # var_list=t_vars)
        tf.logging.info('Seting up the main structure')

    def build_arch(self):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            self.embedded_chars_expanded.set_shape([cfg.batch_size, self.sequence_length, self.embedding_size, 1])
        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv1 = tf.contrib.layers.conv2d(self.embedded_chars_expanded, num_outputs=self.embedding_size,
                                             kernel_size=(3, self.embedding_size), stride=(1, 1),
                                             padding='VALID')
        #     print("conv1 shape = {}\n".format(conv1.get_shape()))
        # Primary Capsules layer, return [batch_size, 1152, 8, 1]
        self.conv1 = tf.reshape(conv1, [cfg.batch_size, -1, self.embedding_size, 1])
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=32, vec_len=4, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(self.conv1, kernel_size=(2, self.embedding_size), stride=(1, 1))
        # DigitCaps layer, return [batch_size, 10, 16, 1]
        with tf.variable_scope('secondCaps_layer'):
            secondCaps = CapsLayer(num_outputs=16, vec_len=4, with_routing=True, input_atoms=4, layer_type='FC')
            self.caps2 = secondCaps(caps1)
        with tf.variable_scope('DigitCaps_layer'):
            outputCaps = CapsLayer(num_outputs=2, vec_len=4, with_routing=True, input_atoms=4, layer_type='FC')
            self.caps3 = outputCaps(self.caps2)
        # Decoder structure in Fig. 2
        # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
            self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps3),
                                               axis=2, keepdims=True) + epsilon)
            self.softmax_v = softmax(self.v_length, axis=1)

            # b). pick out the index of max softmax val of the 10 caps
            # [batch_size, 10, 1, 1] => [batch_size] (index)
            self.argmax_idx = tf.argmax(self.softmax_v, axis=1)
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size,))
            # self.argmax_idx = tf.one_hot(self.argmax_idx, depth=2, axis=0)
            # Method 1.

            self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps3), axis=2, keepdims=True) + epsilon)

        # # 2. Reconstructe the MNIST images with 3 FC layers
        # # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
        # with tf.variable_scope('Decoder'):
        #     vector_j = tf.reshape(self.masked_v, shape=(cfg.batch_size, -1))
        #     fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
        #     assert fc1.get_shape() == [cfg.batch_size, 512]
        #     fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
        #     assert fc2.get_shape() == [cfg.batch_size, 1024]
        #     self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

    def _compute_loss(self):
        # 1. The margin loss
        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
        # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))
        # calc T_c: [batch_size, 10]
        # T_c = Y, is my understanding correct? Try it.
        T_c = self.input_y
        # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r
        print("T_c shape ={},L_c shape = {}\n".format(T_c.get_shape(), L_c.get_shape()))
        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss

        # self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        # The paper uses sum of squared error as reconstruction error, but we
        # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # mean squared error. In order to keep in line with the paper,the
        # regularization scale should be 0.0005*784=0.392
        self.total_loss = self.margin_loss
        return self.total_loss

    # Summary
    def _summary(self):
        train_summary = []
        # train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        # train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_err))
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        self.train_summary = tf.summary.merge(train_summary)
        correct_predictions = tf.equal(self.argmax_idx, tf.argmax(self.input_y, 1))
        print("!!!correct_predictions shape = {}\n".format(correct_predictions.get_shape()))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # correct_prediction = tf.equal(tf.to_int32(self.input_y), tf.one_hot(self.argmax_idx, depth=2, dtype=tf.int32))
        # self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))