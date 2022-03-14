import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import cv2
import os
import math


class A2C:
    def __init__(self, sess):

        # Parameter for LSTM
        self.Num_dataSize = 364  # 360 sensor size + 4 self state size
        self.Num_cellState = 512

        # Parameters for CNN
        self.first_conv = [8, 8, self.Num_stackFrame, 32]
        self.second_conv = [4, 4, 32, 64]
        self.third_conv = [3, 3, 64, 64]

        self.first_dense = [10*10*64+self.Num_cellState, 512]
        self.second_dense_state = [self.first_dense[1], 1]
        self.second_dense_action = [self.first_dense[1], 11]



        self.actor=1
        self.critic=1
        self.first_dense=
        pass

    def weight_variable(self, shape):
            return tf.Variable(self.xavier_initializer(shape))

    def bias_variable(self, shape):  # 初始化偏置项
        return tf.Variable(self.xavier_initializer(shape))

    # Xavier Weights initializer
    def xavier_initializer(self, shape):
        dim_sum = np.sum(shape)
        if len(shape) == 1:
            dim_sum += 1
        bound = np.sqrt(2.0 / dim_sum)
        return tf.random_uniform(shape, minval=-bound, maxval=bound)

   

    def network(self):
        tf.reset_default_graph()
        # Input------image
        self.x_image = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.Num_stackFrame],name="image")
        self.x_normalize = (self.x_image - (255.0/2)) / (255.0/2)  # 归一化处理
        
        # Input------sensor
        self.x_sensor = tf.placeholder(tf.float32, shape=[None, self.Num_stackFrame, self.Num_dataSize],name="state")
        self.x_unstack = tf.unstack(self.x_sensor, axis=1)

        with tf.variable_scope('network'):
            # Convolution variables
            with tf.variable_scope('CNN'):
                w_conv1 = self.weight_variable(self.first_conv)  # w_conv1 = ([8,8,4,32])
                b_conv1 = self.bias_variable([self.first_conv[3]])  # b_conv1 = ([32])

                # second_conv=[4,4,32,64]
                w_conv2 = self.weight_variable(self.second_conv)  # w_conv2 = ([4,4,32,64])
                b_conv2 = self.bias_variable([self.second_conv[3]])  # b_conv2 = ([64])

                # third_conv=[3,3,64,64]
                w_conv3 = self.weight_variable(self.third_conv)  # w_conv3 = ([3,3,64,64])
                b_conv3 = self.bias_variable([self.third_conv[3]])  # b_conv3 = ([64])

                h_conv1 = tf.nn.relu(self.conv2d(self.x_normalize, w_conv1, 4) + b_conv1)
                h_conv2 = tf.nn.relu(self.conv2d(h_conv1, w_conv2, 2) + b_conv2)
                h_conv3 = tf.nn.relu(self.conv2d(h_conv2, w_conv3, 1) + b_conv3)
                h_pool3_flat = tf.reshape(h_conv3, [-1, 10 * 10 * 64])  # 将tensor打平到vector中

            with tf.variable_scope('LSTM'):
                # LSTM cell
                #TODO:看看LSTM的结构
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.Num_cellState)
                rnn_out, rnn_state = tf.nn.static_rnn(inputs=self.x_unstack, cell=cell, dtype=tf.float32)
                rnn_out = rnn_out[-1]

            h_concat = tf.concat([h_pool3_flat, rnn_out], axis=1)

            # with tf.variable_scope('Critic'):
            #     # Critic
            #     w_Critic_1 = self.weight_variable(
            #         self.first_dense)  # w_fc1_1 = ([6400,512])
            #     b_Critic_1 = self.bias_variable(
            #         [self.first_dense[1]])  # b_fc1_1 = ([512])
            #     w_Critic_2 = self.weight_variable(
            #         self.second_dense_state)  # w_fc2_1 = ([512，1])
            #     b_Critic_2 = self.bias_variable(
            #         [self.second_dense_state[1]])  # b_fc2_1 = ([1])
            #     Critic_output1 = tf.nn.relu(tf.matmul(h_concat, w_Critic_1)+b_Critic_1)
            #     Critic_output2 = tf.matmul(Critic_output1, w_Critic_2)+b_Critic_2

            # with tf.variable_scope('Actor'):
            #     # Actor
            #     w_Actor_1 = self.weight_variable(
            #         self.first_dense)  # w_fc1_1 = ([6400,512])
            #     b_Actor_1 = self.bias_variable(
            #         [self.first_dense[1]])  # b_fc1_1 = ([512])
            #     w_Actor_2 = self.weight_variable(
            #         self.second_dense_action)  # w_fc2_1 = ([512，11])
            #     b_Actor_2 = self.bias_variable(
            #         [self.second_dense_action[1]])  # b_fc2_1 = ([11])
            #     Actor_output1 =tf.nn.relu(tf.matmul(h_concat, w_Actor_1)+b_Actor_1)
            #     Actor_output2 = tf.nn.softmax(tf.matmul(Actor_output1, w_Actor_2)+b_Actor_2)
            
            with tf.variable_scope('Critic'):
                Critic_output1 = tf.layers.dense(
                    inputs=h_concat,
                    units=self.first_dense[1],  # number of hidden units
                    activation=tf.nn.relu,  # None
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='Critic_output1'
                )
                self.Critic_output2 = tf.layers.dense(
                    inputs=Critic_output1,
                    units=1,  # output units
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='Critic_output2'
                )

            with tf.variable_scope('Actor'):
                Actor_output1 = tf.layers.dense(
                    inputs=h_concat,
                    units=self.first_dense[1],    # number of hidden units
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name=' Actor_output1'
                )

                self.Actor_output2 = tf.layers.dense(
                    inputs= Actor_output1,
                    units=self.second_dense_action[1],    # output units
                    activation=tf.nn.softmax,   # get action probabilities
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                    name='Actor_output2'
                )
        return  self.Actor_output2, self.Critic_output2



    def choose_action(self):
        pass

    def loss_and_train(self):
        with tf.variable_scope('Train'):
            self.output_critic_ = tf.placeholder(tf.float32,[None], "v_next")     #[32]
            self.r = tf.placeholder(tf.float32,[None], 'r')                      #[n]
            self.a = tf.placeholder(tf.float32, [None, self.Num_action], "action")
            self.w_is = tf.placeholder(tf.float32, shape=[None])
            self.m=self.GAMMA *  tf.reduce_sum(self.output_critic,reduction_indices=1) - self.output_critic_ 
            
            self.TD_error =self.r  + self.m   #[32]

            self.loss =tf.reduce_mean(tf.square(self.TD_error))
            self.train_Critic = tf.train.AdamOptimizer(self.Lr_C).minimize(self.loss)
            log_prob=tf.log(tf.reduce_sum(tf.multiply(self.output_actor, self.a), reduction_indices=1))   #[32]
            self.exp_v = tf.reduce_mean(tf.multiply(log_prob, self.TD_error))   #[1]
            self.train_Actor = tf.train.AdamOptimizer(self.Lr_A).minimize(-self.exp_v)


    def train(self):

        pass

