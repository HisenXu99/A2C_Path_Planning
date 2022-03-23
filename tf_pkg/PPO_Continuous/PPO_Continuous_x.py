# -*- coding: UTF-8 -*-
from cv2 import merge
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import cv2
import os
import math


A_LR = 0.00004
C_LR = 0.00008
A_DIM =2
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective
][1]


class PPO:
    def __init__(self,loadpath):
        # Parameter for LSTM
        self.Num_dataSize = 364  # 360 sensor size + 4 self state size
        self.Num_cellState = 512

        # Parameters for CNN
        self.first_conv = [8, 8, self.Num_stackFrame, 32]
        self.second_conv = [4, 4, 32, 64]
        self.third_conv = [3, 3, 64, 64]
        # parameters for skipping and stacking
        self.Num_skipFrame = 1
        self.Num_stackFrame = 4

        self.first_dense = [10*10*64+self.Num_cellState, 512]
        self.second_dense_state = [self.first_dense[1], 1]
        self.second_dense_action = [self.first_dense[1], 11]


        # Parameters for network
        self.img_size = 80  # input image size

        # date - hour - minute - second of training time
        self.date_time = str(datetime.date.today())

        self.load_path = loadpath

        self.network()
        self.sess,self.saver,self.writer=self.init_sess()
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

    def conv2d(self, x, w, stride):  # 定义一个函数，用于构建卷积层
        return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')

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

            #Critic
            with tf.variable_scope('Critic'):
                Critic_outpot1 = tf.layers.dense(h_concat,self.first_dense[1],tf.nn.relu)
                self.Critic_output2 = tf.layers.dense(Critic_outpot1,1) # state-value
                self.tfdc_r = tf.placeholder(tf.float32,[None,1],'discounted_r')
                self.advantage = self.tfdc_r - self.Critic_output2
                self.Critic_loss = tf.reduce_mean(tf.square(self.advantage))
                self.Critic_train_op = tf.train.AdamOptimizer(C_LR).minimize(self.Critic_loss)
                closs = tf.summary.scalar("Critic_loss", self.Critic_loss)

            #Actor
            with tf.variable_scope('Actor'):
                self.pi = self._build_anet('pi',trainable=True,input=h_concat)

        with tf.variable_scope('network_old'):
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

            h_concat_old = tf.concat([h_pool3_flat, rnn_out], axis=1)

            #Critic
            with tf.variable_scope('Critic'):
                Critic_outpot1_old = tf.layers.dense(h_concat_old,self.first_dense[1],tf.nn.relu)
                Critic_output2_old = tf.layers.dense(Critic_outpot1_old,1) # state-value
                # self.tfdc_r = tf.placeholder(tf.float32,[None,1],'discounted_r')
                # self.advantage = self.tfdc_r - self.Critic_output2
                # self.Critic_loss = tf.reduce_mean(tf.square(self.advantage))
                # self.Critic_train_op = tf.train.AdamOptimizer(C_LR).minimize(self.Critic_loss)
                # closs = tf.summary.scalar("Critic_loss", self.Critic_loss)

            #Actor
            with tf.variable_scope('Actor'):
                self.oldpi = self._build_anet('oldpi',trainable=True,input=h_concat_old)

        with tf.variable_scope('Train'):
            with tf.variable_scope('sample_action'):
                self.sample_op = tf.squeeze(self.pi.sample(1),axis=0)   # pi.sample(1) 采样一次获得[1,2]，tf.squeeze()从张量形状中移除大小为1的维度. 最终为[2]
            with tf.variable_scope('update_oldpi'):
                self.update_oldpi_op = [oldp.assign(p) for p,oldp in zip(self.pi_params,self.oldpi_params)]
                # self.update_oldpi_op=oldpi.set_weights(pi.get_weights())
                ##############改###############
            with tf.variable_scope('prob'):
                self.ev_prob=self.oldpi.prob(self.tfa)


            self.tfa = tf.placeholder(tf.float32,[None,A_DIM],'action')
            self.oldpi_prob=tf.placeholder(tf.float32,[None,A_DIM],'odlpi_prob')
            self.tfadv = tf.placeholder(tf.float32,[None,1],'advantage')
            with tf.variable_scope('loss'):
                with tf.variable_scope('surrogate'):
                    ratio = self.pi.prob(self.tfa)/self.oldpi_prob
                    surr = ratio * self.tfadv

                if METHOD['name'] == 'kl_pen':
                    pass
                    # self.tflam = tf.placeholder(tf.float32,None,'lambda')
                    # kl = tf.distributions.kl_divergence(self.oldpi,self.pi)
                    # self.kl_mean = tf.reduce_mean(kl)
                    # self.Actor_loss = -tf.reduce_mean(surr-self.tflam * kl)

                else:
                    self.Actor_loss = -tf.reduce_mean(tf.minimum(surr,
                                                            tf.clip_by_value(ratio,1.-METHOD['epsilon'],1.+METHOD['epsilon'])*self.tfadv)
                                            )
                with tf.variable_scope('atrain'):
                    self.Actor_train_op = tf.train.AdamOptimizer(A_LR).minimize(self.Actor_loss)
            # aloss = tf.summary.scalar("Aritic_loss", self.Actor_loss)
        self.merged = tf.summary.merge_all()
        pass


    def _build_anet(self,name,trainable,input):
        with tf.variable_scope(name):
            Actor_output1 = tf.layers.dense(input,self.first_dense[1],tf.nn.relu,trainable=trainable)
            Actor_output2 = tf.layers.dense(Actor_output1,11,tf.nn.softmax,trainable=trainable)
        #     # Actor_mu_=tf.concat([tf.reshape((Actor_mu[:,0]+1)/2,[-1,1]),tf.reshape(Actor_mu[:,1],[-1,1])], 1)                   #把第一列转化的[0,1],再合并到一起
        #     # Actor_sigma = tf.layers.dense(Actor_output1,A_DIM,tf.nn.softplus,trainable=trainable)
        #     # norm_dist = tf.distributions.Normal(loc=Actor_mu_,scale=Actor_sigma ) # 一个正态分布
        # params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=name)   #收集name空间下的所有参数
        return Actor_output2

    def init_sess(self):
        # Initialize variables
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 程序最多只能占用指定gpu50%的显存  
        sess = tf.InteractiveSession(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        # Load the file if the saved file exists
        check_save = input('Load Model? (1=yes/2=no): ')
        if check_save == 1:
            # Restore variables from disk.
            saver.restore(sess, self.load_path + "/model.ckpt")
            print("Model restored.")
            check_train = input('Inference or Training? (1=Inference / 2=Training): ')
            if check_train == 1:
                self.Num_start_training = 0
                self.Num_training = 0

        writer=tf.summary.FileWriter("log_PPO/", tf.get_default_graph())
        return sess,saver,writer


    def choose_action(self,observation_stack,state_stack):
        # s = s[np.newaxis,:]
        a = self.sess.run(self.sample_op,{self.x_image:observation_stack,self.x_sensor:state_stack})[0]
        # print(a)
        a0=np.clip(a[0],0,1)
        a1=np.clip(a[1],-1,1)
        a=[a0,a1]
        return(a)

    def train(self,observation_stack,state_stack,r,a,train_step):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage,{self.x_image:observation_stack,self.x_sensor:state_stack,self.tfdc_r:r}) # 得到advantage value
        oldpi_prob=self.sess.run(self.ev_prob,{self.tfa: a})
        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _,kl = self.sess.run([self.Actor_train_op,self.kl_mean],
                                     {self.x_image:observation_stack,self.x_sensor:state_stack,self.tfadv:adv,self.tflam:METHOD['lam']})
                if kl > 4 * METHOD['kl_target']:
                    break
                elif kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                    METHOD['lam'] /= 2
                elif kl > METHOD['kl_target'] * 1.5:
                    METHOD['lam'] *= 2
                METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)  # sometimes explode, this clipping is my solution

        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.Actor_train_op, {self.x_image:observation_stack,self.x_sensor:state_stack, self.tfa: a, self.tfadv: adv,self.oldpi_prob:oldpi_prob}) for _ in range(A_UPDATE_STEPS)]

        for _ in range(C_UPDATE_STEPS):
            _,merged=self.sess.run([self.Critic_train_op,self.merged], {self.x_image:observation_stack,self.x_sensor:state_stack,self.tfdc_r: r})
        self.writer.add_summary(merged,train_step) #将日志数据写入文件
        pass