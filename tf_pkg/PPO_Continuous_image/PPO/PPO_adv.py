# -*- coding: UTF-8 -*-
#没写oldpi网络的情况
from cv2 import merge
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow import keras
import time
import cv2
import os
import math
import sys


A_LR = 0.0002
A_DIM =2
A_UPDATE_STEPS = 10
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective
][1]


class PPO:
    def __init__(self,loadpath):
        # Parameter for LSTM
        self.Num_dataSize = 364  # 360 sensor size + 4 self state size
        self.Num_cellState = 512

        # parameters for skipping and stacking
        self.Num_skipFrame = 1
        self.Num_stackFrame = 4

        # Parameters for CNN
        self.first_conv = [8, 8, self.Num_stackFrame, 32]
        self.second_conv = [4, 4, 32, 64]
        self.third_conv = [3, 3, 64, 64]


        self.first_dense = [10*10*64+self.Num_cellState, 512]
        self.second_dense_state = [self.first_dense[1], 1]
        self.second_dense_action = [self.first_dense[1], 11]


        # Parameters for network
        self.img_size = 80  # input image size

        # date - hour - minute - second of training time
        self.date_time = str(datetime.date.today())

        self.load_path = loadpath

        self.oldpi_prob_list=[]

        self.sigma=[[1.2,1.2]]


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

        # # Input------sensor
        # self.x_sensor = tf.placeholder(tf.float32, shape=[None, self.Num_stackFrame, self.Num_dataSize],name="state")
        # self.x_unstack = tf.unstack(self.x_sensor, axis=1)

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
                self.check_closs=tf.check_numerics(self.Critic_loss,"critic")
                

            #Actor
            with tf.variable_scope('Actor'):
                self.pi = self._build_anet('pi',trainable=True,input=h_concat)


        with tf.variable_scope('Train'):

            self.tfa = tf.placeholder(tf.float32,[None,A_DIM],'action')
            self.oldpi_prob=tf.placeholder(tf.float32,[None,A_DIM],'odlpi_prob')
            # self.oldpi_prob=tf.clip_by_value(self.oldpi_prob,1e-4,10)
            self.tfadv = tf.placeholder(tf.float32,[None,1],'advantage')

            with tf.variable_scope('sample_action'):
                self.sample_op = tf.squeeze(self.pi.sample(1),axis=0)   # pi.sample(1) 采样一次获得[1,2]，tf.squeeze()从张量形状中移除大小为1的维度. 最终为[2]
                self.check_action=tf.check_numerics(self.sample_op,"action")    

            with tf.variable_scope('prob'):
                self.ev_prob=self.pi.prob(self.tfa)


            with tf.variable_scope('loss'):
                with tf.variable_scope('surrogate'):
                    ratio = self.pi.prob(self.tfa)/self.oldpi_prob
                    # self.m=self.pi.prob(self.tfa)
                    self.surr = ratio * self.tfadv
                    self.check_surr=tf.check_numerics(self.surr,"surr")

                if METHOD['name'] == 'kl_pen':
                    pass
                    # self.tflam = tf.placeholder(tf.float32,None,'lambda')
                    # kl = tf.distributions.kl_divergence(self.oldpi,self.pi)
                    # self.kl_mean = tf.reduce_mean(kl)
                    # self.Actor_loss = -tf.reduce_mean(surr-self.tflam * kl)

                else:
                    self.Actor_loss = -tf.reduce_mean(tf.minimum(self.surr,
                                                            tf.clip_by_value(ratio,1.-METHOD['epsilon'],1.+METHOD['epsilon'])*self.tfadv)
                                            )
                    self.Total_loss = self.Actor_loss+ self.Critic_loss * 0.5
                with tf.variable_scope('train'):
                    self.train_op = tf.train.AdamOptimizer(A_LR,epsilon=1e-08).minimize(self.Total_loss)

        
        with tf.variable_scope('record'):
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # 运行时记录运行信息的proto
            run_metadata = tf.RunMetadata()
            # 将配置信息和运行记录信息的proto传入运行过程，从而进行记录
            tf.summary.scalar("Critic_loss", self.Critic_loss)
            tf.summary.scalar("Actor_loss",self.Actor_loss)
            self.merged=tf.summary.merge_all()  

        pass


    def _build_anet(self,name,trainable,input):
        with tf.variable_scope(name):
            Actor_output1 = tf.layers.dense(input,self.first_dense[1],tf.nn.relu,trainable=trainable)/30   #输出不要太大，否则tanh很容易满
            # w1 = self.weight_variable([self.first_dense[1],A_DIM])
            # b1 =  tf.Variable(tf.zeros([2])) # b_conv1 = ([32])
            # Actor_mu  =tf.nn.tanh( tf.matmul(Actor_output1, w1)+b1)
            Actor_mu = tf.layers.dense(Actor_output1,A_DIM,tf.nn.tanh,trainable=trainable,kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),bias_initializer='zeros')    #tf.layers.dense(l1,A_DIM,tf.nn.tanh,trainable=trainable)
            self.Actor_mu_=tf.concat([tf.reshape((Actor_mu[:,0]+1)/2,[-1,1]),tf.reshape(Actor_mu[:,1],[-1,1])], 1)                   #把第一列转化的[0,1],再合并到一起
            # self.Actor_sigma = tf.layers.dense(Actor_output1,A_DIM,tf.nn.softplus,trainable=trainable)
            self.Actor_sigma=tf.placeholder(tf.float32,[1,A_DIM],'action')
            norm_dist = tf.distributions.Normal(loc=self.Actor_mu_,scale=self.Actor_sigma ) # 一个正态分布
        # params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=name)   #收集name空间下的所有参数
        return norm_dist

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
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/PPO/' + current_time 
        writer=tf.summary.FileWriter(log_dir, tf.get_default_graph())
        return sess,saver,writer


    def choose_action(self,observation_stack):
        # s = s[np.newaxis,:]
        a = self.sess.run(self.sample_op,{self.x_image:observation_stack,self.Actor_sigma:self.sigma})[0]
        # print(a)
        a0=np.clip(a[0],0,1)
        a1=np.clip(a[1],-1,1)
        a=[a0,a1]
        return(a)

    def train(self,observation_stack,state_stack,r,a,train_step):
        # self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage,{self.x_image:observation_stack,self.tfdc_r:r}) # 得到advantage value
        oldpi_prob,mu=self.sess.run([self.ev_prob,self.Actor_mu_],{self.x_image:observation_stack, self.tfa: a,self.Actor_sigma:self.sigma})
        # oldpi_prob=np.clip(oldpi_prob,1e-8,100000)
        if train_step<=50000:
            self.sigma=[[1.2,1.2]]
        elif train_step<=150000:
            self.sigma=[[0.8,0.8]]
        else:
            self.sigma=[[0.35,0.35]]
        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _,kl = self.sess.run([self.Actor_train_op,self.kl_mean],
                                     {self.x_image:observation_stack,self.tfadv:adv,self.tflam:METHOD['lam']})
                if kl > 4 * METHOD['kl_target']:
                    break
                elif kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                    METHOD['lam'] /= 2
                elif kl > METHOD['kl_target'] * 1.5:
                    METHOD['lam'] *= 2
                METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)  # sometimes explode, this clipping is my solution

        else:   # clipping method, find this is better (OpenAI's paper)
            for i in range(A_UPDATE_STEPS):
                _,merged,self.out=self.sess.run([self.train_op,self.merged,self.surr], {self.x_image:observation_stack,self.tfa: a, self.tfadv: adv,self.oldpi_prob:oldpi_prob,self.tfdc_r: r,self.Actor_sigma:self.sigma})
                # print(i,"o",o[0,0])
                # print(i,"m",m[0,0])
        with open('/home/hisen/Path_Planning_A2C/src/tf_pkg/PPO_Continuous_image/PPO_openai.txt','a') as f:             
            f.write('oldpi_prob')             
            f.write('\r\n')             
            np.savetxt(f, oldpi_prob, delimiter=',', fmt = '%s')            
            f.write('adv')             
            f.write('\r\n')             
            np.savetxt(f, adv, delimiter=',', fmt = '%s')             
            f.write('surr')             
            f.write('\r\n')             
            np.savetxt(f, self.out, delimiter=',', fmt = '%s')              
            f.write('mu')             
            f.write('\r\n')             
            np.savetxt(f, mu, delimiter=',', fmt = '%s')             
            f.write('\r\n')
        self.writer.add_summary(merged, train_step) 

        pass

    def get_v(self,observation_stack,state_stack):
        # if observation_stack.ndim < 2:s = s[np.newaxis,:]
        return self.sess.run(self.Critic_output2,{self.x_image:observation_stack})[0,0]

    def save_model(self):
        # ------------------------------
        save_path = self.saver.save(
            self.sess, 'saved_networks/' + 'PPO_image_justmu' + '_' + self.date_time + "/model.ckpt")
        # ------------------------------
        pass