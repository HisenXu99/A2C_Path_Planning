# -*- coding: UTF-8 -*-
#没写oldpi网络的情况
from linecache import updatecache
from turtle import update
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
import sys

from distributions import DiagGaussianPd


A_LR = 0.0002
C_LR = 0.0004

A_DIM =2
UPDATE_STEPS = 10
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

        self.ent_coef=0.01
        self.vf_coef=0.5


        # Parameters for network
        self.img_size = 80  # input image size

        # date - hour - minute - second of training time
        self.date_time = str(datetime.date.today())

        self.load_path = loadpath

        self.oldpi_prob_list=[]

        self.network()
        self.sess,self.saver,self.writer=self.init_sess()
        pass

    def weight_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))

    def bias_variable(self, shape):  # 初始化偏置项
        return tf.Variable(self.xavier_initializer(shape))

    # Xavier Weights initializer
    def normc_initializer(self,std=1.0, axis=0):
        def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
            out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
            out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
            return tf.constant(out)
        return _initializer

    # def conv2d(self, x, w, stride):  # 定义一个函数，用于构建卷积层
    #     return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
    
    def intprod(self,x):
      return int(np.prod(x))
    
    def conv2d(self,x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None,
           summary_tag=None):
        with tf.variable_scope(name):
            stride_shape = [1, stride[0], stride[1], 1]
            filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = self.intprod(filter_shape[:3])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = self.intprod(filter_shape[:2]) * num_filters
            # initialize weights with random weights
            w_bound = np.sqrt(6. / (fan_in + fan_out))

            w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                                collections=collections)
            b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.zeros_initializer(),
                                collections=collections)

            if summary_tag is not None:
                tf.summary.image(summary_tag,
                                tf.transpose(tf.reshape(w, [filter_size[0], filter_size[1], -1, 1]),
                                            [2, 0, 1, 3]),
                                max_images=10)

            return tf.nn.conv2d(x, w, stride_shape, pad) + b
    
    def flattenallbut0(self,x):
      return tf.reshape(x, [-1, self.intprod(x.get_shape().as_list()[1:])])


    def network(self):
        # Input------image
        self.x_image = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.Num_stackFrame],name="image")
        x=self.x_normalize = (self.x_image - (255.0/2)) / (255.0/2)  # 归一化处理
        # x = ob / 255.0

        x = tf.nn.relu(self.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
        x = tf.nn.relu(self.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
        x = tf.nn.relu(self.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
        x = self.flattenallbut0(x)
        x = tf.nn.relu(tf.layers.dense(x, 512, name='lin', kernel_initializer=self.normc_initializer(1.0)))

        # with tf.variable_scope('LSTM'):
        #     # LSTM cell
        #     #TODO:看看LSTM的结构
        #     cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.Num_cellState)
        #     rnn_out, rnn_state = tf.nn.static_rnn(inputs=self.x_unstack, cell=cell, dtype=tf.float32)
        #     rnn_out = rnn_out[-1]

        # h_concat = tf.concat([h_pool3_flat, rnn_out], axis=1)

        logits = tf.layers.dense(x, 4, name='logits', kernel_initializer=self.normc_initializer(0.01))
        self.pd = DiagGaussianPd(logits)
        self.vpred = tf.layers.dense(x, 1, name='value', kernel_initializer=self.normc_initializer(1.0))[:,0]

        self.ac = self.pd.sample() 
        # self._act = U.function([stochastic, ob], [ac, self.vpred])

        with tf.variable_scope('train'):
            self.A = A = tf.placeholder(tf.float32,[None,A_DIM],'action')
            self.ADV = ADV = tf.placeholder(tf.float32, [None],"ADV")
            self.R = R = tf.placeholder(tf.float32, [None],"r")
            self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
            self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
            self.LR = LR = tf.placeholder(tf.float32, [])
            # Cliprange
            self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

            neglogpac = self.pd.neglogp(A)

            entropy = tf.reduce_mean(self.pd.entropy())

            vpred = self.vpred
            vpredclipped = OLDVPRED + tf.clip_by_value(self.vpred - OLDVPRED, - CLIPRANGE, CLIPRANGE) 
            # Unclipped value
            vf_losses1 = tf.square(vpred - R)
            # Clipped value
            vf_losses2 = tf.square(vpredclipped - R)

            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

            # Calculate ratio (pi current policy / pi old policy)
            ratio = tf.exp(OLDNEGLOGPAC - neglogpac)                            #应该是为了去掉分母为零的风险

            # Defining Loss = - J is equivalent to max J
            pg_losses = -ADV * ratio
            pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

            # Final PG loss
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
            clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

            # Total loss
            loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            grads_and_var = self.trainer.compute_gradients(loss)
            self._train_op = self.trainer.apply_gradients(grads_and_var)


        with tf.variable_scope('record'):
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # # 运行时记录运行信息的proto
            # run_metadata = tf.RunMetadata()
            # 将配置信息和运行记录信息的proto传入运行过程，从而进行记录
            closs=tf.summary.scalar("Critic_loss", vf_loss)
            aloss=tf.summary.scalar("Actor_loss",pg_loss)
            merged=tf.summary.merge_all()

        self.stats_list = [merged,pg_loss, vf_loss, entropy, approxkl, clipfrac]


    def choose_action(self,observation_stack):
        a=self.sess.run(self.ac,{self.x_image:observation_stack})[0]
        a0=np.clip(a[0],0,1)
        a1=np.clip(a[1],-1,1)
        a=[a0,a1]
        return a

    def train(self,obs,returns,actions,train_step, lr, cliprange, values):
        
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values
        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        neglogpacs=self.sess.run(self.pd.neglogp(actions),{self.x_image:obs})


        td_map = {
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values,
            self.x_image:obs
        }
        for start in range(UPDATE_STEPS):
            merged,_,_,_,_,_ ,_=self.sess.run(self.stats_list + [self._train_op],td_map)

        with open('/home/hisen/Path_Planning_A2C/src/tf_pkg/PPO_Continuous_image/oldpi.txt','a') as f:
            f.write('oldpi_prob')
            f.write('\r\n')
            np.savetxt(f, neglogpacs, delimiter=',', fmt = '%s')
            f.write('adv')
            f.write('\r\n')
            np.savetxt(f, advs, delimiter=',', fmt = '%s')
            f.write('actions')
            f.write('\r\n')
            np.savetxt(f, actions, delimiter=',', fmt = '%s')
            f.write('\r\n')

        # self.writer.add_summary(list[0], train_step) 

        pass

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

        
    def get_v(self,observation_stack):
        # if observation_stack.ndim < 2:s = s[np.newaxis,:]
        return self.sess.run(self.vpred,{self.x_image:observation_stack})[0]

    def save_model(self):
        # ------------------------------
        save_path = self.saver.save(
            self.sess, 'saved_networks/' + '10_D3QN_PER_image_add_sensor_empty_world_30m' + '_' + self.date_time + "/model.ckpt")
        # ------------------------------
        pass