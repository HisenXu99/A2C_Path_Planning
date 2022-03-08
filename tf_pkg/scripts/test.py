import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import cv2
import os
import math

from gazebo_env_D3QN_PER_image_add_sensor_empty_world_30m import envmodel


os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 默认显卡0

env = envmodel()

# 动作指令集---> v,w
action_dict = {0: [1.0, -1.0], 1: [1.0, -0.5], 2: [1.0, 0.0],
               3: [1.0, 0.5], 4: [1.0, 1.0], 5: [0.5, -1.0],
               6: [0.5, 0.0], 7: [0.5, 1.0], 8: [0.0, -1.0],
               9: [0.0, 0.0], 10: [0.0, 1.0]}


class A2C:
    def __init__(self):
        # Algorithm Information
        self.algorithm = 'A2C_PER'

        # Get parameters
        self.progress = ''
        self.Num_action = len(action_dict)

        # Initial parameters
        # ------------------------------
        self.Num_start_training = 50
        self.Num_training = 50000
        # ------------------------------
        self.Num_test = 0
        self.Lr_A = 0.001
        self.Lr_C = 0.01
        self.Gamma = 0.99
        self.GAMMA = 0.9
        self.Final_epsilon = 0.1
        # ------------------------------
        self.Epsilon = 1.0
        # ------------------------------
        self.load_path = '/home/hisen/DRL_Path_Planning-master/saved_networks/10_D3QN_PER_image_add_sensor_empty_world_30m_2022-02-21'
        self.step = 1
        self.score = 0
        self.episode = 0
        # ------------------------------

        # date - hour - minute - second of training time
        self.date_time = str(datetime.date.today())

        # parameters for skipping and stacking
        self.Num_skipFrame = 1
        self.Num_stackFrame = 4

        # Parameter for Experience Replay
        # ------------------------------
        self.Num_replay_memory = 5000
        # ------------------------------
        self.Num_batch = 32
        self.Replay_memory = []

        # Parameters for PER
        self.eps = 0.00001
        self.alpha = 0.6
        self.beta_init = 0.4
        self.beta = self.beta_init
        self.TD_list = np.array([])

        # Parameter for LSTM
        self.Num_dataSize = 364  # 360 sensor size + 4 self state size
        self.Num_cellState = 512

        # Parameters for CNN
        self.first_conv = [8, 8, self.Num_stackFrame, 32]
        self.second_conv = [4, 4, 32, 64]
        self.third_conv = [3, 3, 64, 64]

        self.first_dense = [10*10*64+self.Num_cellState, 512]
        self.second_dense_state = [self.first_dense[1], 1]
        self.second_dense_action = [self.first_dense[1], self.Num_action]

        # Parameters for network
        self.img_size = 80  # input image size

        # Initialize agent robot
        self.agentrobot = 'jackal0'

        # Define the distance from start point to goal point
        self.d = 15.0

        # Define the step for updating the environment
        self.MAXSTEPS = 300
        # ------------------------------
        self.MAXEPISODES = 500
        # ------------------------------

        # Initialize Network
        self.output_actor, self.output_critic = self.network()
        self.train_Actor, self.train_Critic, self.loss_train, self.w_is, self.TD_error = self.loss_and_train()
        self.sess, self.saver = self.init_sess()
        pass

    def init_sess(self):
        # Initialize variables
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 程序最多只能占用指定gpu50%的显存
        sess = tf.InteractiveSession(config=config)
        # ------------------------------
        os.makedirs('saved_networks/' +
                    '10_D3QN_PER_image_add_sensor_empty_world_30m' + '_' + self.date_time)
        # ------------------------------

        init = tf.global_variables_initializer()
        sess.run(init)

        writer=tf.summary.FileWriter('log_A2C', tf.get_default_graph())

        writer.close()

        # Load the file if the saved file exists
        saver = tf.train.Saver()
        # check_save = input('Load Model? (1=yes/2=no): ')

        # if check_save == 1:
        #     # Restore variables from disk.
        #     saver.restore(sess, self.load_path + "/model.ckpt")
        #     print("Model restored.")

        #     check_train = input(
        #         'Inference or Training? (1=Inference / 2=Training): ')
        #     if check_train == 1:
        #         self.Num_start_training = 0
        #         self.Num_training = 0

        return sess, saver

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

    # Convolution function
    def conv2d(self, x, w, stride):  # 定义一个函数，用于构建卷积层
        return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')

    def network(self):
        tf.reset_default_graph()
        # Input------image
        self.x_image = tf.placeholder(
            tf.float32, shape=[None, self.img_size, self.img_size, self.Num_stackFrame],name="image")
        self.x_normalize = (self.x_image - (255.0/2)) / (255.0/2)  # 归一化处理

        # Input------sensor
        self.x_sensor = tf.placeholder(
            tf.float32, shape=[None, self.Num_stackFrame, self.Num_dataSize])
        self.x_unstack = tf.unstack(self.x_sensor, axis=1)

        with tf.variable_scope('network'):
            # Convolution variables
            with tf.variable_scope('CNN'):
                w_conv1 = self.weight_variable(
                    self.first_conv)  # w_conv1 = ([8,8,4,32])
                b_conv1 = self.bias_variable(
                    [self.first_conv[3]])  # b_conv1 = ([32])

                # second_conv=[4,4,32,64]
                w_conv2 = self.weight_variable(
                    self.second_conv)  # w_conv2 = ([4,4,32,64])
                b_conv2 = self.bias_variable(
                    [self.second_conv[3]])  # b_conv2 = ([64])

                # third_conv=[3,3,64,64]
                w_conv3 = self.weight_variable(
                    self.third_conv)  # w_conv3 = ([3,3,64,64])
                b_conv3 = self.bias_variable(
                    [self.third_conv[3]])  # b_conv3 = ([64])

                h_conv1 = tf.nn.relu(self.conv2d(
                    self.x_normalize, w_conv1, 4) + b_conv1)
                h_conv2 = tf.nn.relu(self.conv2d(h_conv1, w_conv2, 2) + b_conv2)
                h_conv3 = tf.nn.relu(self.conv2d(h_conv2, w_conv3, 1) + b_conv3)
                h_pool3_flat = tf.reshape(
                    h_conv3, [-1, 10 * 10 * 64])  # 将tensor打平到vector中

            with tf.variable_scope('LSTM'):
                # LSTM cell
                #TODO:看看LSTM的结构
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.Num_cellState)
                rnn_out, rnn_state = tf.nn.static_rnn(
                    inputs=self.x_unstack, cell=cell, dtype=tf.float32)
                rnn_out = rnn_out[-1]

            h_concat = tf.concat([h_pool3_flat, rnn_out], axis=1)

            with tf.variable_scope('Critic'):
                # Critic
                w_Critic_1 = self.weight_variable(
                    self.first_dense)  # w_fc1_1 = ([6400,512])
                b_Critic_1 = self.bias_variable(
                    [self.first_dense[1]])  # b_fc1_1 = ([512])
                w_Critic_2 = self.weight_variable(
                    self.second_dense_state)  # w_fc2_1 = ([512，1])
                b_Critic_2 = self.bias_variable(
                    [self.second_dense_state[1]])  # b_fc2_1 = ([1])
                Critic_output1 = tf.nn.relu(tf.matmul(h_concat, w_Critic_1)+b_Critic_1)
                Critic_output2 = tf.matmul(Critic_output1, w_Critic_2)+b_Critic_2
                
            with tf.variable_scope('Actor'):
                # Actor
                w_Actor_1 = self.weight_variable(
                    self.first_dense)  # w_fc1_1 = ([6400,512])
                b_Actor_1 = self.bias_variable(
                    [self.first_dense[1]])  # b_fc1_1 = ([512])
                w_Actor_2 = self.weight_variable(
                    self.second_dense_action)  # w_fc2_1 = ([512，10])
                b_Actor_2 = self.bias_variable(
                    [self.second_dense_action[1]])  # b_fc2_1 = ([10])
                Actor_output1 =tf.nn.relu(tf.matmul(h_concat, w_Actor_1)+b_Actor_1)
                Actor_output2 = tf.nn.softmax(tf.matmul(Actor_output1, w_Actor_2)+b_Actor_2)

        return Actor_output2, Critic_output2


    def loss_and_train(self):
        #TODO:看看这里的output_critic_的形状，传入32个样本的minibach的时候
        self.output_critic_ = tf.placeholder(tf.float32,[None,1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        r=tf.reshape(self.r,[-1,1])
        #????????????????
        self.a = tf.placeholder(tf.int32, None, "action")
        ################################################## PER ############################################################
        w_is = tf.placeholder(tf.float32, shape=[None])
        td_error = r + self.GAMMA * self.output_critic - self.output_critic_   #[32,1]

        # Loss = tf.reduce_mean(tf.square(y_prediction - y_target))
        loss = tf.square(td_error)    # TD_error = (r+gamma*V_next) - V_eval
        Loss = tf.reduce_sum(tf.multiply(w_is, loss))
        ###################################################################################################################
        output_1=tf.reshape(self.output_actor,[1,-1])
        log_prob=tf.log(output_1[0,self.a])
        
        # for ax in range(self.a.shape[0].value):
        #     #TODO:检查log_porb是不是一个一维tensor
        #     #NOTE:self.a是一个二维矩阵
        #     log_prob.append(tf.log(self.output_actor[ax,tf.argmax(self.a[ax,:],0)]))
        #     tf.reshape(log_prob,[-1,1])

        # log_prob = tf.log(self.output_actor[:, self.a])
        #TODO:检查两个一维矩阵相乘
        self.exp_v = tf.reduce_mean(log_prob * td_error)
        train_Actor = tf.train.AdamOptimizer(self.Lr_A).minimize(-self.exp_v)
        train_Critic = tf.train.AdamOptimizer(self.Lr_C).minimize(loss)

        return train_Actor, train_Critic, Loss, w_is, td_error

        pass

    # Initialize input
    def input_initialization(self, env_info):
        state = env_info[0]  # laser info + self state
        state_set = []
        for i in range(self.Num_skipFrame * self.Num_stackFrame):
            state_set.append(state)
        state_stack = np.zeros((self.Num_stackFrame, self.Num_dataSize))
        for stack_frame in range(self.Num_stackFrame):
            state_stack[(self.Num_stackFrame - 1) - stack_frame,
                        :] = state_set[-1 - (self.Num_skipFrame * stack_frame)]

        observation = env_info[1]  # image info
        observation_set = []
        for i in range(self.Num_skipFrame * self.Num_stackFrame):
            observation_set.append(observation)
        # 使用观察组根据跳帧和堆叠帧的数量堆叠帧
        observation_stack = np.zeros(
            (self.img_size, self.img_size, self.Num_stackFrame))
        # print("shape of observation stack={}".format(observation_stack.shape))
        for stack_frame in range(self.Num_stackFrame):
            observation_stack[:, :, stack_frame] = observation_set[-1 -
                                                                   (self.Num_skipFrame * stack_frame)]
        observation_stack = np.uint8(observation_stack)

        return observation_stack, observation_set, state_stack, state_set

    # Resize input information
    def resize_input(self, env_info, observation_set, state_set):
        observation = env_info[1]
        observation_set.append(observation)
        # 使用观察组根据跳帧和堆叠帧的数量堆叠帧
        observation_stack = np.zeros(
            (self.img_size, self.img_size, self.Num_stackFrame))
        for stack_frame in range(self.Num_stackFrame):
            observation_stack[:, :, stack_frame] = observation_set[-1 -
                                                                   (self.Num_skipFrame * stack_frame)]
        del observation_set[0]
        observation_stack = np.uint8(observation_stack)

        state = env_info[0]
        state_set.append(state)
        state_stack = np.zeros((self.Num_stackFrame, self.Num_dataSize))
        for stack_frame in range(self.Num_stackFrame):
            state_stack[(self.Num_stackFrame - 1) - stack_frame,
                        :] = state_set[-1 - (self.Num_skipFrame * stack_frame)]

        del self.state_set[0]

        return observation_stack, observation_set, state_stack, state_set

    def get_progress(self, step, Epsilon):
        if step <= self.Num_start_training:
            # Obsersvation
            progress = 'Observing'
            Epsilon = 1

        elif step <= self.Num_start_training + self.Num_training:
            # Training
            progress = 'Training'

            # Decrease the epsilon value
            if self.Epsilon > self.Final_epsilon:
                Epsilon -= 1.0/self.Num_training

        elif step < self.Num_start_training + self.Num_training + self.Num_test:
            # Testing
            progress = 'Testing'
            Epsilon = 0

        else:
            # Finished
            progress = 'Finished'
            Epsilon = 0

        return progress, Epsilon

    def select_action(self, progress, sess, observation_stack, state_stack, Epsilon):
        if progress == "Observing":
            # 观察的情况下，随机选择一个action
            action = np.zeros([self.Num_action])
            action[random.randint(0, self.Num_action - 1)] = 1.0
        elif progress == "Training":
            if random.random() < Epsilon:
                action = np.zeros([self.Num_action])
                action[random.randint(0, self.Num_action - 1)] = 1
            else:
                # 否则，动作是具有最大Q值的动作
                action_actor = self.output_actor.eval(
                    feed_dict={self.x_image: [observation_stack], self.x_sensor: [state_stack]})
                p = action_actor.ravel()
                action[np.argmax(p)] = 1
        else:
            # 动作是具有最大Q值的动作
            action_actor = self.output_actor.eval(
                feed_dict={self.x_image: [observation_stack], self.x_sensor: [state_stack]})
            p = action_actor.ravel()
            action[np.argmax(p)] = 1
        return action

    def train(self, minibatch, w_batch, batch_index):
        observation_batch      = [batch[0] for batch in minibatch]      #生成batch[0]构成的列表
        state_batch            = [batch[1] for batch in minibatch]
        action_batch           = [batch[2] for batch in minibatch]
        reward_batch           = [batch[3] for batch in minibatch]
        observation_next_batch = [batch[4] for batch in minibatch]
        state_next_batch       = [batch[5] for batch in minibatch]
        terminal_batch         = [batch[6] for batch in minibatch]
        # print("observation_batch:",observation_batch,type(observation_batch))
        # print("state_batch:",state_batch,type(state_batch))
        # print("action_batch:",action_batch,type(action_batch))
        # print("reward_batch:",reward_batch,type(reward_batch))
        # print("observation_next_batch:",observation_next_batch,type(observation_next_batch))
        # print("state_next_batch:",state_next_batch,type(state_next_batch))
        # print("terminal_batch:",terminal_batch,type(terminal_batch))


        Q_batch = self.output_critic.eval(
                feed_dict = {self.x_image: observation_next_batch, self.x_sensor: state_next_batch})

        Q_target=[]

        for i in range(len(minibatch)):
            if terminal_batch[i] == True:
                Q_target.append([reward_batch[i]])
            else:
                Q_target.append([reward_batch[i] + self.Gamma * np.max(Q_batch)])
        Q=np.array(Q_target).reshape((-1,1))

        

        # v_ = self.sess.run(self.output_critic, feed_dict = {self.action_target: action_batch,
        #                                                                                                                 self.y_target: Q_target,
        #                                                                                                                 self.x_image: observation_batch,
        #                                                                                                                 self.x_sensor:state_batch,
        #                                                                                                                 self.w_is: w_batch,
        #                                                                                                                 self.output_critic_: [Q_target]})

        a=[]
        for ax in range(len(action_batch)):
            a.append(action_batch[ax][np.argmax(action_batch[ax])]+ax*self.Num_action)

        # print(reward_batch)
        # print(Q)
        # print(a)
        # print(w_batch)
             

        #NOTE：先计算TD_error，然后同时训练Actor和Critic网络
        # TD_error_batch, _ ,exp_v ,_= self.sess.run([self.TD_error, self.train_Critic,self.exp_v,self.train_Actor],feed_dict ={self.x_image: observation_batch,
        #                                                                                                                                                                 self.x_sensor:state_batch,
        #                                                                                                                                                                 self.r: reward_batch,
        #                                                                                                                                                                 self.output_critic_: Q,
        #                                                                                                                                                                 self.a:a,
        #                                                                                                                                                                 self.w_is: w_batch})
        TD_error_batch= self.sess.run([self.TD_error],feed_dict ={self.x_image: observation_batch,
                                                                                                                                                                        self.x_sensor:state_batch,
                                                                                                                                                                        self.r: reward_batch,
                                                                                                                                                                        self.output_critic_: Q,
                                                                                                                                                                        self.a:a,
                                                                                                                                                                        self.w_is: w_batch})
        print("TD",TD_error_batch)
        print("1",abs(TD_error_batch[0][2]))
        print("2",pow((abs(TD_error_batch[0][i_batch]) + self.eps), self.alpha))

        # Update TD_list

        for i_batch in range(len(batch_index)):
            self.TD_list[batch_index[i_batch]] = pow((abs(TD_error_batch[0][i_batch]) + self.eps), self.alpha)

        # Update Beta
        self.beta = self.beta + (1 - self.beta_init)/self.Num_training
        pass

    def Experience_Replay(self, observation, state, action, reward, next_observation, next_state, terminal):
        # If Replay memory is longer than Num_replay_memory, delete the oldest one
        if len(self.Replay_memory) >= self.Num_replay_memory:
            del self.Replay_memory[0]
            self.TD_list = np.delete(self.TD_list, 0)

        if self.progress == 'Exploring':
            self.Replay_memory.append(
                [observation, state, action, reward, next_observation, next_state, terminal])
            self.TD_list = np.append(self.TD_list, pow(
                (abs(reward) + self.eps), self.alpha))
        elif self.progress == 'Training':
            self.Replay_memory.append(
                [observation, state, action, reward, next_observation, next_state, terminal])
            # ################################################## PER ############################################################
            Q_batch = self.output_critic.eval(
                feed_dict={self.x_image: [next_observation], self.x_sensor: [next_state]})

            if terminal == True:
                Q_target = [reward]
            else:
                Q_target = [reward + self.Gamma * np.max(Q_batch)]
            np.array(Q_target).reshape((-1,1))

            # TD_error = self.TD_error.eval(feed_dict={self.action_target: [
            #                               action], self.y_target: y, self.x_image: [observation],self.x_sensor: [state]})[0]
            TD_error = self.TD_error.eval(feed_dict={self.r: [reward], self.x_image: [
                                          observation], self.x_sensor: [state], self.output_critic_: [Q_target]})

            #TODO：看看这个TD_error是0维还是1维

            #NOTE:这里的TD_error是critic网络的偏差，所以每次选取的都是让cirtic变准确的样本，之后可以使者修改，从而选取出使actor网络偏差较大的样本,其实td_error越大，actor网络的偏差应该也越大

            self.TD_list = np.append(self.TD_list, pow(
                (abs(TD_error) + self.eps), self.alpha))
            # ###################################################################################################################

    ################################################## PER ############################################################
    def prioritized_minibatch(self):
        # Update TD_error list
        TD_normalized = self.TD_list / np.linalg.norm(self.TD_list, 1)
        TD_sum = np.cumsum(TD_normalized)

        # Get importance sampling weights
        weight_is = np.power(
            (self.Num_replay_memory * TD_normalized), - self.beta)
        weight_is = weight_is / np.max(weight_is)

        # Select mini batch and importance sampling weights
        minibatch = []
        batch_index = []
        w_batch = []
        for i in range(self.Num_batch):
            rand_batch = random.random()
            TD_index = np.nonzero(TD_sum >= rand_batch)[0][0]
            batch_index.append(TD_index)
            w_batch.append(weight_is[TD_index])
            # minibatch.append(self.Replay_memory[TD_index])
            minibatch.append(np.array(self.Replay_memory)[TD_index])
        return minibatch, w_batch, batch_index
    ###################################################################################################################

    def save_model(self):
        # ------------------------------
        save_path = self.saver.save(
            self.sess, 'saved_networks/' + '10_D3QN_PER_image_add_sensor_empty_world_30m' + '_' + self.date_time + "/model.ckpt")
        # ------------------------------

    def main(self):

        reward_list = []

        # 随机种子
        random.seed(1000)
        np.random.seed(1000)
        tf.set_random_seed(1234)

        # 随机初始化起点和终点的位置
        while(True):
            randposition = 2 * self.d * \
                np.random.random_sample((2, 2)) - self.d
            if math.sqrt((randposition[0][0]-randposition[1][0])**2+(randposition[0][1]-randposition[1][1])**2) > 20.0:
                break

        env.reset_env(start=[randposition[0][0], randposition[0][1]], goal=[randposition[1][0], randposition[1][1]])
        env_info = env.get_env()
        # env.info为4维，第1维为相机消息，第2维为agent robot的self state，第3维为terminal，第4维为reward
        self.observation_stack, self.observation_set, self.state_stack, self.state_set = self.input_initialization(env_info)

        step_for_newenv = 0

        # Training & Testing
        while True:
            # Get Progress, train mode 获取当前的进度和train_mode的值
            self.progress, self.Epsilon = self.get_progress(
                self.step, self.Epsilon)

            # Select Actions 根据进度选取动作
            action = self.select_action(self.progress, self.sess, self.observation_stack, self.state_stack, self.Epsilon)
            action_in = np.argmax(action)
            cmd = [0.0, 0.0]
            v_cmd = action_dict[action_in][0]
            omiga_cmd = action_dict[action_in][1]
            cmd[0] = v_cmd
            cmd[1] = omiga_cmd
            env.step(cmd)

            # Get information for update
            env_info = env.get_env()

            self.next_observation_stack, self.observation_set, self.next_state_stack, self.state_set = self.resize_input(env_info, self.observation_set, self.state_set)  # 调整输入信息
            terminal = env_info[-2]  # 获取terminal
            reward = env_info[-1]  # 获取reward

            # Experience Replay
            self.Experience_Replay(self.observation_stack, self.state_stack, action,reward, self.next_observation_stack, self.next_state_stack, terminal)

            if self.progress == 'Training':
                # Train!!
                minibatch, w_batch, batch_index = self.prioritized_minibatch()
                # Training
                self.train(minibatch, w_batch, batch_index)

            # If progress is finished -> close!
            if self.progress == 'Finished' or self.episode == self.MAXEPISODES:
                print('Finished!!')
                break

            # Update information
            self.step += 1
            self.score += reward
            self.observation_stack = self.next_observation_stack
            self.state_stack = self.next_state_stack

            step_for_newenv = step_for_newenv + 1

            if step_for_newenv == self.MAXSTEPS:
                terminal = True

            # If terminal is True
            if terminal == True:
                self.save_model()
                step_for_newenv = 0
                # Print informations
                print('step:'+str(self.step)+'/'+'episode:'+str(self.episode)+'/'+'progress:' +self.progress+'/'+'epsilon:'+str(self.Epsilon)+'/'+'score:' + str(self.score))

                if self.progress == 'Training':
                    reward_list.append(self.score)
                    reward_array = np.array(reward_list)
                    # ------------------------------
                    np.savetxt('10_D3QN_PER_image_add_sensor_empty_world_30m_reward.txt', reward_array, delimiter=',')
                    # ------------------------------
                if self.progress != 'Observing':
                    self.episode += 1

                self.score = 0

                # Initialize game state
                # 随机初始化起点和终点的位置
                while(True):
                    randposition = 2 * self.d * \
                        np.random.random_sample((2, 2)) - self.d
                    if math.sqrt((randposition[0][0]-randposition[1][0])**2+(randposition[0][1]-randposition[1][1])**2) > 20.0:
                        break
                env.reset_env(start=[randposition[0][0], randposition[0][1]], goal=[
                              randposition[1][0], randposition[1][1]])
                env_info = env.get_env()
                self.observation_stack, self.observation_set, self.state_stack, self.state_set = self.input_initialization(env_info)


if __name__ == '__main__':
    agent = A2C()
    agent.main()
