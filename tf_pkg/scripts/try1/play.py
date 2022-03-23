import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import cv2
import os
import math
from A2C import A2C
from gazebo_env_D3QN_PER_image_add_sensor_empty_world_30m import envmodel

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 默认显卡0

env = envmodel()

# 动作指令集---> v,w
action_dict = {0: [1.0, -1.0], 1: [1.0, -0.5], 2: [1.0, 0.0],
               3: [1.0, 0.5], 4: [1.0, 1.0], 5: [0.5, -1.0],
               6: [0.5, 0.0], 7: [0.5, 1.0], 8: [0.0, -1.0],
               9: [0.0, 0.0], 10: [0.0, 1.0]}


class play:
    def init(self):
        # Get parameters
        self.progress = ''
        self.Num_action = len(action_dict)

        # Initial parameters
        # ------------------------------
        self.Num_start_training = 50
        self.Num_training = 50000
        # ------------------------------
        self.Num_test = 0
        self.Lr_A = 0.0001
        self.Lr_C = 0.000002
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

        # Parameter for LSTM
        self.Num_dataSize = 364  # 360 sensor size + 4 self state size



        self.sess, self.saver = self.init_sess()

        self.a2c=A2C(self.sess)

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

        # Load the file if the saved file exists
        saver = tf.train.Saver()
        # check_save = input('Load Model? (1=yes/2=no): ')

        # if check_save == 1:
        #     # Restore variables from disk.
        #     saver.restore(sess, self.load_path + "/model.ckpt")
        #     print("Model restored.")

        #     check_train = input('Inference or Training? (1=Inference / 2=Training): ')
        #     if check_train == 1:
        #         self.Num_start_training = 0
        #         self.Num_training = 0

        return sess, saver

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
                action = np.zeros([self.Num_action])
                action_actor = self.output_actor.eval(
                    feed_dict={self.x_image: [observation_stack], self.x_sensor: [state_stack]})
                p = action_actor.ravel()
                action[np.argmax(p)] = 1
            # 观察的情况下，随机选择一个action
            # action = np.zeros([self.Num_action])
            # action[random.randint(0, self.Num_action - 1)] = 1.0
        else:
            # 动作是具有最大Q值的动作
            action_actor = self.output_actor.eval(
                feed_dict={self.x_image: [observation_stack], self.x_sensor: [state_stack]})
            p = action_actor.ravel()
            action[np.argmax(p)] = 1
        return action


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
            self.progress, self.Epsilon = self.get_progress(self.step, self.Epsilon)

            # Select Actions 根据进度选取动作
            action = self.select_action(self.progress, self.sess, self.observation_stack, self.state_stack, self.Epsilon)
            action_in = np.argmax(action)
            env.step(action_dict[action_in])

            # Get information for update
            env_info = env.get_env()

            self.next_observation_stack, self.observation_set, self.next_state_stack, self.state_set = self.resize_input(env_info, self.observation_set, self.state_set)  # 调整输入信息
            terminal = env_info[-2]  # 获取terminal
            reward = env_info[-1]  # 获取reward

            if self.progress == 'Training':
                self.a2c.train()

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

        pass


if __name__ == '__main__':
    agent = play()
    agent.main()