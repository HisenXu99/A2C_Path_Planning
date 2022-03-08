import tensorflow as tf
import random
import numpy as np
import math
from env_mbot import envmodel
import A2C


env = envmodel()
LR_A=0.001
LR_C=0.01

# 动作指令集---> v,w
action_dict = {0: [1.0, -1.0], 1: [1.0, -0.5], 2: [1.0, 0.0],
               3: [1.0, 0.5], 4: [1.0, 1.0], 5: [0.5, -1.0],
               6: [0.5, 0.0], 7: [0.5, 1.0], 8: [0.0, -1.0],
               9: [0.0, 0.0], 10: [0.0, 1.0]}

class Process:
    def __init__(self) -> None:

        self.sess = tf.Session()
        self.actor = A2C.Actor(self.sess, n_features=6, n_actions=11, lr=LR_A)
        self.critic = A2C.Critic(self.sess, n_features=6, lr=LR_C)


        # Define the distance from start point to goal point
        self.d = 15.0


        pass

    # Initialize input
    def input_initialization(self, env_info):
        state     = env_info[0]  # laser info + self state
        state_set = []
        for i in range(self.Num_skipFrame * self.Num_stackFrame):
            state_set.append(state)
        state_stack = np.zeros((self.Num_stackFrame, self.Num_dataSize))
        for stack_frame in range(self.Num_stackFrame):
            state_stack[(self.Num_stackFrame - 1) - stack_frame, :] = state_set[-1 - (self.Num_skipFrame * stack_frame)]

        observation = env_info[1]  # image info
        observation_set = []       
        for i in range(self.Num_skipFrame * self.Num_stackFrame):
            observation_set.append(observation)      
        # 使用观察组根据跳帧和堆叠帧的数量堆叠帧
        observation_stack = np.zeros((self.img_size, self.img_size, self.Num_stackFrame))
        # print("shape of observation stack={}".format(observation_stack.shape))    
        for stack_frame in range(self.Num_stackFrame):
            observation_stack[:, :, stack_frame] = observation_set[-1 - (self.Num_skipFrame * stack_frame)]
        observation_stack = np.uint8(observation_stack)
        
        return observation_stack, observation_set, state_stack, state_set

    def select_action(self, progress, sess, observation_stack, state_stack, Epsilon):
        pass

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

    def train(self, minibatch, w_batch, batch_index):
        # Select minibatch
        # Num_batch = 32
        # minibatch = random.sample(Replay_memory, self.Num_batch)  # 从 Replay_memory 中随机获取 Num_batch 个元素，作为一个片断返回

        # Save the each batch data
        observation_batch      = [batch[0] for batch in minibatch]
        state_batch            = [batch[1] for batch in minibatch]
        action_batch           = [batch[2] for batch in minibatch]
        reward_batch           = [batch[3] for batch in minibatch]
        observation_next_batch = [batch[4] for batch in minibatch]
        state_next_batch       = [batch[5] for batch in minibatch]
        terminal_batch         = [batch[6] for batch in minibatch]


        td_error = self.critic.learn(state_batch, reward_batch, state_next_batch)  # gradient = grad[r + gamma * V(s_) - V(s)]
        self.actor.learn(state_batch, action_batch, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        pass

    def Experience_Replay(self, observation, state, action, reward, next_observation, next_state, terminal):
		# If Replay memory is longer than Num_replay_memory, delete the oldest one
		if len(self.Replay_memory) >= self.Num_replay_memory:
			del self.Replay_memory[0]
			self.TD_list = np.delete(self.TD_list, 0)

		if self.progress == 'Exploring':
			self.Replay_memory.append([observation, state, action, reward, next_observation, next_state, terminal])
			self.TD_list = np.append(self.TD_list, pow((abs(reward) + self.eps), self.alpha))
		elif self.progress == 'Training':
			self.Replay_memory.append([observation, state, action, reward, next_observation, next_state, terminal])
			# ################################################## PER ############################################################
			Q_batch = self.output_target.eval(feed_dict = {self.x_image: [next_observation],self.x_sensor:[next_state]})

			if terminal == True:
				y = [reward]
			else:
				y = [reward + self.Gamma * np.max(Q_batch)]

			TD_error = self.TD_error.eval(feed_dict = {self.action_target: [action], self.y_target: y, self.x_image: [observation], self.x_sensor:[state]})[0]
			self.TD_list = np.append(self.TD_list, pow((abs(TD_error) + self.eps), self.alpha))
			# ###################################################################################################################

    ################################################## PER ############################################################
    def prioritized_minibatch(self):
        # Update TD_error list
        TD_normalized = self.TD_list / np.linalg.norm(self.TD_list, 1)
        TD_sum = np.cumsum(TD_normalized)

        # Get importance sampling weights
        weight_is = np.power((self.Num_replay_memory * TD_normalized), - self.beta)
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




    def main(self):

        reward_list = []

        # 随机种子	????????????????????????????????????
        random.seed(1000)
        np.random.seed(1000)
        tf.set_random_seed(1234)

        # 随机初始化起点和终点的位置
        while(True):
            #随机生成2*2维数组，取值为[-self.d,self.d) ,self.d==15
            randposition = 2 * self.d * np.random.random_sample((2, 2)) - self.d
            if math.sqrt((randposition[0][0]-randposition[1][0])**2+(randposition[0][1]-randposition[1][1])**2) > 20.0:
                break

        env.reset_env(start=[randposition[0][0], randposition[0][1]], goal=[randposition[1][0], randposition[1][1]])
        env_info = env.get_env()
        # env.info为4维，第1维为agent robot的self state和laser ，第2维为相机消息，第3维为terminal，第4维为reward
        #将state与observation（image）信息初始化
        self.observation_stack, self.observation_set ,self.state_stack, self.state_set= self.input_initialization(env_info)

        step_for_newenv = 0

        # Training & Testing
        while True:
            # Get Progress, train mode 获取当前的进度和train_mode的值
            self.progress, self.Epsilon = self.get_progress(self.step, self.Epsilon)

            # Select Actions 根据进度选取动作
            action, Q_value = self.select_action(self.progress, self.sess, self.observation_stack, self.state_stack, self.Epsilon)
            action_in = np.argmax(action)
            env.step(action_dict[action_in])
            # cmd       = [0.0, 0.0] 
            # v_cmd     = action_dict[action_in][0]
            # omiga_cmd = action_dict[action_in][1]
            # cmd[0]=v_cmd
            # cmd[1]=omiga_cmd
            # env.step(cmd)

            # Get information for update
            env_info = env.get_env()
            self.next_observation_stack, self.observation_set, self.next_state_stack, self.state_set = self.resize_input(env_info, self.observation_set, self.state_set)  # 调整输入信息
            terminal = env_info[-2]  # 获取terminal
            reward = env_info[-1]  # 获取reward

            # Experience Replay把经验信息存到经验回放池中
            self.Experience_Replay(self.observation_stack, self.state_stack, action, reward, self.next_observation_stack, self.next_state_stack, terminal)

            if self.progress == 'Training':
                # # Update target network
                # if self.step % self.Num_update == 0:
                #     self.assign_network_to_target()    
                # Train!! 

                minibatch, w_batch, batch_index  = self.prioritized_minibatch()

                # Training
                self.train(minibatch, w_batch, batch_index)

            # If progress is finished -> close! 
            if self.progress == 'Finished' or self.episode==self.MAXEPISODES:
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

            if terminal == True:
                # # if self.episode % 10 == 0:
                # if self.progress == 'Training' and self.episode % 5 == 0:
                #     self.save_model()
                self.save_model()

                step_for_newenv = 0
                # Print informations
                print('step:'+str(self.step)+'/'+'episode:'+str(self.episode)+'/'+'progress:'+self.progress+'/'+'epsilon:'+str(self.Epsilon)+'/'+'score:'+ str(self.score))

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
                    randposition = 2 * self.d * np.random.random_sample((2, 2)) - self.d
                    if math.sqrt((randposition[0][0]-randposition[1][0])**2+(randposition[0][1]-randposition[1][1])**2) > 20.0:
                        break
                env.reset_env(start=[randposition[0][0], randposition[0][1]], goal=[randposition[1][0], randposition[1][1]])
                env_info = env.get_env()
                self.observation_stack, self.observation_set, self.state_stack, self.state_set = self.input_initialization(env_info)


if __name__ == '__main__':
	agent =Process()
	agent.main()