# import imp
import tensorflow as tf
import random
import  numpy as np
# import tensorflow as tf


import cv2
import os
import A2C
from env_mbot import envmodel




LR_A = 0.01    # learning rate for actor
LR_C = 0.01     # learning rate for critic
MAX_EPISODE = 500
DISPLAY_REWARD_THRESHOLD = 0.1  # renders environment if total episode reward is greater then this threshold

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 默认显卡0

env=envmodel()

# 动作指令集---> v,w
action_dict = {0: [1.0, -1.0], 1: [1.0, -0.5], 2: [1.0, 0.0],
               3: [1.0, 0.5], 4: [1.0, 1.0], 5: [0.5, -1.0],
               6: [0.5, 0.0], 7: [0.5, 1.0], 8: [0.0, -1.0],
               9: [0.0, 0.0], 10: [0.0, 1.0]}

sess = tf.Session()
actor = A2C.Actor(sess, n_features=6, n_actions=11, lr=LR_A)
critic = A2C.Critic(sess, n_features=6, lr=LR_C)
saver = tf.train.Saver()  #声明ta.train.Saver()类用于保存
sess.run(tf.global_variables_initializer())

agent_start_x=random.randint(-3, 3)
agent_start_y=random.randint(-3, 3)
goal_x=random.randint(-3, 3)
goal_y=random.randint(-3, 3)

agent_start_x=5
agent_start_y=5
goal_x=-5
goal_y=-5



def train():
    for i_episode in range(MAX_EPISODE):

        env.reset_env(start=[agent_start_x,agent_start_y],goal=[goal_x,goal_y])
        observation = np.ones(6)
        observation[0]=agent_start_x
        observation[1]=agent_start_y
        observation[2]=goal_x
        observation[3]=goal_y
        observation=observation.reshape(-1)
        track_r = []
        for i in range(300):
            action = actor.choose_action(observation)
            env.step(action_dict[action])
            observation_,reward,done = env.get_env()

            td_error = critic.learn(observation, reward, observation_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(observation, action, td_error)     # true_gradient = grad[logPi(s,a) * td_error]
            observation=observation_
            track_r.append(reward)

            if done or i>=299:
                ep_rs_sum=sum(track_r)
                print("episode:", i_episode, "  reward:", ep_rs_sum)
                break

    print("123")
    save_path = saver.save(sess,'save1_/filename.ckpt')#保存路径为相对路径的save文件夹,保存名为filename.ckpt
    print ("[+] Model saved in file: %s" % save_path)


def rand():
    env.reset_env(start=[5.0, 5.0], goal=[-5.0,-5.0])
    for i_episode in range(MAX_EPISODE):
        action = action_dict[random.randint(0, 9)]
        print(action)
        env.step(action)
        observation_, agent_x, agent_y = env.get_env()
        print(observation_[1])
        print(observation_[2])

if __name__ == '__main__':
    train()
    pass