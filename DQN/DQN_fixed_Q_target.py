import random
import time
import torch
from torch import nn
import gym
import numpy as np
import collections
import copy
# 定义Q函数的神经网络
import torchUtils


class MLP(nn.Module):
    def __init__(self, obs_size, n_act):
        super().__init__()
        self.mlp = self.__mlp(obs_size, n_act)

    def __mlp(self, obs_size, n_act):
        return nn.Sequential(
            nn.Linear(obs_size, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, n_act),
        )

    def forward(self, x):
        return self.mlp(x)


# 定义强化学习模型
class DQNAgent():
    def __init__(self, q_func, optimizer, n_act, reply_buffer, batch_size, update_target_steps=100, reply_start_size=200,
                 e_grade=0.1,
                 gamma=0.9, ):
        self.pred_func = q_func
        self.target_func = copy.deepcopy(q_func)

        self.update_target_steps = update_target_steps

        self.criterion = nn.MSELoss()
        self.optimizer = optimizer

        self.epsilon = e_grade
        self.n_act = n_act
        self.gamma = gamma

        self.rb = reply_buffer
        self.batch_size = batch_size
        self.reply_start_size = reply_start_size
        self.global_step = 0

    def predict(self, obs):
        obs = torch.FloatTensor(obs)
        Q_list = self.pred_func(obs)
        # action = np.argmax(Q_list)
        action = int(torch.argmax(Q_list).detach().numpy())
        return action

    def act(self, obs):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_act)
        else:
            action = self.predict(obs)
        return action

    def learn_batch(self, batch_obs, batch_action, batch_reward, batch_next_obs, batch_done):
        pred_Vs = self.pred_func(batch_obs)
        action_onehot = torchUtils.one_hot(batch_action, self.n_act)
        predict_Q = (pred_Vs * action_onehot).sum(1)
        # target_Q
        next_pred_Vs = self.target_func(batch_next_obs)
        best_V = next_pred_Vs.max(1)[0]
        target_Q = batch_reward + (1 - batch_done) * self.gamma * best_V

        # 更新参数
        self.optimizer.zero_grad()
        loss = self.criterion(predict_Q, target_Q)
        loss.backward()
        self.optimizer.step()

    def learn(self, obs, action, reward, next_obs, done):
        self.global_step += 1
        self.rb.append((obs, action, reward, next_obs, done))
        # 每4次学习
        if len(self.rb) > self.reply_start_size and self.global_step % 4 == 0:
            self.learn_batch(*self.rb.sample(self.batch_size))
        if self.global_step % self.update_target_steps == 0:
            self.sync_target()

    def sync_target(self):
        for target_parma, parma in zip(self.target_func.parameters(), self.pred_func.parameters()):
            target_parma.data.copy_(parma.data)


# 经验池
class replyBuffer():
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*mini_batch)
        #  转换成张量的形式
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_obs_batch = torch.FloatTensor(next_obs_batch)
        done_batch = torch.FloatTensor(done_batch)
        obs_batch = torch.FloatTensor(obs_batch)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self):
        return len(self.buffer)


# 训练模型
class TrainManager():
    def __init__(self, env, episode=1000, lr=0.001, gamma=0.9, e_grade=0.9, memory_size=2000, reply_start_size=200,
                 batch_size=23):
        self.env = env
        self.episode = episode
        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.n
        q_func = MLP(n_obs, n_act)
        optimizer = torch.optim.Adam(q_func.parameters(), lr)
        rb = replyBuffer(memory_size)
        self.agent = DQNAgent(
            q_func=q_func,
            optimizer=optimizer,
            n_act=n_act,
            gamma=gamma,
            e_grade=e_grade,
            reply_buffer=rb,
            reply_start_size=reply_start_size,
            batch_size=batch_size
        )

    def train_episode(self):
        total_reward = 0
        obs = self.env.reset()
        while True:
            action = self.agent.act(obs)
            next_obs, reward, done, _ = self.env.step(action)
            self.agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            if done:
                break
        return total_reward

    def test_episode(self):
        total_reward = 0
        obs = self.env.reset()
        obs = torch.FloatTensor(obs)
        while True:
            action = self.agent.predict(obs)
            next_obs, reward, done, _ = self.env.step(action)
            obs = next_obs
            total_reward += reward
            self.env.render()
            time.sleep(0.005)
            if done:
                break
        return total_reward

    def train(self):
        num = 0
        for epoch in range(self.episode):

            ep_reward = self.train_episode()
            print('Epsode %s: reward= %.1f' % (epoch, ep_reward))
            if epoch % 100 == 0:
                test_reward = self.test_episode()
                if num < test_reward:
                    num = test_reward
                print('test_reward= %.1f' % test_reward)
        return num


env1 = gym.make("CartPole-v1")
tm = TrainManager(env1)
num = tm.train()
print('本轮游戏的最大值为%d' % num)
