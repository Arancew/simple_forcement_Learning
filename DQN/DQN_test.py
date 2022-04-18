import time
import torch
from torch import nn
import gym
import numpy as np


# 定义Q函数的神经网络
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
    def __init__(self, q_func, optimizer, n_act, e_grade=0.1, gamma=0.9):
        self.q_func = q_func

        self.criterion = nn.MSELoss()
        self.optimizer = optimizer

        self.epsilon = e_grade
        self.n_act = n_act
        self.gamma = gamma

    def predict(self, obs):
        Q_list = self.q_func(obs)
        # action = np.argmax(Q_list)
        action = int(torch.argmax(Q_list).detach().numpy())
        return action

    def act(self, obs):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_act)
        else:
            action = self.predict(obs)
        return action

    def learn(self, obs, action, reward, next_obs, done):
        predict_Q = self.q_func(obs)[action]
        target_Q = reward + (1 - float(done)) * self.gamma * self.q_func(next_obs).max()
        # 更新网络
        self.optimizer.zero_grad()
        loss = self.criterion(predict_Q, target_Q)
        loss.backward()
        self.optimizer.step()


# 训练模型
class TrainManager():
    def __init__(self, env, episode=1000, lr=0.001, gamma=0.9, e_grade=0.9):
        self.env = env
        self.episode = episode
        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.n
        q_func = MLP(n_obs, n_act)
        optimizer = torch.optim.Adam(q_func.parameters(), lr)
        self.agent = DQNAgent(
            q_func=q_func,
            optimizer=optimizer,
            n_act=n_act,
            gamma=gamma,
            e_grade=e_grade
        )

    def train_episode(self):
        total_reward = 0
        obs = self.env.reset()
        obs = torch.FloatTensor(obs)
        while True:
            action = self.agent.act(obs)
            next_obs, reward, done, _ = self.env.step(action)
            next_obs = torch.FloatTensor(next_obs)
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
            next_obs = torch.FloatTensor(next_obs)
            obs = next_obs
            total_reward += reward
            self.env.render()
            time.sleep(0.1)
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

env1 = gym.make("CartPole-v0")
tm = TrainManager(env1)
num=tm.train()
print('本轮游戏的最大值为%d' % num)
