import time

import gym
import numpy as np
import gridworld


class SarsaAgent():
    def __init__(self, n_states, n_act, e_grade=0.1, lr=0.1, gamma=0.9):
        self.Q = np.zeros((n_states, n_act))
        self.e_grade = e_grade
        self.n_act = n_act
        self.lr = lr
        self.gamma = gamma

    def predict(self, state):
        Q_list = self.Q[state, :]
        # action = np.argmax(Q_list)
        action = np.random.choice(np.flatnonzero(Q_list == Q_list.max()))
        return action

    def act(self, state):
        if np.random.uniform(0, 1) < self.e_grade:
            action = np.random.choice(self.n_act)
        else:
            action = self.predict(state)
        return action

    def learn(self, state, action, reward, next_state, next_action, done):
        cur_Q = self.Q[state, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] += self.lr * (target_Q - cur_Q)


def train_episode(env, agent):
    total_reward = 0
    state = env.reset()
    action = agent.act(state)
    while True:
        next_state, reward, done, _ = env.step(action)
        next_action = agent.act(next_state)
        agent.learn(state, action, reward, next_state, next_action, done)
        action = next_action
        state = next_state
        total_reward += reward
        if done:
            break
    return total_reward


def test_episode(env, agent):
    total_reward = 0
    state = env.reset()
    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        env.render()
        time.sleep(0.5)
        if done:
            break
    return total_reward


def train(env, episodes=500, e_grade=0.1, lr=0.1, gamma=0.9):
    agent = SarsaAgent(
        n_states=env.observation_space.n,
        n_act=env.action_space.n,
        e_grade=e_grade,
        lr=lr,
        gamma=gamma,
    )
    for epoch in range(episodes):
        ep_reward = train_episode(env, agent)
        print('Epsode %s: reward= %.1f' % (epoch, ep_reward))
    test_reward = test_episode(env, agent)
    print('test_reward= %.1f' % test_reward)


env = gym.make("CliffWalking-v0")
env = gridworld.CliffWalkingWapper(env)
train(env)
