import gym
import numpy as np

import gridworld

env = gym.make("CliffWalking-v0")
env = gridworld.CliffWalkingWapper(env)
state = env.reset()
cnt = 0
while True:
    cnt += 1
    action = np.random.randint(0, 3)
    state, reward, done, info = env.step(action)
    # state 交互后的状态
    # reward 奖励
    # done 完成与否
    # 一点 信息
    env.render()
    if done:
        break
print('一共走了%d步' % cnt)

# cliffWalking_by_Sarsa.py