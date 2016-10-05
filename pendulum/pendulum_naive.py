import gym
import numpy as np
import math
import random


def run():
  env = gym.make('Pendulum-v0')

  test_theta = theta = np.random.rand(3);
  # print(test_theta)
  episode_reward = -10000
  best_reward = -10000


  for episode in range(300):
    # print(theta)
    # print(test_theta)
    observation = env.reset()
    action = np.zeros((observation.size))
    # print(action, test_theta)
    # print(episode_reward, best_reward)
    if(episode_reward > best_reward):
      print(best_reward, episode_reward)
      best_reward = episode_reward
      theta = test_theta
      # print(best_reward)

    test_theta = theta + (np.random.rand(3) * (1 - (2 * np.random.randint(0,2))))


    episode_reward = 0
    for frame in range(500):
      env.render()

      z = (test_theta * observation)
      for n in range(test_theta.size):
        g = round(1/(1+math.exp(z[n])),1)
        # print(g * 10 - 2)
        action[n] = g*10-2
      print(action)
      observation, reward, done, info = env.step(action)
      print(observation)
      episode_reward = episode_reward + reward
      if (done or frame == 199):
        # print(episode_reward)
        break
run()


# observation: length 3
