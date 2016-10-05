import gym
import numpy as np
from random import randint
from random import uniform
import math

# create theta with 5 paramaters, where paramater 0 is a constant and 1-4 correspond to observation[0-3]


# for i in range(theta.size):
#   theta[i, 0] = randint(-1, 1) * uniform(0, 1)

def run(theta):
  env = gym.make('CartPole-v0')
  # env.monitor.start('../cartpole-experiment-0', force=True)

  best_reward = 1
  test_theta = np.ones((4,1))
  episode_reward = 0
  scoreTotal = 0

  for i_episode in range(100):

    observation = env.reset()


    if(best_reward <= episode_reward/3):
      best_reward = episode_reward/3
      theta = test_theta

    if(episode_reward < 195):

      if(i_episode > 1 and episode_reward < 10):
        theta = np.random.rand(4, 1)

      param_to_change = randint(0, 3)

      test_theta = theta

      test_theta[param_to_change] = test_theta[param_to_change] + uniform(-1, 1) * randint(1,5) * .2

    episode_reward = 0

    for t in range(200):
      env.render()

      z = np.dot(actionableObservation(observation), test_theta)
      action = 1/(1+ math.exp(-z))
      # print(action)

      if(action >= .5):
        action = 1
      else:
        action = 0
      observation, reward, done, info = env.step(action)
      episode_reward = episode_reward + reward
      if (done or t == 199):
        scoreTotal += episode_reward
        print('episode score:' + str(t)+ '| average score:' + str(scoreTotal/(i_episode + 1)))
        break
  # env.monitor.close()


def actionableObservation(observation):

  actionable = np.zeros((1, 4))

  # actionable[0,0] = 1
  actionable[0,0] = observation[0]
  actionable[0,1] = observation[1]
  actionable[0,2] = observation[2]
  actionable[0,3] = observation[3]

  return actionable

