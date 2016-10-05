import gym
import tensorflow as tf
import tflearn
import numpy as np

from actor import Actor
from critic import Critic
from replay_buffer import Replay_Buffer


num_episodes = 50
num_steps = 200
actor_learning_rate = .000001
critic_learning_rate = .0000001
temperature = .0001
discount_factor = .8

buffer_size = 10000
batch_size = 64
random_seed = 123


def main():
  with tf.Session() as sess:
    env = gym.make('Humanoid-v1')

    np.random.seed(123)
    env.seed(123)
    tf.set_random_seed(123)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_bounds = env.action_space.high

    actor = Actor(sess, state_size, action_size, action_bounds, actor_learning_rate, temperature)

    critic = Critic(sess, state_size, action_size, critic_learning_rate, temperature, actor.get_trainable_vars())

    train(sess, env, actor, critic)


def train(sess, env, actor, critic):

  replay_buffer = Replay_Buffer(buffer_size, random_seed)

  sess.run(tf.initialize_all_variables())

  actor.update_target_network()
  critic.update_target_network()

  for episode in xrange(num_episodes):

    episode_reward = 0
    episode_avg_max_q = 0

    state = env.reset()

    for step in xrange(num_steps):
      env.render()

      action = actor.predict(np.reshape(state, (1, actor.state_size))) + (1/(1 + step + episode))

      if((step+1) % (episode+1) == 0):
        action = env.action_space.sample()

      next_state, reward, _, __ = env.step(action)

      print(' at step ' +  str(step) + ' in episode ' + str(episode) +  ' had reward ' + str(reward))

      replay_buffer.add(np.reshape(state, actor.state_size), np.reshape(action, actor.action_size), reward, np.reshape(next_state, actor.state_size))

      if(replay_buffer.get_size() > batch_size):

        state_batch, action_batch, reward_batch, next_state_batch = replay_buffer.sample_batch(batch_size)

        target_q = critic.predict_target(next_state_batch, actor.predict(next_state_batch))

        y_i = []
        for k in range(batch_size):
          y_i.append(reward_batch[k] + discount_factor * target_q[k])

        predicted_q = critic.train(state_batch, action_batch, np.reshape(y_i, (batch_size, 1)))

        episode_avg_max_q = np.amax(predicted_q[0])

        actor_outputs = actor.predict(state_batch)
        gradients = critic.action_gradients(state_batch, action_batch)
        actor.train(state_batch, gradients[0])

        actor.update_target_network()
        critic.update_target_network()

      state = next_state
      episode_reward += reward

      if(step == num_steps-1):
        print('reward: ', episode_reward, 'Qmax: ', episode_avg_max_q)


