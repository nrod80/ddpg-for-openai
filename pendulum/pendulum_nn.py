#Embedded file name: pendulum_nn.py
import gym
import tensorflow as tf
import numpy as np
import tflearn
import time
from actor import Actor
from critic import Critic
from replay_buffer import Replay_Buffer
max_episodes = 1000
max_steps = 250
actor_learning_rate = 0.001
critic_learning_rate = 0.0001
discount_factor = 0.99
temperature = 0.001
batch_size = 64
buffer_size = 10000
random_seed = 123

def main():
    with tf.Session() as sess:
        env = gym.make('Pendulum-v0')
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        env.seed(random_seed)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        action_bound_high = env.action_space.high
        action_bound_low = env.action_space.low
        if action_bound_high == -action_bound_low:
            action_bounds = action_bound_high
        actor = Actor(sess, state_size, action_size, action_bounds, actor_learning_rate, temperature)
        critic = Critic(sess, state_size, action_size, critic_learning_rate, temperature, actor.get_trainable_vars())
        train(sess, env, actor, critic)


def train(sess, env, actor, critic):
    sess.run(tf.initialize_all_variables())
    actor.update_target_network()
    critic.update_target_network()
    replay_buffer = Replay_Buffer(buffer_size, random_seed)
    total_episode_reward = 0
    for i in xrange(max_episodes):
        state = env.reset()
        current_episode_reward = 0
        current_episode_ave_max_q = 0
        for j in xrange(max_steps):
            env.render()
            action = actor.predict(np.reshape(state, (1, actor.state_size))) + 1 / (1 + i + j)
            next_state, reward, done, info = env.step(action[0])
            replay_buffer.add(np.reshape(state, (actor.state_size,)), np.reshape(action, (actor.action_size,)), reward, np.reshape(next_state, (actor.state_size,)))
            if replay_buffer.get_size() > batch_size:
                state_batch, action_batch, reward_batch, next_state_batch = replay_buffer.sample_batch(batch_size)
                target_q = critic.predict_target(next_state_batch, actor.predict_target(next_state_batch))
                y_i = []
                for k in range(batch_size):
                    y_i.append(reward_batch[k] + discount_factor * target_q[k])

                predicted_q, _ = critic.train(state_batch, action_batch, np.reshape(y_i, (batch_size, 1)))
                current_episode_ave_max_q += np.amax(predicted_q)
                actor_outputs = actor.predict(state_batch)
                grads = critic.action_gradients(state_batch, actor_outputs)
                actor.train(state_batch, grads[0])
                actor.update_target_network()
                critic.update_target_network()
            state = next_state
            current_episode_reward += reward
            if j == 199:
                total_episode_reward += current_episode_reward
                print 'episode score:' + str(current_episode_reward) + '| average score:' + str(total_episode_reward / (i + 1))
