import random
from collections import deque
import numpy as np

class Replay_Buffer:
  def __init__(self, buffer_size, seed=123):
    self.buffer_size = buffer_size
    self.count = 0
    self.buffer = deque()
    random.seed(seed)

  def add(self, state, action, reward, next_state):
    experience = (state, action, reward, next_state)
    if self.count < self.buffer_size:
      self.buffer.append(experience)
      self.count += 1
    else:
      self.buffer.popleft()
      self.buffer.append(experience)

  def get_size(self):
    return self.count

  def sample_batch(self, batch_size):
    batch = []

    if self.count < batch_size:
      batch = random.sample(self.buffer, self.count)
    else:
      batch = random.sample(self.buffer, batch_size)

    state_batch = np.array([_[0] for _ in batch])
    action_batch = np.array([_[1] for _ in batch])
    reward_batch = np.array([_[2] for _ in batch])
    next_state_batch = np.array([_[3] for _ in batch])

    return state_batch, action_batch, reward_batch, next_state_batch

  def clear(self):
    self.deque.clear()
    self.count = 0
