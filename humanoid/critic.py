import tensorflow as tf
import tflearn

class Critic:
  def __init__(self, sess, state_size, action_size, learning_rate, temperature, num_actor_vars):
    self.sess = sess
    self.state_size = state_size
    self.action_size = action_size
    self.learning_rate = learning_rate
    self.temperature = temperature

    self.inputs, self.action, self.outputs = self.createNetwork()
    self.params = tf.trainable_variables()[num_actor_vars:]

    self.target_inputs, self.target_action, self.target_outputs = self.createNetwork()
    self.target_params = tf.trainable_variables()[(len(self.params) + num_actor_vars):]

    self.update_target_params = [self.target_params[i].assign(tf.mul(self.params[i], self.temperature) + tf.mul(self.target_params[i], 1. - self.temperature)) for i in range(len(self.target_params))]

    self.predicted_q = tf.placeholder(tf.float32, [None, 1])

    # calculate the cost
    self.loss = tflearn.mean_square(self.predicted_q, self.outputs)

    # optimize to reduce the cost
    self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    # assemble the gradients: d(outputs)/d(parameters)
    self.action_gradient = tf.gradients(self.outputs, self.action)

  # create the critic network
  def createNetwork(self):
    inputs = tflearn.input_data(shape=[None, self.state_size])
    action = tflearn.input_data(shape=[None, self.action_size])
    net = tflearn.fully_connected(inputs, 400, activation='relu')

    t1 = tflearn.fully_connected(net, 1000)
    t2 = tflearn.fully_connected(action, 1000)

    net = tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')


    weights_init = tflearn.initializations.uniform(minval=-self.temperature, maxval=self.temperature)
    outputs = tflearn.fully_connected(net, 1, weights_init=weights_init)
    return inputs, action, outputs

# optimize the parameter
  def train(self, inputs, action, predicted_q):
    return self.sess.run([self.outputs, self.optimize], feed_dict={self.inputs: inputs, self.action: action, self.predicted_q: predicted_q})

  # given inputs and actions, using the network, return the predicted reward
  def predict(self, inputs, action):
    return self.sess.run(self.outputs, feed_dict={self.inputs: inputs, self.action: action})

  # given inputs and actions, using the target network, return the predicted reward
  def predict_target(self, inputs, action):
    return self.sess.run(self.target_outputs, feed_dict={self.target_inputs: inputs, self.target_action:action})

  # return the gradients given the inputs and an action
  def action_gradients(self, inputs, action):
    return self.sess.run(self.action_gradient, feed_dict={self.inputs: inputs, self.action: action})

  # update the params of the target network
  def update_target_network(self):
    self.sess.run(self.update_target_params)
