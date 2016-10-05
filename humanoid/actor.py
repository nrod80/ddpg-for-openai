import tensorflow as tf
import tflearn

class Actor:
  def __init__(self, sess, state_size, action_size, action_bounds, learning_rate, temperature):
    self.sess = sess
    self.state_size = state_size
    self.action_size = action_size
    self.action_bounds = action_bounds
    self.learning_rate = learning_rate
    self.temperature = temperature

    self.inputs, self.unscaled_outputs, self.outputs = self.createNetwork()
    self.params = tf.trainable_variables()

    self.target_inputs, self.target_unscaled_outputs, self.target_outputs = self.createNetwork()
    self.target_params = tf.trainable_variables()[len(self.params):]

    self.update_target_params = [self.target_params[i].assign(tf.mul(self.params[i], self.temperature) + tf.mul(self.target_params[i], 1. - self.temperature)) for i in range(len(self.target_params))]

    # placeholder for the derivitive of the actions
    self.action_gradient = tf.placeholder(tf.float32, [None, self.action_size])

    # assemble the gradients: d(outputs)/d(parameters), and stores the gradients in -action_gradient
    self.actor_gradients = tf.gradients(self.outputs, self.params, -self.action_gradient)

    # an optimization function, with an applied gradient of the actor_gradients zipped with the parameters
    self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.params))

    # the number of trainable paramters (including normal and target)
    self.num_trainable_vars = len(self.params) + len(self.target_params)

  # contstructs the network
  def createNetwork(self):
    # the state observation
    inputs = tflearn.input_data(shape=[None, self.state_size])

    # hidden layer one of the network
    net = tflearn.fully_connected(inputs, 400, activation='relu')

    # hidden layer two of the network
    net = tflearn.fully_connected(net, 1000, activation='relu')

    # initialization of the weights, with symmetry breaking by temperature
    weights_init = tflearn.initializations.uniform(minval=-.0001, maxval=.0001)

    # ouput layer of the network
    unscaled_outputs = tflearn.fully_connected(net, self.action_size, activation='tanh', weights_init=weights_init)

    # scale the actions to be within the bounds of the expected action
    outputs = tf.mul(unscaled_outputs, self.action_bounds)

    return inputs, unscaled_outputs, outputs

  # optimize the params of the network
  def train(self, inputs, action_gradient):
    self.sess.run(self.optimize, feed_dict={self.inputs: inputs, self.action_gradient: action_gradient})

  # get the prediction based on network
  def predict(self, inputs):
    return self.sess.run(self.outputs, feed_dict={self.inputs: inputs})

  # get the prediction based on the target network
  def predict_target(self, inputs):
    return self.sess.run(self.target_outputs, feed_dict={self.target_inputs: inputs})

  # update the params in the target network
  def update_target_network(self):
    self.sess.run(self.update_target_params)

  # get the number of trainable variables
  def get_trainable_vars(self):
    return self.num_trainable_vars
