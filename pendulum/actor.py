#Embedded file name: actor.py
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
        self.update_target_params = [ self.target_params[i].assign(tf.mul(self.params[i], self.temperature) + tf.mul(self.target_params[i], 1.0 - self.temperature)) for i in range(len(self.target_params)) ]
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_size])
        self.actor_gradients = tf.gradients(self.outputs, self.params, -self.action_gradient)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.params))
        self.num_trainable_vars = len(self.params) + len(self.target_params)

    def createNetwork(self):
        inputs = tflearn.input_data(shape=[None, self.state_size])
        net = tflearn.fully_connected(inputs, 400, activation='relu')
        net = tflearn.fully_connected(net, 300, activation='relu')
        weights_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        unscaled_outputs = tflearn.fully_connected(net, self.action_size, activation='tanh', weights_init=weights_init)
        outputs = tf.mul(unscaled_outputs, self.action_bounds)
        return (inputs, unscaled_outputs, outputs)

    def train(self, inputs, action_gradient):
        self.sess.run(self.optimize, feed_dict={self.inputs: inputs,
         self.action_gradient: action_gradient})

    def predict(self, inputs):
        return self.sess.run(self.outputs, feed_dict={self.inputs: inputs})

    def predict_target(self, inputs):
        return self.sess.run(self.target_outputs, feed_dict={self.target_inputs: inputs})

    def update_target_network(self):
        self.sess.run(self.update_target_params)

    def get_trainable_vars(self):
        return self.num_trainable_vars
