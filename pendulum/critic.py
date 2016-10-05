#Embedded file name: critic.py
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
        self.target_params = tf.trainable_variables()[len(self.params) + num_actor_vars:]
        self.update_target_params = [ self.target_params[i].assign(tf.mul(self.params[i], self.temperature) + tf.mul(self.target_params[i], 1.0 - self.temperature)) for i in range(len(self.target_params)) ]
        self.predicted_q = tf.placeholder(tf.float32, [None, 1])
        self.loss = tflearn.mean_square(self.predicted_q, self.outputs)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.action_gradient = tf.gradients(self.outputs, self.action)

    def createNetwork(self):
        inputs = tflearn.input_data(shape=[None, self.state_size])
        action = tflearn.input_data(shape=[None, self.action_size])
        net = tflearn.fully_connected(inputs, 400, activation='relu')
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)
        net = tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        weights_init = tflearn.initializations.uniform(minval=-self.temperature, maxval=self.temperature)
        outputs = tflearn.fully_connected(net, 1, weights_init=weights_init)
        return (inputs, action, outputs)

    def train(self, inputs, action, predicted_q):
        return self.sess.run([self.outputs, self.optimize], feed_dict={self.inputs: inputs,
         self.action: action,
         self.predicted_q: predicted_q})

    def predict(self, inputs, action):
        return self.sess.run(self.outputs, feed_dict={self.inputs: inputs,
         self.action: action})

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_outputs, feed_dict={self.target_inputs: inputs,
         self.target_action: action})

    def action_gradients(self, inputs, action):
        return self.sess.run(self.action_gradient, feed_dict={self.inputs: inputs,
         self.action: action})

    def update_target_network(self):
        self.sess.run(self.update_target_params)
