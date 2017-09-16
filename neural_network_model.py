import tensorflow as tf


class NeuralNetworkModel():

    def __init__(self, data, n_input):
        self.data = data
        self.n_input = n_input

    def use_model(self):
        n_nodes = self.n_input
        n_output = 1

        hidden_1_layer = {'weights': tf.Variable(tf.random_normal([self.n_input, n_nodes])),
                          'biases': tf.Variable(tf.random_normal([n_nodes]))}
        hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes, n_nodes])),
                          'biases': tf.Variable(tf.random_normal([n_nodes]))}
        hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes, n_nodes])),
                          'biases': tf.Variable(tf.random_normal([n_nodes]))}
        output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes, n_output])),
                        'biases': tf.Variable(tf.random_normal([n_output])), }

        l1 = tf.add(tf.matmul(self.data, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)
        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)
        l3 = tf.add(tf.matmul(l1, hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.sigmoid(l3)

        output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
        return output
