import tensorflow as tf


class NeuralNetworkModel():
    n_hidden_layers = 3
    n_nodes = 1

    def __init__(self, data, n_input):
        self.data = data
        self.n_input = n_input

    def use_model(self):
        self.n_nodes = self.n_input
        n_output = 1
        hidden_layers = []

        input_layer = {'weights': tf.Variable(tf.random_normal([self.n_input, self.n_nodes])),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes]))}
        for n in range(self.n_hidden_layers):
            layer = {'weights': tf.Variable(tf.random_normal([self.n_nodes, self.n_nodes])),
                          'biases': tf.Variable(tf.random_normal([self.n_nodes]))}
            hidden_layers.append(layer)

        output_layer = {'weights': tf.Variable(tf.random_normal([self.n_nodes, n_output])),
                        'biases': tf.Variable(tf.random_normal([n_output])), }

        l1 = tf.add(tf.matmul(self.data, input_layer['weights']), input_layer['biases'])
        l1 = tf.nn.relu(l1)

        l0 = l1
        for n in range(self.n_hidden_layers):
            l0 = tf.add(tf.matmul(l0, hidden_layers[n]['weights']), hidden_layers[n]['biases'])
            l0 = tf.nn.relu(l0)

        output = tf.matmul(l0, output_layer['weights']) + output_layer['biases']
        return output
