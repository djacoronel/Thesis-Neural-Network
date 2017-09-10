import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

filename_queue = tf.train.string_input_producer(["data.csv"])
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10])

n_nodes_hl1 = 5
n_nodes_hl2 = 5
n_nodes_hl3 = 5

n_total_data = 14
n_input = 10
n_classes = 1
batch_size = 3

x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_input, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])), }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def next_batch(batch_size, sess):
    batch_x = []
    batch_y = []
    for iteration in range(batch_size):
        example, label = sess.run([features, col11])
        batch_x.append(example)
        batch_y.append(label)

    return batch_x, batch_y

def train_neural_network(x):
    prediction = neural_network_model(x)

    cost = tf.reduce_mean(tf.square(prediction-y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(hm_epochs):
            epoch_loss = 0

            for _ in range(int(n_total_data / batch_size)):

                epoch_x, epoch_y = next_batch(batch_size,sess)

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        coord.request_stop()
        coord.join(threads)

        correct = tf.equal(tf.argmax(prediction, 1), tf.cast(y,tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        test_x, test_y = next_batch(2,sess)
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

train_neural_network(x)