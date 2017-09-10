import tensorflow as tf


n_total_data = 1893
n_output = 1

batch_size = 10
test_batch_size = 10
n_batches = int(n_total_data / batch_size)

learning_rate = 0.001
training_epochs = 1800

display_step = 100


def neural_network_model(data, n_inputs, n_nodes):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_inputs, n_nodes])),
                      'biases': tf.Variable(tf.random_normal([n_nodes]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes, n_nodes])),
                      'biases': tf.Variable(tf.random_normal([n_nodes]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes, n_nodes])),
                      'biases': tf.Variable(tf.random_normal([n_nodes]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes, n_output])),
                    'biases': tf.Variable(tf.random_normal([n_output])), }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def next_batch(batch_size, features, coly, sess):
    batch_x = []
    batch_y = []
    for iteration in range(batch_size):
        x, y = sess.run([features, coly])
        batch_x.append(x)
        batch_y.append(y)

    return batch_x, batch_y

def train_neural_network(x, y, save_name, features, coly, n_inputs, n_nodes):
    prediction = neural_network_model(x, n_inputs, n_nodes)
    cost = tf.reduce_mean(tf.square(prediction - y))
    saver = tf.train.Saver()

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(training_epochs):

            epoch_loss = 0

            for _ in range(n_batches):
                train_x, train_y = next_batch(batch_size, features, coly, sess)
                _, c, p = sess.run([optimizer, cost, prediction], feed_dict={x: train_x, y: train_y})
                epoch_loss += c/n_batches

            saver.save(sess, save_name)

            actual_value = train_y
            estimated_value = p

            # Display logs per epoch step
            if epoch % display_step == 0:
                print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(epoch_loss))
                print ("[*]----------------------------")
                sse = 0
                for i in range(batch_size):
                    print ("actual value:", actual_value[i], \
                        "estimated value:", estimated_value[i][0])
                    sse += (actual_value[i]-estimated_value[i][0])**2
                mse = sse/batch_size
                print("MSE:" + str(mse))
                print ("[*]============================")

        coord.request_stop()
        coord.join(threads)

        test_x, test_y = next_batch(test_batch_size, features, coly, sess)

        testing_cost = sess.run(cost, feed_dict={x: test_x, y: test_y})
        predicted_values = sess.run(prediction, feed_dict={x: test_x})

        print("Testing cost=", testing_cost)
        print("Absolute mean square loss difference:", abs(epoch_loss - testing_cost))

filename_queue = tf.train.string_input_producer(["dataset.csv"])
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[1], [""], [""], [""], [1.0],
                   [""], [1.0], [1.0], [1.0], [1.0],
                   [1.0], [1.0], [1.0], [1.0], [1.0],
                   [1.0], [1.0], [1.0], [1.0], [1.0],
                   [1.0], [1.0], [1.0], [1.0], [1.0],
                   [1.0]]

ROW_NUMBER,NAME,REGION,PROVINCE,YEAR, \
TYPE,DURATION,WIND,INTENSITY,SIGNAL, \
POP,DEN,AI,PR,DR, \
SR,FLR,HP,HS, HMA, \
HMB, HMC, HMD,\
CASUALTIES,DAMAGED_HOUSES,DAMAGED_PROPERTIES = tf.decode_csv(value, record_defaults=record_defaults)

features_1_list = [DURATION, WIND, INTENSITY, SIGNAL, DR, FLR]
features_1 = tf.stack(features_1_list)
col_y_1 = CASUALTIES

features_2_list = [DURATION, WIND, INTENSITY, SIGNAL, DEN, FLR, HMB, HMD]
features_2 = tf.stack(features_2_list)
col_y_2 = DAMAGED_HOUSES

features_3_list = [WIND, INTENSITY, SIGNAL, DEN, DR, FLR, HS, HMB, HMD]
features_3 = tf.stack(features_3_list)
col_y_3 = DAMAGED_PROPERTIES

n_input_1 = len(features_1_list)
n_input_2 = len(features_2_list)
n_input_3 = len(features_3_list)

model_name_1 = "casualties1.ckpt"
model_name_2 = "damagedhouses.ckpt"
model_name_3 = "damagedproperties.ckpt"

n_nodes_1 = n_input_1 * 3
n_nodes_2 = n_input_2 * 3
n_nodes_3 = n_input_3 * 3

x1 = tf.placeholder('float')
x2 = tf.placeholder('float', [None, n_input_2])
x3 = tf.placeholder('float', [None, n_input_3])

y = tf.placeholder('float')

#CASUALTIES
#train_neural_network(x1,y, model_name_1, features_1, col_y_1, n_input_1, n_nodes_1)
#DAMAGED HOUSES
train_neural_network(x2,y, model_name_2, features_2, col_y_2, n_input_2, n_nodes_2)
#DAMAGED PROPERTIES
train_neural_network(x3,y, model_name_3, features_3, col_y_3, n_input_3, n_nodes_3)