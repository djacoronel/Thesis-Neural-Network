import csv
import random

import tensorflow as tf

datasource = "arranged/random/LUZ-CA.csv"

n_total_data = sum(1 for row in csv.reader(open(datasource)))
split = 0.95

train_size = int(n_total_data*split)
test_size = int(n_total_data*(1-split))

n_output = 1
learning_rate = 0.1
training_epochs = 50

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

    output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']

    return output

def next_batch(batch_size, features, coly, sess):
    batch_x = []
    batch_y = []
    for iteration in range(batch_size):
        x, y = sess.run([features, coly])
        batch_x.append(x)
        batch_y.append(y)

    #batch_x = normalize(batch_x)
    batch_x = standardize(batch_x)

    combined = list(zip(batch_x, batch_y))
    random.shuffle(combined)

    batch_x[:], batch_y[:] = zip(*combined)

    return batch_x, batch_y

def normalize(data):
    from sklearn.preprocessing import Normalizer
    scaler = Normalizer().fit(data)
    normalized_x = scaler.transform(data)

    return normalized_x

def standardize(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(data)
    standardized_x = scaler.transform(data)

    return standardized_x

def train_neural_network(x, y, save_name, features, coly, n_inputs, n_nodes):
    prediction = neural_network_model(x, n_inputs, n_nodes)
    cost = tf.reduce_mean(tf.square(prediction - y))
    saver = tf.train.Saver()

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        train_x, train_y = next_batch(train_size, features, coly, sess)

        for epoch in range(training_epochs):

            _, c, p = sess.run([optimizer, cost, prediction], feed_dict={x: train_x, y: train_y})

            for n in range(len(train_y)):
                output = "actual value: " + str(train_y[n]) + \
                          " estimated value: " + str(p[n][0])

                print(output)


            saver.save(sess, save_name)

        coord.request_stop()
        coord.join(threads)

        test_x, test_y = next_batch(test_size, features, coly, sess)
        predicted_values = sess.run(prediction, feed_dict={x: test_x})


        sum = 0
        spe = 0
        n_spe_included = 0
        for i in range(len(predicted_values)):
            output = "actual value: " + str(test_y[i]) + \
                     " estimated value: " + str(predicted_values[i][0])
            sum += ((test_y[0] - predicted_values[i][0])**2)


            if test_y[i] != 0:
                spe += abs((test_y[i] - predicted_values[i][0]) / test_y[i])
                n_spe_included += 1

            print(output)
        print("MSE: " + str(sum/len(predicted_values)))
        print("Accuracy: " + str(100 - ((spe/n_spe_included)*100)))


filename_queue = tf.train.string_input_producer([datasource])
reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)


record_defaults = [[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]]
var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13, var14 = tf.decode_csv(value, record_defaults=record_defaults)
features_1_list = [var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13]
features_1 = tf.stack(features_1_list)
x1 = tf.placeholder('float')
y = tf.placeholder('float')

train_neural_network(x1,y, "insufsdffasdfasrance.ckpt", features_1, var14, 13, 13)
