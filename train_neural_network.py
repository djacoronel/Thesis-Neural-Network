import tensorflow as tf

n_total_data = 1893
batch_size = 10
n_batches = int(n_total_data / batch_size)

load_previous_training = False
learning_rate = 0.001
training_epochs = 100

display_step = 1

def neural_network_model(data, n_inputs):
    n_nodes = n_inputs * 3
    n_output = 1

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
    l3 = tf.nn.sigmoid(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    return output

def next_batch(features, col_y, sess):
    batch_x = []
    batch_y = []
    for iteration in range(batch_size):
        x, y = sess.run([features, col_y])
        batch_x.append(x)
        batch_y.append(y)
    return batch_x, batch_y

def compute_mse_mape(actual_value, estimated_value):
    sse = 0
    spe = 0
    print("[*]----------------------------")
    for i in range(batch_size):
        print("actual value:", actual_value[i],
              "estimated value:", estimated_value[i][0])
        sse += (actual_value[i] - estimated_value[i][0]) ** 2
        if (actual_value[i] != 0):
            spe += abs((actual_value[i] - estimated_value[i][0]) / actual_value[i])

    mse = sse / batch_size
    mape = (100 / batch_size) * spe

    print("MSE:" + str(mse))
    print("MAPE:" + str(mape))
    print("[*]============================")

def train_neural_network(model_name, feature_list, col_y):

    x = tf.placeholder('float')
    y = tf.placeholder('float')
    features = tf.stack(feature_list)
    n_inputs = len(feature_list)

    prediction = neural_network_model(x, n_inputs)
    cost = tf.reduce_mean(tf.square(prediction - y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(training_epochs):
            if (load_previous_training and epoch==0):
                saver.restore(sess, model_name)

            epoch_loss = 0

            for i in range(n_batches):
                train_x, train_y = next_batch(features, col_y, sess)
                _, c, p = sess.run([optimizer, cost, prediction], feed_dict={x: train_x, y: train_y})
                epoch_loss += c/n_batches

            saver.save(sess, model_name)

            actual_value = train_y
            estimated_value = p

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=",
                      "{:.9f}".format(epoch_loss))
                compute_mse_mape(actual_value, estimated_value)

        coord.request_stop()
        coord.join(threads)

        test_x, test_y = next_batch(features, col_y, sess)

        testing_cost = sess.run(cost, feed_dict={x: test_x, y: test_y})
        predicted_values = sess.run(prediction, feed_dict={x: test_x})

        print("***Test Batch Accuracy***")
        compute_mse_mape(test_y, predicted_values)
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

feature_list_1 = [DURATION, WIND, INTENSITY, SIGNAL, DR, FLR]
col_y_1 = CASUALTIES

feature_list_2 = [DURATION, WIND, INTENSITY, SIGNAL, DEN, FLR, HMB, HMD]
col_y_2 = DAMAGED_HOUSES

feature_list_3 = [WIND, INTENSITY, SIGNAL, DEN, DR, FLR, HS, HMB, HMD]
col_y_3 = DAMAGED_PROPERTIES

model_name_1 = "models/casualties_fixed_mape.ckpt"
model_name_2 = "models/damagedhouses_e_500.ckpt"
model_name_3 = "models/damagedproperties_e_500.ckpt"

#CASUALTIES
train_neural_network(model_name_1, feature_list_1, col_y_1)
#DAMAGED HOUSES
#train_neural_network(model_name_2, feature_list_2, col_y_2)
#DAMAGED PROPERTIES
#train_neural_network(model_name_3, feature_list_3, col_y_3)