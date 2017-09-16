import tensorflow as tf

class TrainingSettings:

    n_total_data = 1893
    batch_size = 10
    n_batches = int(n_total_data / batch_size)

    load_previous_training = False

    learning_rate = 0.001
    n_epoch = 100

    display_step = 1

    def __init__(self, model_name):
        self.model_name = model_name

    def get_features(self, model_name):
        filename_queue = tf.train.string_input_producer(["dataset.csv"])
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)

        record_defaults = [[1], [""], [""], [""], [1.0],
                           [""], [1.0], [1.0], [1.0], [1.0],
                           [1.0], [1.0], [1.0], [1.0], [1.0],
                           [1.0], [1.0], [1.0], [1.0], [1.0],
                           [1.0], [1.0], [1.0], [1.0], [1.0],
                           [1.0]]

        ROW_NUMBER, NAME, REGION, PROVINCE, YEAR, \
        TYPE, DURATION, WIND, INTENSITY, SIGNAL, \
        POP, DEN, AI, PR, DR, \
        SR, FLR, HP, HS, HMA, \
        HMB, HMC, HMD, \
        CASUALTIES, DAMAGED_HOUSES, DAMAGED_PROPERTIES = tf.decode_csv(value, record_defaults=record_defaults)

        feature_list_1 = [DURATION, WIND, INTENSITY, SIGNAL, DR, FLR]
        col_y_1 = CASUALTIES
        feature_list_2 = [DURATION, WIND, INTENSITY, SIGNAL, DEN, FLR, HMB, HMD]
        col_y_2 = DAMAGED_HOUSES
        feature_list_3 = [WIND, INTENSITY, SIGNAL, DEN, DR, FLR, HS, HMB, HMD]
        col_y_3 = DAMAGED_PROPERTIES

        if "casualties" in model_name:
            return feature_list_1, col_y_1
        elif "damagedhouses" in model_name:
            return feature_list_2, col_y_2
        elif "damagedproperties" in model_name:
            return feature_list_3,col_y_3

    def next_batch(self, features, col_y, sess):
        batch_x = []
        batch_y = []
        for iteration in range(self.batch_size):
            x, y = sess.run([features, col_y])
            batch_x.append(x)
            batch_y.append(y)
        return batch_x, batch_y

    def compute_mse_mape(self, actual_value, estimated_value):
        sse = 0
        spe = 0
        print("[*]----------------------------")
        for i in range(self.batch_size):
            print("actual value:", actual_value[i],
                  "estimated value:", estimated_value[i][0])
            sse += (actual_value[i] - estimated_value[i][0]) ** 2
            if actual_value[i] != 0:
                spe += abs((actual_value[i] - estimated_value[i][0]) / actual_value[i])

        mse = sse / self.batch_size
        mape = (100 / self.batch_size) * spe

        print("MSE:" + str(mse))
        print("MAPE:" + str(mape))
        print("[*]============================")

    def train_neural_network(self):
        feature_list, col_y = self.get_features(self.model_name)

        x = tf.placeholder('float')
        y = tf.placeholder('float')
        features = tf.stack(feature_list)
        n_inputs = len(feature_list)

        from neural_network_model import NeuralNetworkModel
        model = NeuralNetworkModel(x, n_inputs)
        prediction = model.use_model()

        cost = tf.reduce_mean(tf.square(prediction - y))
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for epoch in range(self.n_epoch):
                if self.load_previous_training and epoch == 0:
                    saver.restore(sess, self.model_name)

                epoch_loss = 0

                for i in range(self.n_batches):
                    train_x, train_y = self.next_batch(features, col_y, sess)
                    _, c, p = sess.run([optimizer, cost, prediction], feed_dict={x: train_x, y: train_y})
                    epoch_loss += c / self.n_batches

                saver.save(sess, self.model_name)

                #Accuracy is based on the results of last batch
                actual_value = train_y
                estimated_value = p

                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=",
                          "{:.9f}".format(epoch_loss))
                    self.compute_mse_mape(actual_value, estimated_value)

            coord.request_stop()
            coord.join(threads)

            test_x, test_y = self.next_batch(features, col_y, sess)

            testing_cost = sess.run(cost, feed_dict={x: test_x, y: test_y})
            predicted_values = sess.run(prediction, feed_dict={x: test_x})

            print("***Test Batch Accuracy***")
            self.compute_mse_mape(test_y, predicted_values)
            print("Testing cost=", testing_cost)
            print("Absolute mean square loss difference:", abs(epoch_loss - testing_cost))

        tf.reset_default_graph()
