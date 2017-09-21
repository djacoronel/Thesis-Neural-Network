import tensorflow as tf
import math


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
        self.log_file_name = model_name[0:-5] + "_log.txt"

    def log_training_settings(self, n_nodes_per_layer, n_hidden_layers):
        output = "Number of hidden layers: " + str(n_hidden_layers) + \
            "\nNodes per layer: " + str(n_nodes_per_layer) + \
            "\nLearning rate: " + str(self.learning_rate) + \
            "\nNumber of epoch: " + str(self.n_epoch) + \
            "\nBatch size: " + str(self.batch_size)
        self.log_to_file(output)

    def get_features(self, model_name):
        # filename_queue = tf.train.string_input_producer(["insurance2.csv"])
        filename_queue = tf.train.string_input_producer(["dataset.csv"])
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)
        # record_defaults = [[1.0], [1.0]]
        # claims, payment = tf.decode_csv(value, record_defaults=record_defaults)

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

        # feature_list_1 = [DURATION, WIND, INTENSITY, SIGNAL, DR, FLR]
        feature_list_1 = [DURATION, INTENSITY, SIGNAL, DR, FLR]
        # feature_list_1 = [INTENSITY, FLR]
        col_y_1 = CASUALTIES
        # feature_list_2 = [DURATION, WIND, INTENSITY, SIGNAL, DEN, FLR, HMB, HMD]
        feature_list_2 = [DURATION, WIND, INTENSITY, SIGNAL, DEN, FLR, HMA, HMB, HMC, HMD]
        col_y_2 = DAMAGED_HOUSES
        # feature_list_3 = [WIND, INTENSITY, SIGNAL, DEN, DR, FLR, HS, HMB, HMD]
        feature_list_3 = [INTENSITY, SIGNAL, DEN, DR, FLR, HS, HMB, HMD, AI, HP]
        col_y_3 = DAMAGED_PROPERTIES

        if "casualties" in model_name:
            return feature_list_1, col_y_1
        elif "damagedhouses" in model_name:
            return feature_list_2, col_y_2
        elif "damagedproperties" in model_name:
            return feature_list_3, col_y_3
        # return [claims], payment

    def next_batch(self, features, col_y, sess):
        batch_x = []
        batch_y = []
        for iteration in range(self.batch_size):
            x, y = sess.run([features, col_y])
            batch_x.append(x)
            batch_y.append(y)
        return batch_x, batch_y

    def log_actual_estimated_values(self, actual_value, estimated_value):
        output = "[*]----------------------------" + "\n"
        for i in range(self.batch_size):
            output += "actual value: " + str(actual_value[i]) + \
                    " estimated value: " + str(estimated_value[i][0]) + "\n"
        output += "[*]----------------------------"

        self.log_to_file(output)
        print(output)

    def log_rmse_mape(self, mse, mape):
        output = "RMSE: " + str(math.sqrt(mse)) + "\n" \
                + "MAPE: " + str(mape) + "\n" \
                + "[*]============================"

        self.log_to_file(output)
        print(output)

    def log_test_cost_difference(self, testing_cost, cost_difference):
        print("Testing cost = ", testing_cost)
        print("Absolute mean square loss difference:", cost_difference)

    def compute_mse_mape(self, actual_value, estimated_value):
        sse = 0
        spe = 0
        n_spe_included = 0
        for i in range(len(actual_value)):
            error = (actual_value[i] - estimated_value[i][0])
            sqerror = error**2
            sse += sqerror

            if actual_value[i] != 0:
                spe += abs((actual_value[i] - estimated_value[i][0]) / actual_value[i])
                n_spe_included += 1

        mse = sse / len(actual_value)
        if n_spe_included != 0:
            mape = (spe / n_spe_included) * 100
        else:
            mape = 404

        return mse, mape

    def log_epoch_cost(self, epoch, epoch_loss):
        output = "Epoch: " + '%04d' % (epoch + 1) + " cost = " + "{:.9f}".format(epoch_loss)
        self.log_to_file(output)
        print(output)

    def log_to_file(self, line):
        with open(self.log_file_name, 'a') as f:
            f.write(line + '\n')

    def train_neural_network(self):
        feature_list, col_y = self.get_features(self.model_name)

        x = tf.placeholder('float')
        y = tf.placeholder('float')
        features = tf.stack(feature_list)
        n_inputs = len(feature_list)

        from neural_network_model import NeuralNetworkModel
        model = NeuralNetworkModel(x, n_inputs)
        prediction = model.use_model()

        self.log_training_settings(model.n_nodes, model.n_hidden_layers)

        cost = tf.reduce_mean(tf.square(prediction - y))
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
        # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

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

                # Accuracy is based on the results of last batch
                actual_value = train_y
                estimated_value = p

                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    self.log_epoch_cost(epoch, epoch_loss)
                    self.log_actual_estimated_values(actual_value, estimated_value)
                    mse, mape = self.compute_mse_mape(actual_value, estimated_value)
                    self.log_rmse_mape(mse, mape)

            coord.request_stop()
            coord.join(threads)

            test_x, test_y = self.next_batch(features, col_y, sess)

            testing_cost = sess.run(cost, feed_dict={x: test_x, y: test_y})
            predicted_values = sess.run(prediction, feed_dict={x: test_x})

            cost_difference = abs(epoch_loss - testing_cost)

            self.log_to_file("***Test Batch Accuracy***")
            print("***Test Batch Accuracy***")
            mse, mape = self.compute_mse_mape(test_y, predicted_values)
            self.log_actual_estimated_values(test_y, predicted_values)
            self.log_rmse_mape(mse, mape)
            self.log_test_cost_difference(testing_cost, cost_difference)

        tf.reset_default_graph()
