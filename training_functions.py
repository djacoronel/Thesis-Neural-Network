import csv
import random

import tensorflow as tf
from neural_network_model import NeuralNetworkModel
from logging_functions import LoggingFunctions


class TrainingSettings:
    load_previous_training = False

    learning_rate = 0.01
    n_epoch = 400

    def __init__(self, dataset_source, model_name, variable_list, test_split):
        self.dataset_source = dataset_source
        self.n_total_data = sum(1 for row in csv.reader(open(dataset_source)))

        self.n_train = int(self.n_total_data * test_split)
        self.n_test = int(self.n_total_data * (1 - test_split))
        self.test_split = test_split

        self.model_name = model_name
        self.x = tf.placeholder('float')
        self.y = tf.placeholder('float')
        self.feature_list, self.col_y = self.get_features(variable_list)
        self.features = tf.stack(self.feature_list)
        self.n_inputs = len(self.feature_list)
        self.model = NeuralNetworkModel(self.x, self.n_inputs)
        self.prediction = self.model.use_model()
        self.cost = tf.reduce_mean(tf.square(self.prediction - self.y))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.logger = LoggingFunctions(self.model_name)
        self.logger.log_variables_used(variable_list)

    def get_features(self, variable_list):
        filename_queue = tf.train.string_input_producer([self.dataset_source])
        reader = tf.TextLineReader(skip_header_lines=1)
        key, value = reader.read(filename_queue)

        record_defaults = [[1], [""], [""], [""], [1.0],
                           [1.0], [1.0], [1.0], [1.0], [1.0],
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

        feature_dict = {
            "ROW_NUMBER": ROW_NUMBER, "NAME": NAME, "REGION": REGION, "PROVINCE": PROVINCE, "YEAR": YEAR,
            "TYPE": TYPE, "DURATION": DURATION, "WIND": WIND, "INTENSITY": INTENSITY, "SIGNAL": SIGNAL,
            "POP": POP, "DEN": DEN, "AI": AI, "PR": PR, "DR": DR,
            "SR": SR, "FLR": FLR, "HP": HP, "HS": HS, "HMA": HMA,
            "HMB": HMB, "HMC": HMC, "HMD": HMD,
            "CASUALTIES": CASUALTIES, "DAMAGED_HOUSES": DAMAGED_HOUSES, "DAMAGED_PROPERTIES": DAMAGED_PROPERTIES
        }

        feature_list = []
        for variable in variable_list:
            feature_list.append(feature_dict[variable])

        if "cas" in self.model_name:
            return feature_list, CASUALTIES
        elif "dh" in self.model_name:
            return feature_list, DAMAGED_HOUSES
        elif "dp" in self.model_name:
            return feature_list, DAMAGED_PROPERTIES

    def get_rows(self, features, col_y, sess, batch_size):
        batch_x = []
        batch_y = []
        for iteration in range(batch_size):
            x, y = sess.run([features, col_y])
            batch_x.append(x)
            batch_y.append(y)

        batch_x, batch_y = self.shuffle_rows(batch_x, batch_y)
        batch_x = self.normalize(batch_x)
        #batch_x = self.standardize(batch_x)

        return batch_x, batch_y

    def shuffle_rows(self, batch_x, batch_y):
        combined = list(zip(batch_x, batch_y))
        random.shuffle(combined)

        shuffled_x, shuffled_y = [],[]
        shuffled_x[:], shuffled_y[:] = zip(*combined)
        return shuffled_x,shuffled_y


    def normalize(self, data):
        from sklearn.preprocessing import Normalizer
        scaler = Normalizer().fit(data)
        normalized_x = scaler.transform(data)

        return normalized_x

    def standardize(self, data):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(data)
        standardized_x = scaler.transform(data)

        return standardized_x

    def compute_mse_mape(self, actual_value, estimated_value):
        sse = 0
        spe = 0
        n_spe_included = 0
        for i in range(len(actual_value)):
            error = (actual_value[i] - estimated_value[i][0])
            sqerror = error ** 2
            sse += sqerror

            if actual_value[i] != 0:
                spe += abs((actual_value[i] - estimated_value[i][0]) / actual_value[i])
                n_spe_included += 1

        mse = sse / len(actual_value)
        if n_spe_included != 0:
            mape = (spe / n_spe_included) * 100
        else:
            mape = 0

        return mse, mape

    def train_and_test_network(self):
        self.logger.log_training_settings(self.model.n_nodes,
                                          self.model.n_hidden_layers,
                                          self.learning_rate,
                                          self.n_epoch,
                                          self.n_total_data,
                                          self.test_split,
                                          self.n_train,
                                          self.n_test)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            self.train_neural_network(sess)
            self.test_neural_network(sess)

            coord.request_stop()
            coord.join(threads)

    def train_neural_network(self, sess):
        saver = tf.train.Saver()
        train_x, train_y = self.get_rows(self.features, self.col_y, sess, self.n_train)

        for epoch in range(self.n_epoch):
            if self.load_previous_training and epoch == 0:
                saver.restore(sess, self.model_name)

            train_x, train_y  = self.shuffle_rows(train_x, train_y)

            fetches = [self.optimizer, self.cost, self.prediction]
            feed_dict = {self.x: train_x, self.y: train_y}
            _, c, p = sess.run(fetches, feed_dict)

            saver.save(sess, self.model_name)

            epoch_loss = c / self.n_train
            actual_value = train_y
            estimated_value = p
            mse, mape = self.compute_mse_mape(actual_value, estimated_value)

            if mape < 100:
                accuracy = 100 - mape
            else:
                accuracy = 0

            self.logger.log_epoch_cost(epoch, epoch_loss)
            self.logger.log_actual_estimated_values(actual_value, estimated_value)
            self.logger.log_accuracy_rmse(accuracy, mse)

    def test_neural_network(self, sess):
        test_x, test_y = self.get_rows(self.features, self.col_y, sess, self.n_test)
        predicted_values = sess.run(self.prediction, feed_dict={self.x: test_x})

        mse, mape = self.compute_mse_mape(test_y, predicted_values)

        if mape < 100:
            accuracy = 100 - mape
        else:
            accuracy = 0

        self.logger.log_to_file("*****TEST*****")
        self.logger.log_actual_estimated_values(test_y, predicted_values)
        self.logger.log_accuracy_rmse(accuracy, mse)
