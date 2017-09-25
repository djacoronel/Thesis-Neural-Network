import tensorflow as tf
from neural_network_model import NeuralNetworkModel
from logging_functions import LoggingFunctions


class TrainingSettings:

    n_total_data = 1893
    batch_size = 10
    n_batches = 170

    load_previous_training = False

    learning_rate = 0.001
    n_epoch = 100

    display_step = 1

    def __init__(self, model_name):
        self.model_name = model_name
        self.x = tf.placeholder('float')
        self.y = tf.placeholder('float')
        self.feature_list, self.col_y = self.get_features()
        self.features = tf.stack(self.feature_list)
        self.n_inputs = len(self.feature_list)
        self.model = NeuralNetworkModel(self.x, self.n_inputs)
        self.prediction = self.model.use_model()
        self.cost = tf.reduce_mean(tf.square(self.prediction - self.y))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        self.logger = LoggingFunctions(self.model_name)

    def get_features(self):
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

        feature_list_1 = [DURATION, INTENSITY, SIGNAL, DR, FLR]
        col_y_1 = CASUALTIES

        feature_list_2 = [DURATION, WIND, SIGNAL, DEN, FLR, HMB, HMD]
        col_y_2 = DAMAGED_HOUSES

        feature_list_3 = [INTENSITY, SIGNAL, DEN, DR, FLR, HMD]
        col_y_3 = DAMAGED_PROPERTIES

        if "casualties" in self.model_name:
            return feature_list_1, col_y_1
        elif "damagedhouses" in self.model_name:
            return feature_list_2, col_y_2
        elif "damagedproperties" in self.model_name:
            return feature_list_3, col_y_3

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

    def train_and_test_network(self):
        self.logger.log_training_settings(self.model.n_nodes,
                                          self.model.n_hidden_layers,
                                          self.learning_rate,
                                          self.n_epoch, self.batch_size)

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

        for epoch in range(self.n_epoch):
            if self.load_previous_training and epoch == 0:
                saver.restore(sess, self.model_name)

            epoch_loss = 0

            for i in range(self.n_batches):
                train_x, train_y = self.next_batch(self.features, self.col_y, sess)
                fetches = [self.optimizer, self.cost, self.prediction]
                feed_dict = {self.x: train_x, self.y: train_y}
                _, c, p = sess.run(fetches, feed_dict)
                epoch_loss += c / self.n_batches

            saver.save(sess, self.model_name)

            actual_value = train_y
            estimated_value = p

            if epoch % self.display_step == 0:
                self.logger.log_epoch_cost(epoch, epoch_loss)
                self.logger.log_actual_estimated_values(actual_value, estimated_value)
                mse, mape = self.compute_mse_mape(actual_value, estimated_value)
                self.logger.log_rmse_mape(mse, mape)

    def test_neural_network(self, sess):
        accuracies = []
        smse = []

        for n in range(5):
            test_x, test_y = self.next_batch(self.features, self.col_y, sess)
            predicted_values = sess.run(self.prediction, feed_dict={self.x: test_x})

            mse, mape = self.compute_mse_mape(test_y, predicted_values)
            smse.append(mse)
            if mape < 100:
                accuracies.append(100 - mape)

        mse = sum(smse) / len(smse)
        if len(accuracies) != 0:
            mean_accuracy = sum(accuracies) / len(accuracies)
        else:
            mean_accuracy = 0

        self.logger.log_accuracy_rmse(mean_accuracy, mse)
