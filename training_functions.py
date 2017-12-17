import numpy
import tensorflow as tf
from neural_network_model import NeuralNetworkModel
from logging_functions import LoggingFunctions
from data_processing_functions import DataFunctions


class NeuralNetworkTrainer:
    load_previous_training = False

    learning_rate = 0.001
    n_epoch = 20

    def __init__(self, dataset_source, model_name, variable_list, test_split):
        self.dataset_source = dataset_source
        self.model_name = model_name

        self.x = tf.placeholder('float')
        self.y = tf.placeholder(tf.float32, shape=[None, 5])

        self.feature_list, self.col_y = self.get_features(variable_list)
        self.features = tf.stack(self.feature_list)

        self.model = NeuralNetworkModel(self.x, len(self.feature_list))
        self.prediction = self.model.use_model()

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.logger = LoggingFunctions(self.model_name)
        self.logger.log_variables_used(variable_list)

        self.data_functions = DataFunctions(dataset_source, test_split)

    def get_features(self, variable_list):
        filename_queue = tf.train.string_input_producer([self.dataset_source])
        reader = tf.TextLineReader(skip_header_lines=1)
        key, value = reader.read(filename_queue)

        record_defaults = [[1], [""], [""], [""], [1.0],
                           [1.0], [1.0], [1.0], [1.0], [1.0],
                           [1.0], [1.0], [1.0], [1.0], [1.0],
                           [1.0], [1.0], [1.0], [1.0], [1.0],
                           [1.0], [1.0], [1.0], [1], [1],
                           [1]]

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
        elif "dah" in self.model_name:
            return feature_list, DAMAGED_HOUSES
        elif "dap" in self.model_name:
            return feature_list, DAMAGED_PROPERTIES

    def train_and_test_network(self):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            dataset_x, dataset_y = self.data_functions.load_data(self.features, self.col_y, sess)
            train_x, train_y, test_x, test_y = self.data_functions.split_train_test(dataset_x, dataset_y)

            self.logger.log_training_settings(self.model.n_nodes,
                                              self.model.n_hidden_layers,
                                              self.learning_rate,
                                              self.n_epoch,
                                              self.data_functions.split,
                                              len(dataset_x),
                                              len(train_x),
                                              len(test_x))

            self.train_neural_network(sess, train_x, train_y)
            self.test_neural_network(sess, test_x, test_y)

            coord.request_stop()
            coord.join(threads)

        tf.reset_default_graph()

    def train_neural_network(self, sess, train_x, train_y):
        saver = tf.train.Saver()

        for epoch in range(self.n_epoch):
            if self.load_previous_training and epoch == 0:
                saver.restore(sess, self.model_name)

            fetches = [self.optimizer, self.cost, self.prediction]
            feed_dict = {self.x: train_x, self.y: train_y}
            _, c, p = sess.run(fetches, feed_dict)

            saver.save(sess, self.model_name)

            epoch_loss = c / len(train_x)
            actual_class = train_y
            predicted_class = p

            self.logger.log_epoch_cost(epoch, epoch_loss)
            self.logger.log_actual_predicted_values(actual_class, predicted_class)

    def test_neural_network(self, sess, test_x, test_y):
        correct = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        accuracy_value = accuracy.eval({self.x: test_x, self.y: test_y})

        feed_dict = {self.x: test_x, self.y: test_y}
        p = sess.run(self.prediction, feed_dict)

        num_of_correct = 0
        for i in range(len(p)):
            if numpy.argmax(test_y[i], axis=None) == p[i].argmax(axis=None):
                num_of_correct += 1

        variable_names, values = self.get_weights(sess)

        self.logger.log_to_file("*****TEST*****")
        self.logger.log_actual_predicted_values(test_y, p)
        self.logger.log_to_file(str(num_of_correct) + " correct predictions out of " + str(len(p)))
        self.logger.log_accuracy(accuracy_value * 100)

        precisions = self.compute_precision_recall(self.convert_labels(p), self.convert_labels(test_y))
        for precision in precisions:
            self.logger.log_to_file(precision)

        self.logger.log_weights(variable_names, values)

    def get_weights(self, sess):
        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)

        return variables_names, values

    def compute_precision_recall(self, predicted_y, test_y):
        labels = []
        outputs = []

        total_precision = 0
        total_recall = 0
        total_harmony = 0

        for row in test_y:
            if row not in labels:
                labels.append(row)

        for label in labels:
            true_positive = 0
            predicted_positive = 0
            total_positive = 0

            for i in range(len(predicted_y)):
                if predicted_y[i] == label:
                    predicted_positive += 1
                    if predicted_y[i] == test_y[i]:
                        true_positive += 1

            for row in test_y:
                if row == label:
                    total_positive += 1

            precision = 0
            harmony = 0
            recall = 0

            if predicted_positive != 0:
                precision = true_positive / predicted_positive
            if total_positive != 0:
                recall = true_positive / total_positive
            if precision + recall != 0:
                harmony = 2 * ((precision * recall) / (precision + recall))

            total_precision += precision
            total_recall += recall
            total_harmony += harmony

            output = "\nclass: " + str(label) + \
                     "\n    precision: " + str(precision) + \
                     "\n    recall: " + str(recall) + \
                     "\n    harmonic mean: " + str(harmony)

            outputs.append(output)

        n_labels = len(labels)
        average_precision = total_precision / n_labels
        average_recall = total_recall / n_labels
        average_harmony = total_harmony / n_labels

        output = "\nAverage" + \
                 "\n    precision: " + str(average_precision) + \
                 "\n    recall: " + str(average_recall) + \
                 "\n    harmonic mean: " + str(average_harmony) + "\n"
        outputs.append(output)

        return outputs
