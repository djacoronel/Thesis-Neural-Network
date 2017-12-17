import csv
import random

import numpy


class DataFunctions:
    cas_thresholds = [[0, 9], [10, 19], [20, 31], [32, 129], [130, None]]
    dah_thresholds = [[0, 57], [58, 213], [214, 1137], [1138, 6264], [6265, None]]
    dap_thresholds = [[0, 3549500], [3549501, 19040810], [19040811, 55701274], [55701275, 180522000], [180522001, None]]

    def __init__(self, dataset_source, split):
        self.dataset_source = dataset_source
        self.n_total_data = sum(1 for row in csv.reader(open(dataset_source)))
        self.split = split

    def load_data(self, features, col_y, sess):
        x = []
        y = []
        for iteration in range(self.n_total_data):
            x_row, y_row = sess.run([features, col_y])
            x.append(x_row)
            y.append(self.convert_to_one_hot(self.convert_to_categorical(y_row)))

        x, y = self.shuffle_rows(x, y)
        x = self.normalize(x)
        x = self.standardize(x)

        return x, y

    def split_train_test(self, x, y):
        n_train = int(self.n_total_data * self.split)

        train_x = []
        train_y = []
        test_x = []
        test_y = []

        for i in range(self.n_total_data):
            if i < n_train:
                train_x.append(x[i])
                train_y.append(y[i])
            else:
                test_x.append(x[i])
                test_y.append(y[i])

        return train_x, train_y, test_x, test_y

    def convert_to_categorical(self, y):
        thresholds = []

        if "CAS" in self.dataset_source:
            thresholds = self.cas_thresholds
        elif "DAH" in self.dataset_source:
            thresholds = self.dah_thresholds
        elif "DAP" in self.dataset_source:
            thresholds = self.dap_thresholds

        if thresholds[0][0] <= y <= thresholds[0][1]:
            return 1
        elif thresholds[1][0] <= y <= thresholds[1][1]:
            return 2
        elif thresholds[2][0] <= y <= thresholds[2][1]:
            return 3
        elif thresholds[3][0] <= y <= thresholds[3][1]:
            return 4
        elif thresholds[4][0] <= y:
            return 5

    def convert_to_one_hot(self, y):
        one_hot_y = [0 for n in range(5)]
        one_hot_y[y - 1] = 1
        return one_hot_y

    def convert_to_labels(self, y):
        converted_y = []
        for row in y:
            converted_y.append(numpy.argmax(row, axis=None) + 1)

        return converted_y

    def shuffle_rows(self, batch_x, batch_y):
        combined = list(zip(batch_x, batch_y))
        random.shuffle(combined)

        shuffled_x, shuffled_y = [], []
        shuffled_x[:], shuffled_y[:] = zip(*combined)
        return shuffled_x, shuffled_y

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



