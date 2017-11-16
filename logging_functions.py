import numpy


class LoggingFunctions:
    def __init__(self, model_name):
        self.log_file_name = model_name[0:-5] + "_log.txt"
        with open(self.log_file_name, 'w') as f:
            f.write(model_name[0:-5] + '\n')

    def log_to_file(self, line):
        with open(self.log_file_name, 'a') as f:
            f.write(line + '\n')
        print(line)

    def log_epoch_cost(self, epoch, epoch_loss):
        output = "Epoch: " + '%04d' % (epoch + 1) + " cost = " + "{:.9f}".format(epoch_loss)
        self.log_to_file(output)

    def log_accuracy(self, accuracy):
        output = "Model Accuracy: " + "{:.2f}".format(accuracy) + "%"
        self.log_to_file(output)

    def log_actual_predicted_values(self, actual_value, estimated_value):
        output = "[*]----------------------------" + "\n"
        for i in range(len(actual_value)):
            output += "actual: " + str(numpy.array(actual_value[i]).argmax(axis=None) + 1) + \
                      " predicted: " + str(estimated_value[i].argmax(axis=None) + 1) + \
                      " " + str(actual_value[i]) + str(estimated_value[i]) + "\n"
        output += "[*]----------------------------"

        self.log_to_file(output)

    def log_training_settings(self, n_nodes_per_layer, n_hidden_layers, learning_rate,
                              n_epoch, split, n_total_data, train_rows, test_rows):
        output = "Number of hidden layers: " + str(n_hidden_layers) + \
                 "\nNodes per layer: " + str(n_nodes_per_layer) + \
                 "\nLearning rate: " + str(learning_rate) + \
                 "\nNumber of epoch: " + str(n_epoch) + \
                 "\nTotal data: " + str(n_total_data) + \
                 "\nSplit: " + str(split) + \
                 "\nTraining rows: " + str(train_rows) + " Test rows: " + str(test_rows)
        self.log_to_file(output)

    def log_variables_used(self, variable_list):
        output = "Variables used: " + " ".join(variable for variable in variable_list)
        self.log_to_file(output)

    def log_weights(self, variable_names, values):
        output = "***** NEURAL NETWORK WEIGHTS *****"
        self.log_to_file(output)
        for k, v in zip(variable_names, values):
            self.log_to_file(str(k))
            self.log_to_file(str(v) + "\n")
