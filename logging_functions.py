import math

class LoggingFunctions:

    def __init__(self, model_name):
        self.log_file_name = model_name[0:-5] + "_log.txt"
        with open(self.log_file_name, 'w') as f:
            f.write(model_name[0:-5] + '\n')

    def log_to_file(self, line):
        with open(self.log_file_name, 'a') as f:
            f.write(line + '\n')

    def log_epoch_cost(self, epoch, epoch_loss):
        output = "Epoch: " + '%04d' % (epoch + 1) + " cost = " + "{:.9f}".format(epoch_loss)
        self.log_to_file(output)
        print(output)

    def log_test_cost_difference(self, testing_cost, cost_difference):
        print("Testing cost = ", testing_cost)
        print("Absolute mean square loss difference:", cost_difference)

    def log_rmse_mape(self, mse, mape):
        output = "RMSE: " + str(math.sqrt(mse)) + "\n" \
                + "MAPE: " + str(mape) + "\n" \
                + "[*]============================"

        self.log_to_file(output)
        print(output)

    def log_accuracy_rmse(self, accuracy, mse):
        output = "Model Accuracy: " + "{:.2f}".format(accuracy) + "%\n" \
                + "RMSE: " + str(math.sqrt(mse))

        self.log_to_file(output)
        print(output)

    def log_actual_estimated_values(self, actual_value, estimated_value):
        output = "[*]----------------------------" + "\n"
        for i in range(len(actual_value)):
            output += "actual value: " + str(actual_value[i]) + \
                    " estimated value: " + str(estimated_value[i][0]) + "\n"
        output += "[*]----------------------------"

        self.log_to_file(output)
        print(output)

    def log_training_settings(self, n_nodes_per_layer, n_hidden_layers, learning_rate, n_epoch, batch_size):
        output = "Number of hidden layers: " + str(n_hidden_layers) + \
            "\nNodes per layer: " + str(n_nodes_per_layer) + \
            "\nLearning rate: " + str(learning_rate) + \
            "\nNumber of epoch: " + str(n_epoch) + \
            "\nBatch size: " + str(batch_size)
        self.log_to_file(output)
        print(output)
