import tensorflow as tf

class TrainingSettings:
    n_total_data = 1893
    batch_size = 10
    n_batches = int(n_total_data / batch_size)

    load_previous_training = False

    learning_rate = 0.001
    n_epoch = 100

    display_step = 1

    def __init__(self, model_name, feature_list, col_y):
        self.model_name = model_name
        self.feature_list = feature_list
        self.col_y = col_y

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

        x = tf.placeholder('float')
        y = tf.placeholder('float')
        features = tf.stack(self.feature_list)
        n_inputs = len(self.feature_list)

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
                    train_x, train_y = self.next_batch(features, self.col_y, sess)
                    _, c, p = sess.run([optimizer, cost, prediction], feed_dict={x: train_x, y: train_y})
                    epoch_loss += c / self.n_batches

                saver.save(sess, self.model_name)

                actual_value = train_y
                estimated_value = p

                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=",
                          "{:.9f}".format(epoch_loss))
                    self.compute_mse_mape(actual_value, estimated_value)

            coord.request_stop()
            coord.join(threads)

            test_x, test_y = self.next_batch(features, self.col_y, sess)

            testing_cost = sess.run(cost, feed_dict={x: test_x, y: test_y})
            predicted_values = sess.run(prediction, feed_dict={x: test_x})

            print("***Test Batch Accuracy***")
            self.compute_mse_mape(test_y, predicted_values)
            print("Testing cost=", testing_cost)
            print("Absolute mean square loss difference:", abs(epoch_loss - testing_cost))