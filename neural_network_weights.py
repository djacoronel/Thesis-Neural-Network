import tensorflow as tf

def get_weights(n_input, model_location):
    x_placeholder = tf.placeholder('float')
    
    n_hidden_layers = 3
    n_output = 5
    hidden_layers = []

    input_layer = {'weights': tf.Variable(tf.random_normal([n_input, n_input])),
                   'biases': tf.Variable(tf.random_normal([n_input]))}

    for n in range(n_hidden_layers):
        layer = {'weights': tf.Variable(tf.random_normal([n_input, n_input])),
                 'biases': tf.Variable(tf.random_normal([n_input]))}
        hidden_layers.append(layer)

    output_layer = {'weights': tf.Variable(tf.random_normal([n_input, n_output])),
                    'biases': tf.Variable(tf.random_normal([n_output]))}

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_location)

        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)

        return variables_names, values


def print_weights(variable_names, values):
    for k, v in zip(variable_names, values):
        print(str(k))

        if len(v.shape) == 2:
            for row in v:
                print(','.join(str(x) for x in list(row)))
        else:
            print(','.join(str(x) for x in list(v)))

        print("")


model_location = "Iterations/models/CAS/Q-CAS-ARIMAspl0.6set1e5r1.cktp"
variable_names, values = get_weights(4,model_location)
print_weights(variable_names, values)

