import tensorflow as tf

n_output = 1

def neural_network_model(data, n_inputs, n_nodes):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_inputs, n_nodes])),
                      'biases': tf.Variable(tf.random_normal([n_nodes]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes, n_nodes])),
                      'biases': tf.Variable(tf.random_normal([n_nodes]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes, n_nodes])),
                      'biases': tf.Variable(tf.random_normal([n_nodes]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes, n_output])),
                    'biases': tf.Variable(tf.random_normal([n_output])), }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def use_neural_network(x, model_name):
    tf.reset_default_graph()

    n_inputs = len(x[0])
    n_nodes = n_inputs * 3

    x_placeholder = tf.placeholder('float')
    x_input = x

    prediction = neural_network_model(x_placeholder, n_inputs, n_nodes)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_name)

        result = prediction.eval(feed_dict={x_placeholder: x_input})
        return result[0][0]



CASUALTIES_MODEL = "casualties.ckpt"
DAMAGED_HOUSES_MODEL = "damagedhouses.ckpt"
DAMAGED_PROPERTIES_MODEL = "damagedproperties.ckpt"

#input for casualties
DURATION = 4.0
WIND = 105.0
INTENSITY = 1004.9
SIGNAL = 3.0
DEN = 258

DR = 76.0
FLR = 75.0

HS = 805
HMB = 135829.0
HMD = 12955.0

CASUALTY = 48
DAMAGED_HOUSES = 1283
DAMAGED_PROPERTIES = 8392000


def predict_casualties():
    CASUALTIES_x = [[DURATION, WIND, INTENSITY, SIGNAL, DR, FLR]]
    result = use_neural_network(CASUALTIES_x, CASUALTIES_MODEL)

    print("\n*****PREDICT CASUALTIES*****")
    print("DURATION: " + str(DURATION))
    print("WIND: " + str(WIND))
    print("INTENSITY: " + str(INTENSITY))
    print("PR: " + str(DR))
    print("FLR: " + str(FLR))
    print("Actual casualties: " + str(CASUALTY))
    print("Predicted casualties: " + str(result))

def predict_damaged_houses():
    DAMAGED_HOUSES_x = [[DURATION, WIND, INTENSITY, SIGNAL, DEN, FLR, HMB, HMD]]
    result = use_neural_network(DAMAGED_HOUSES_x, DAMAGED_HOUSES_MODEL)

    print("\n*****PREDICT DAMAGED HOUSES*****")
    print("DURATION: " + str(DURATION))
    print("WIND: " + str(WIND))
    print("INTENSITY: " + str(INTENSITY))
    print("SIGNAL: " + str(SIGNAL))
    print("DEN: " + str(DEN))
    print("FLR: " + str(FLR))
    print("HMB: " + str(HMB))
    print("HMD: " + str(HMD))
    print("Actual damaged houses: " + str(DAMAGED_HOUSES))
    print("Predicted damaged houses: " + str(result))

def predict_damaged_properties():
    DAMAGED_PROPERTIES_x = [[WIND, INTENSITY, SIGNAL, DEN, DR, FLR, HS, HMB, HMD]]
    result = use_neural_network(DAMAGED_PROPERTIES_x, DAMAGED_PROPERTIES_MODEL)

    print("\n*****PREDICT DAMAGED PROPERTIES*****")
    print("WIND: " + str(WIND))
    print("INTENSITY: " + str(INTENSITY))
    print("SIGNAL: " + str(SIGNAL))
    print("DEN: " + str(DEN))
    print("DR: " + str(DR))
    print("FLR: " + str(FLR))
    print("HMB: " + str(HMB))
    print("HMD: " + str(HMD))
    print("Actual damaged properties: " + str(DAMAGED_PROPERTIES))
    print("Predicted damaged properties: " + str(result))


predict_casualties()
predict_damaged_houses()
#predict_damaged_properties()