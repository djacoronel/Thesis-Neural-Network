import tensorflow as tf


CASUALTIES_MODEL = "models/casualties_test.ckpt"
DAMAGED_HOUSES_MODEL = "models/damagedhouses_test.ckpt"
DAMAGED_PROPERTIES_MODEL = "models/damagedproperties_test.ckpt"

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


def use_neural_network(x, model_name):
    tf.reset_default_graph()

    n_inputs = len(x[0])

    x_placeholder = tf.placeholder('float')

    from neural_network_model import NeuralNetworkModel
    model = NeuralNetworkModel(x, n_inputs)
    prediction = model.use_model()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_name)

        result = prediction.eval(feed_dict={x_placeholder: x})
        return result[0][0]


def predict_casualties():
    CASUALTIES_x = [[DURATION, INTENSITY, SIGNAL, DR, FLR]]
    result = use_neural_network(CASUALTIES_x, CASUALTIES_MODEL)
    if result < 0:
        result = 0

    print("\n*****PREDICT CASUALTIES*****")
    print("DURATION: " + str(DURATION))
    print("WIND: " + str(WIND))
    print("INTENSITY: " + str(INTENSITY))
    print("PR: " + str(DR))
    print("FLR: " + str(FLR))
    print("Actual casualties: " + str(CASUALTY))
    print("Predicted casualties: " + str(result))


def predict_damaged_houses():
    DAMAGED_HOUSES_x = [[DURATION, WIND, SIGNAL, DEN, FLR, HMB, HMD]]
    result = use_neural_network(DAMAGED_HOUSES_x, DAMAGED_HOUSES_MODEL)
    if result < 0:
        result = 0

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
    DAMAGED_PROPERTIES_x = [[INTENSITY, SIGNAL, DEN, DR, FLR, HMD]]
    result = use_neural_network(DAMAGED_PROPERTIES_x, DAMAGED_PROPERTIES_MODEL)
    if result < 0:
        result = 0

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
predict_damaged_properties()
