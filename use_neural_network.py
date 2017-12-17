import csv

import tensorflow as tf
import numpy as np
import time

from Iterations.neural_network_model import NeuralNetworkModel

CASUALTIES_MODEL = "DWT-ARIMA-ANN/CAS/D_cas.ckpt"
DAMAGED_HOUSES_MODEL = "DWT-ARIMA-ANN/DAH/D_dah.ckpt"
DAMAGED_PROPERTIES_MODEL = "DWT-ARIMA-ANN/DAP/D_dap.ckpt"



def get_predicted_levels(result):
    result_levels = []
    for i in result:
        result_levels.append(i.argmax(axis=None) + 1)
    return result_levels


def use_neural_network(x, model_name):
    tf.reset_default_graph()

    n_inputs = len(x[0])

    x_placeholder = tf.placeholder('float')

    x = np.array(x).astype(np.float32)

    model = NeuralNetworkModel(x, n_inputs)
    prediction = model.use_model()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_name)

        result = prediction.eval(feed_dict={x_placeholder: x})
        return result


def predict_casualties(input):
    start_time = time.time()
    CASUALTIES_x = input
    result = use_neural_network(CASUALTIES_x, CASUALTIES_MODEL)
    return result


def predict_damaged_houses(input):
    start_time = time.time()
    DAMAGED_HOUSES_x = input
    result = use_neural_network(DAMAGED_HOUSES_x, DAMAGED_HOUSES_MODEL)
    return result


def predict_damaged_properties(input):
    start_time = time.time()
    DAMAGED_PROPERTIES_x = input
    result = use_neural_network(DAMAGED_PROPERTIES_x, DAMAGED_PROPERTIES_MODEL)
    return result

def load_file(file_name):
    data = []
    with open(file_name, 'rU') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data.append(row[0:-2])

    return data

def load_sug(file_name):
    data = []
    with open(file_name, 'rU') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row[6:])

    return data

def create_file(file_name, data):
    with open(file_name, 'w') as f:
        for row in data:
            f.write(",".join(str(item) for item in row) + '\n')


cas = load_file("CAS_suggested.csv")
cas_sug = load_sug("cas_suggest.csv")

#[print(row) for row in cas]

newset = []
for row,sug in zip(cas,cas_sug):
    for i in range(len(sug)):
        newrow = list(row)
        newrow[2+i] = sug[i]
        newset.append(newrow)

[print(row) for row in newset]


result = get_predicted_levels(predict_casualties(newset))

sup = []
row = []
for i in result:
    if len(row) == 4:
        print(row)
        sup.append(row)
        row = []
        row.append(i)
    else:
        row.append(i)
sup.append(row)

pred = get_predicted_levels(predict_casualties(cas))
print( len(pred))
print( len(sup))
print( len(result))

for i,j in zip(pred,sup):
    print(",".join(str(item) for item in ([i]+j)))



dah = load_file("DAH_suggested.csv")
dah_sug = load_sug("dah_suggest.csv")

dap = load_file("DAP_suggested.csv")
dap_sug = load_sug("dap_suggest.csv")
