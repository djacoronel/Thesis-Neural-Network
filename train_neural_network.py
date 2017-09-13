import tensorflow as tf
from training_functions import TrainingSettings as train

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
HMB, HMC, HMD,\
CASUALTIES, DAMAGED_HOUSES, DAMAGED_PROPERTIES = tf.decode_csv(value, record_defaults=record_defaults)

feature_list_1 = [DURATION, WIND, INTENSITY, SIGNAL, DR, FLR]
col_y_1 = CASUALTIES

feature_list_2 = [DURATION, WIND, INTENSITY, SIGNAL, DEN, FLR, HMB, HMD]
col_y_2 = DAMAGED_HOUSES

feature_list_3 = [WIND, INTENSITY, SIGNAL, DEN, DR, FLR, HS, HMB, HMD]
col_y_3 = DAMAGED_PROPERTIES

model_name_1 = "models/casualties_test.ckpt"
model_name_2 = "models/damagedhouses_test.ckpt"
model_name_3 = "models/damagedproperties_test.ckpt"

print("\n\n***********CASUALTIES**************")
training = train(model_name_1, feature_list_1, col_y_1)
training.learning_rate = 0.01
training.n_epoch = 1
training.train_neural_network()

print("\n\n***********DAMAGED HOUSES**************")
training = train(model_name_2, feature_list_2, col_y_2)
training.learning_rate = 0.0001
training.n_epoch = 1
training.train_neural_network()

print("\n\n***********DAMAGED PROPERTIES**************")
training = train(model_name_3, feature_list_3, col_y_3)
training.learning_rate = 0.01
training.n_epoch = 12
training.train_neural_network()