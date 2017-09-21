from training_functions import TrainingSettings as train

model_name_1 = "models/casualties_test.ckpt"
# model_name_1 = "models/insurance_test.ckpt"
model_name_2 = "models/damagedhouses_test.ckpt"
model_name_3 = "models/damagedproperties_test.ckpt"

print("\n\n***********CASUALTIES**************")
training = train(model_name_1)
# training.n_epoch = 200
# training.n_total_data = 63
# training.batch_size = 10
# training.n_batches = int(training.n_total_data/training.batch_size)
training.train_neural_network()

print("\n\n***********DAMAGED HOUSES**************")
training = train(model_name_2)
training.train_neural_network()

print("\n\n***********DAMAGED PROPERTIES**************")
training = train(model_name_3)
training.train_neural_network()