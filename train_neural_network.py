from training_functions import TrainingSettings as train

def set_parameters(training, dataset_source):
    training.dataset_source = dataset_source
    training.n_epoch = 50
    training.batch_size = 5
    training.n_batches = int(training.n_total_data / training.batch_size)
    training.learning_rate = .001

dataset_source = "/csv/ds-meannormalisation.csv"

model_name_1 = "models/normalized_ds/mean/casualties_test.ckpt"
model_name_2 = "models/normalized_ds/mean/damagedhouses_test.ckpt"
model_name_3 = "models/normalized_ds/mean/damagedproperties_test.ckpt"

print("\n\n***********CASUALTIES**************")
training = train(model_name_1)
set_parameters(training,dataset_source)
training.train_and_test_network()

print("\n\n***********DAMAGED HOUSES**************")
training = train(model_name_2)
set_parameters(training,dataset_source)
training.train_and_test_network()

print("\n\n***********DAMAGED PROPERTIES**************")
training = train(model_name_3)
set_parameters(training,dataset_source)
training.train_and_test_network()


dataset_source = "/csv/ds-minmaxscaling.csv"

model_name_1 = "models/normalized_ds/minmax/casualties_test.ckpt"
model_name_2 = "models/normalized_ds/minmax/damagedhouses_test.ckpt"
model_name_3 = "models/normalized_ds/minmax/damagedproperties_test.ckpt"

print("\n\n***********CASUALTIES**************")
training = train(model_name_1)
set_parameters(training,dataset_source)
training.train_and_test_network()

print("\n\n***********DAMAGED HOUSES**************")
training = train(model_name_2)
set_parameters(training,dataset_source)
training.train_and_test_network()

print("\n\n***********DAMAGED PROPERTIES**************")
training = train(model_name_3)
set_parameters(training,dataset_source)
training.train_and_test_network()


dataset_source = "/csv/ds-standardization.csv"

model_name_1 = "models/normalized_ds/standard_deviation/casualties_test.ckpt"
model_name_2 = "models/normalized_ds/standard_deviation/damagedhouses_test.ckpt"
model_name_3 = "models/normalized_ds/standard_deviation/damagedproperties_test.ckpt"

print("\n\n***********CASUALTIES**************")
training = train(model_name_1)
set_parameters(training,dataset_source)
training.train_and_test_network()

print("\n\n***********DAMAGED HOUSES**************")
training = train(model_name_2)
set_parameters(training,dataset_source)
training.train_and_test_network()

print("\n\n***********DAMAGED PROPERTIES**************")
training = train(model_name_3)
set_parameters(training,dataset_source)
training.train_and_test_network()


dataset_source = "/csv/ds-thousands.csv"

model_name_1 = "models/normalized_ds/thousands/casualties_test.ckpt"
model_name_2 = "models/normalized_ds/thousands/damagedhouses_test.ckpt"
model_name_3 = "models/normalized_ds/thousands/damagedproperties_test.ckpt"

print("\n\n***********CASUALTIES**************")
training = train(model_name_1)
set_parameters(training,dataset_source)
training.train_and_test_network()

print("\n\n***********DAMAGED HOUSES**************")
training = train(model_name_2)
set_parameters(training,dataset_source)
training.train_and_test_network()

print("\n\n***********DAMAGED PROPERTIES**************")
training = train(model_name_3)
set_parameters(training,dataset_source)
training.train_and_test_network()
