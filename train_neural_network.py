from training_functions import TrainingSettings as train

model_name_1 = "model/casualties_test.ckpt"
model_name_2 = "model/damagedhouses_test.ckpt"
model_name_3 = "model/damagedproperties_test.ckpt"

print("\n\n***********CASUALTIES**************")
training = train(model_name_1)
training.n_epoch = 1
training.train_and_test_network()

print("\n\n***********DAMAGED HOUSES**************")
training = train(model_name_2)
training.n_epoch = 1
training.train_and_test_network()

print("\n\n***********DAMAGED PROPERTIES**************")
training = train(model_name_3)
training.n_epoch = 1
training.train_and_test_network()
