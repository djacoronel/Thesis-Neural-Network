from training_functions import NeuralNetworkTrainer as trainer

dataset_source_1 = "datasets/ANN/Q-CAS.csv"
dataset_source_2 = "datasets/ANN/Q-DAH.csv"
dataset_source_3 = "datasets/ANN/Q-DAP.csv"

model_name_1 = "datasets/ANN/Q_cas.ckpt"
model_name_2 = "datasets/ANN/Q_dah.ckpt"
model_name_3 = "datasets/ANN/Q_dap.ckpt"

print("\n\n***********CASUALTIES**************")
feature_list = ["YEAR", "INTENSITY", "WIND", "SIGNAL", "POP", "DR", "FLR", "HP", "HMB"]
training = trainer(dataset_source_1, model_name_1, feature_list, 0.8)
training.train_and_test_network()

print("\n\n***********DAMAGED HOUSES**************")
feature_list = ["YEAR", "SIGNAL", "POP", "AI", "DR", "FLR", "HP", "HS", "HMA", "HMB"]
training = trainer(dataset_source_2, model_name_2, feature_list, 0.8)
training.train_and_test_network()

print("\n\n***********DAMAGED PROPERTIES**************")
feature_list = ["YEAR", "DURATION", "SIGNAL", "AI", "DR", "SR", "HP", "HS"]
training = trainer(dataset_source_3, model_name_3, feature_list, 0.8)
training.train_and_test_network()
