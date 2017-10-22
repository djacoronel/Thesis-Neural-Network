from training_functions import TrainingSettings as train

dataset_source_1 = "split_test/dataset/LUZ-CA.csv"
dataset_source_2 = "split_test/dataset/LUZ-DAH.csv"
dataset_source_3 = "split_test/dataset/LUZ-DAP.csv"
'''
model_name_1 = "split_test/0.95/LUZON_cas.ckpt"
model_name_2 = "split_test/0.95/LUZON_dh.ckpt"
model_name_3 = "split_test/0.95/LUZON_dp.ckpt"

print("\n\n***********CASUALTIES**************")
feature_list = ["YEAR", "INTENSITY", "SIGNAL", "POP", "DR", "FLR", "HP", "HMB"]
training = train(dataset_source_1, model_name_1, feature_list, 0.95)
training.train_and_test_network()

print("\n\n***********DAMAGED HOUSES**************")
feature_list = ["YEAR", "SIGNAL", "POP", "AI", "DR", "FLR", "HP", "HS", "HMA", "HMB"]
training = train(dataset_source_2,model_name_2, feature_list,0.95)
training.train_and_test_network()

print("\n\n***********DAMAGED PROPERTIES**************")
feature_list = ["YEAR", "DURATION", "SIGNAL", "AI", "DR", "SR", "HP", "HS"]
training = train(dataset_source_3,model_name_3, feature_list,0.95)
training.train_and_test_network()


model_name_1 = "split_test/0.90/LUZON_cas.ckpt"
model_name_2 = "split_test/0.90/LUZON_dh.ckpt"
model_name_3 = "split_test/0.90/LUZON_dp.ckpt"

print("\n\n***********CASUALTIES**************")
feature_list = ["YEAR", "INTENSITY", "SIGNAL", "POP", "DR", "FLR", "HP", "HMB"]
training = train(dataset_source_1, model_name_1, feature_list, 0.90)
training.train_and_test_network()

print("\n\n***********DAMAGED HOUSES**************")
feature_list = ["YEAR", "SIGNAL", "POP", "AI", "DR", "FLR", "HP", "HS", "HMA", "HMB"]
training = train(dataset_source_2,model_name_2, feature_list,0.90)
training.train_and_test_network()

print("\n\n***********DAMAGED PROPERTIES**************")
feature_list = ["YEAR", "DURATION", "SIGNAL", "AI", "DR", "SR", "HP", "HS"]
training = train(dataset_source_3,model_name_3, feature_list,0.90)
training.train_and_test_network()
'''

model_name_1 = "split_test/0.85/LUZON_cas.ckpt"
model_name_2 = "split_test/0.85/LUZON_dh.ckpt"
model_name_3 = "split_test/0.85/LUZON_dp.ckpt"

print("\n\n***********CASUALTIES**************")
feature_list = ["YEAR", "INTENSITY", "SIGNAL", "POP", "DR", "FLR", "HP", "HMB"]
training = train(dataset_source_1, model_name_1, feature_list, 0.85)
training.train_and_test_network()

print("\n\n***********DAMAGED HOUSES**************")
feature_list = ["YEAR", "SIGNAL", "POP", "AI", "DR", "FLR", "HP", "HS", "HMA", "HMB"]
training = train(dataset_source_2,model_name_2, feature_list,0.85)
training.train_and_test_network()

print("\n\n***********DAMAGED PROPERTIES**************")
feature_list = ["YEAR", "DURATION", "SIGNAL", "AI", "DR", "SR", "HP", "HS"]
training = train(dataset_source_3,model_name_3, feature_list,0.85)
training.train_and_test_network()


model_name_1 = "split_test/0.80/LUZON_cas.ckpt"
model_name_2 = "split_test/0.80/LUZON_dh.ckpt"
model_name_3 = "split_test/0.80/LUZON_dp.ckpt"

print("\n\n***********CASUALTIES**************")
feature_list = ["YEAR", "INTENSITY", "SIGNAL", "POP", "DR", "FLR", "HP", "HMB"]
training = train(dataset_source_1, model_name_1, feature_list, 0.80)
training.train_and_test_network()

print("\n\n***********DAMAGED HOUSES**************")
feature_list = ["YEAR", "SIGNAL", "POP", "AI", "DR", "FLR", "HP", "HS", "HMA", "HMB"]
training = train(dataset_source_2,model_name_2, feature_list,0.80)
training.train_and_test_network()

print("\n\n***********DAMAGED PROPERTIES**************")
feature_list = ["YEAR", "DURATION", "SIGNAL", "AI", "DR", "SR", "HP", "HS"]
training = train(dataset_source_3,model_name_3, feature_list,0.80)
training.train_and_test_network()


model_name_1 = "split_test/0.75/LUZON_cas.ckpt"
model_name_2 = "split_test/0.75/LUZON_dh.ckpt"
model_name_3 = "split_test/0.75/LUZON_dp.ckpt"

print("\n\n***********CASUALTIES**************")
feature_list = ["YEAR", "INTENSITY", "SIGNAL", "POP", "DR", "FLR", "HP", "HMB"]
training = train(dataset_source_1, model_name_1, feature_list, 0.75)
training.train_and_test_network()

print("\n\n***********DAMAGED HOUSES**************")
feature_list = ["YEAR", "SIGNAL", "POP", "AI", "DR", "FLR", "HP", "HS", "HMA", "HMB"]
training = train(dataset_source_2,model_name_2, feature_list,0.75)
training.train_and_test_network()

print("\n\n***********DAMAGED PROPERTIES**************")
feature_list = ["YEAR", "DURATION", "SIGNAL", "AI", "DR", "SR", "HP", "HS"]
training = train(dataset_source_3,model_name_3, feature_list,0.75)
training.train_and_test_network()


model_name_1 = "split_test/0.70/LUZON_cas.ckpt"
model_name_2 = "split_test/0.70/LUZON_dh.ckpt"
model_name_3 = "split_test/0.70/LUZON_dp.ckpt"

print("\n\n***********CASUALTIES**************")
feature_list = ["YEAR", "INTENSITY", "SIGNAL", "POP", "DR", "FLR", "HP", "HMB"]
training = train(dataset_source_1, model_name_1, feature_list, 0.70)
training.train_and_test_network()

print("\n\n***********DAMAGED HOUSES**************")
feature_list = ["YEAR", "SIGNAL", "POP", "AI", "DR", "FLR", "HP", "HS", "HMA", "HMB"]
training = train(dataset_source_2,model_name_2, feature_list,0.70)
training.train_and_test_network()

print("\n\n***********DAMAGED PROPERTIES**************")
feature_list = ["YEAR", "DURATION", "SIGNAL", "AI", "DR", "SR", "HP", "HS"]
training = train(dataset_source_3,model_name_3, feature_list,0.70)
training.train_and_test_network()

