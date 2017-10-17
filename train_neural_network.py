from training_functions import TrainingSettings as train

dataset_source_1 = "arranged/byYear/LUZ-CA.csv"
dataset_source_2 = "arranged/byYear/LUZ-DAH.csv"
dataset_source_3 = "arranged/byYear/LUZ-DAP.csv"

model_name_1 = "arranged/byYear/LUZON/LUZON_cas.ckpt"
model_name_2 = "arranged/byYear/LUZON/LUZON_dh.ckpt"
model_name_3 = "arranged/byYear/LUZON/LUZON_dp.ckpt"

print("\n\n***********CASUALTIES**************")
feature_list = ["YEAR", "INTENSITY", "SIGNAL", "POP", "DR", "FLR", "HP", "HMB"]
training = train(dataset_source_1, model_name_1, feature_list, 0.9)
training.train_and_test_network()

print("\n\n***********DAMAGED HOUSES**************")
feature_list = ["YEAR", "SIGNAL", "POP", "AI", "DR", "FLR", "HP", "HS", "HMA", "HMB"]
training = train(dataset_source_2,model_name_2, feature_list)
training.train_and_test_network()

print("\n\n***********DAMAGED PROPERTIES**************")
feature_list = ["YEAR", "DURATION", "SIGNAL", "AI", "DR", "SR", "HP", "HS"]
training = train(dataset_source_3,model_name_3, feature_list)
training.train_and_test_network()


dataset_source_1 = "arranged/byYear/VIS-CA.csv"
dataset_source_2 = "arranged/byYear/VIS-DAH.csv"
dataset_source_3 = "arranged/byYear/VIS-DAP.csv"

model_name_1 = "arranged/byYear/VISAYAS/VIS_cas.ckpt"
model_name_2 = "arranged/byYear/VISAYAS/VIS_dh.ckpt"
model_name_3 = "arranged/byYear/VISAYAS/VIS_dp.ckpt"

print("\n\n***********CASUALTIES**************")
feature_list = ["DURATION", "WIND", "INTENSITY", "FLR"]
training = train(dataset_source_1,model_name_1,feature_list)
training.train_and_test_network()

print("\n\n***********DAMAGED HOUSES**************")
feature_list = ["YEAR", "DURATION", "WIND", "INTENSITY",
                "SIGNAL", "POP", "DEN", "AI", "PR", "DR",
                "SR", "FLR", "HP", "HS", "HMA", "HMC"]
training = train(dataset_source_2,model_name_2, feature_list)
training.train_and_test_network()

print("\n\n***********DAMAGED PROPERTIES**************")
feature_list = ["YEAR", "DURATION", "WIND", "INTENSITY", "SIGNAL", "AI", "SR"]
training = train(dataset_source_3,model_name_3, feature_list)
training.train_and_test_network()


dataset_source_1 = "arranged/byYear/MIN-CA.csv"
dataset_source_2 = "arranged/random/MIN-DAH.csv"
dataset_source_3 = "arranged/byYear/MIN-DAP.csv"

model_name_1 = "arranged/byYear/MINDANAO/MIN_cas.ckpt"
model_name_2 = "arranged/byYear/MINDANAO/MIN_dh.ckpt"
model_name_3 = "arranged/byYear/MINDANAO/MIN_dp.ckpt"

print("\n\n***********CASUALTIES**************")
feature_list = ["YEAR", "WIND", "SIGNAL", "AI", "DR", "FLR", "HMD"]
training = train(dataset_source_1,model_name_1,feature_list)
training.train_and_test_network()

print("\n\n***********DAMAGED HOUSES**************")
feature_list = ["YEAR", "DURATION", "WIND", "INTENSITY", "AI", "DR", "FLR", "HP"]
training = train(dataset_source_2,model_name_2, feature_list,0.9)
training.train_and_test_network()

print("\n\n***********DAMAGED PROPERTIES**************")
feature_list = ["DURATION", "TYPE", "WIND", "INTENSITY", "SIGNAL", "POP"]
training = train(dataset_source_3,model_name_3, feature_list)
training.train_and_test_network()

