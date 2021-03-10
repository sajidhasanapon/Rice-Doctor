
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, Callback

import zipfile
import sys
from shutil import rmtree

from Lib.util import *


################################## HYPERPARAMETERS ###################################################
learning_rate_list = [0.01, 0.001, 0.0001]
dropout_list = [0.2, 0.3, 0.5]
dense_layer_size_list = [64, 128, 256]
######################################################################################################









######################################### LOADING DATA ################################################
datagen = ImageDataGenerator(rescale=1. / 255)

""" 
a generator contains an iterator of all the images and
their labels, divided into chunks of batch_size 
"""
stage_1_train_generator = datagen.flow_from_directory(
    stage_1_train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

stage_1_validation_generator = datagen.flow_from_directory(
    stage_1_validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

stage_2_train_generator = datagen.flow_from_directory(
    stage_2_train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

stage_2_validation_generator = datagen.flow_from_directory(
    stage_2_validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
#######################################################################################################







################################# CREATING OUTPUT FILE STREAMS ########################################

tuning_results_file = open("tuning_results.txt", "w")
print("learning rate\t\tdropout rate\t\tdense layer size\t\ttraining_accuracy\t\tvalidation accuracy", file=tuning_results_file)

best_param_file = open("hyperparameter.txt", "w")
########################################################################################################





############################################# TUNING ##################################################

best_validation_accuracy = -float("inf")
best_params = (None, None, None)

early_stopping = EarlyStopping(monitor='val_acc', patience=1, verbose=1)

for learning_rate in learning_rate_list:
    for dropout in dropout_list:
        for dense_layer_size in dense_layer_size_list:

            print(learning_rate, dropout, dense_layer_size)

            """
            Stage 1 learning
            """
            model = get_stage_1_model(learning_rate=learning_rate,
                                    dropout=dropout,
                                    dense_layer_size=dense_layer_size)
                
            history_1 = model.fit_generator(
                stage_1_train_generator,
                epochs=n_stage_1_epochs_tuning,
                validation_data=stage_1_validation_generator,
                initial_epoch=0,
                callbacks=[early_stopping])


            """
            Stage 2 learning
            """
            model2 = get_stage_2_model(model, learning_rate=learning_rate)

            history_2 = model2.fit_generator(
                stage_2_train_generator,
                epochs=n_stage_2_epochs_tuning,
                validation_data=stage_2_validation_generator,
                initial_epoch=0,
                callbacks=[early_stopping])

            
            """
            Noting down train accuracy and validation accuracy for current hyperparameter set
            """
            training_accuracy = history_2.history["acc"][-1]
            validation_accuracy = history_2.history["val_acc"][-1]
            print(learning_rate, "\t\t\t", dropout, "\t\t\t", dense_layer_size, "\t\t\t", training_accuracy, "\t\t\t", validation_accuracy, file=tuning_results_file)



            """
            Updating best hyperparameter combination
            """
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_params = (learning_rate, dropout, dense_layer_size)


# storing best hyperparameter combination
print(best_params[0], "\t", best_params[1], "\t", best_params[2], "\t", file=best_param_file)          
#######################################################################################################


tuning_results_file.close()
best_param_file.close()



################################# REMOVING TEMPORARY DATA #############################################
#rmtree("tempdata")
#######################################################################################################