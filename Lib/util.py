
from keras import applications, optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense


############################################ VALUES ##################################################
img_width, img_height           = 224, 224
stage_1_train_data_dir          = 'tempdata/train/stage1'
stage_1_validation_data_dir     = 'tempdata/validation/stage1'
stage_2_train_data_dir          = 'tempdata/train/stage2'
stage_2_validation_data_dir     = 'tempdata/validation/stage2'

# tuning_results_file = "tuning_results.txt"
# best_param_file = "hyperparameter.txt"

n_stage_1_epochs_tuning     = 10
n_stage_2_epochs_tuning     = 10
n_stage_1_epochs_training   = 20
n_stage_2_epochs_training   = 20
batch_size  = 16
n_symptoms  = 12
n_diseases  = 5
#####################################################################################################






##################################### PHASE 1 MODEL ##################################################
def get_stage_1_model(learning_rate=0.001, dropout=0.3, dense_layer_size=128):
    # VGG16 is a famous CNN
    # we are adding the bottom layers of VGG16 as the basis of our model
    # we are adopting transfer learning, so the VGG16 layers won't be trained
    # hence, trainable = False
    vgg = applications.VGG16(include_top=False, weights='imagenet', input_shape=(img_width,img_height, 3))
    for layer in vgg.layers[:]:
        layer.trainable = False

    # this is our model
    model = Sequential()
    
    # Add the vgg convolutional base model
    model.add(vgg)

    # Add new layers
    model.add(Flatten())
    model.add(Dense(dense_layer_size, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(n_symptoms, activation='softmax'))

    # compile the model
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.adam(learning_rate),
                metrics=['accuracy'])

    return model
######################################################################################################





################################### PHASE 2 MODEL #####################################################
def get_stage_2_model(stage_1_model, learning_rate=0.001):
    model = stage_1_model
    model.layers.pop()
    new_layer = Dense(n_diseases, activation="softmax")(model.layers[-1].output)

    model = Model(input=model.input, output=new_layer)

    for layer in model.layers[:-2]:
        layer.trainable = False

    # compile the model
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.adam(learning_rate),
                metrics=['accuracy'])

    return model
#######################################################################################################