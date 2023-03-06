
import os

# Declare the global variable, NCPUS: number of cpus
NCPUS = 12

os.environ["OMP_NUM_THREADS"] = str(NCPUS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NCPUS)
os.environ["MKL_NUM_THREADS"] = str(NCPUS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NCPUS) 
os.environ["NUMEXPR_NUM_THREADS"] = str(NCPUS)
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, AveragePooling3D, Conv2D, MaxPool2D, AveragePooling2D
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.optimizers import Adadelta
from keras.models import Model
import pickle
import numpy as np
import keras
import cv2 as cv
from utility_function import img_resize, model_namer, model_namer_description, save_ml_model, load_ml_model, load_training_data

# Declare the global variable, CNN dimension: 2 or 3
CNN_DIM = 2

# Declare the global variable, REMARK: usually the version number
REMARK = 'V1'

def build_model(hp):

    print(f'The dimension of the CNN model is {CNN_DIM}')
    if CNN_DIM == 3:
        input_size, output_size = (895, 16, 16, 1), 7200

        model = Sequential()
        model.add(Input(input_size))

        for i in range(hp.Int('layers', 1, 2)):

            model.add(Conv3D(filters=hp.Int('filters_' + str(i), min_value=1, max_value=16, step=1),
                             kernel_size=(hp.Int('kernels_' + str(i), min_value=3, max_value=5, step=1), 3, 3),
                             activation=hp.Choice('activation_' + str(i), ["relu", "tanh"])))

            if hp.Boolean("dropout_" + str(i)):
                model.add(Dropout(rate=0.1))

            if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
                model.add(MaxPool3D())
            else:
                model.add(AveragePooling3D())

    elif CNN_DIM == 2:
        input_size, output_size = (895, 256, 1), 7200

        model = Sequential()
        model.add(Input(input_size))

        for i in range(hp.Int('layers', 1, 5)):
            model.add(Conv2D(filters=hp.Int('filters_' + str(i), min_value=1, max_value=16, step=1),
                             kernel_size=(hp.Int('kernels_' + str(i), min_value=3, max_value=5, step=1), 3),
                             activation=hp.Choice('activation1_' + str(i), ["relu", "tanh"])))

            if hp.Boolean("dropout_" + str(i)):
                model.add(Dropout(rate=0.1))

            if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
                model.add(MaxPool2D())
            else:
                model.add(AveragePooling2D())

    model.add(Flatten())
    for j in range(hp.Int('dense_layers', 1, 2)):
        model.add(Dense(units=hp.Int('dense_units_'+str(j), min_value=500, max_value=2000, step=500),
                        activation=hp.Choice('activation2_' + str(j), ["softsign", "tanh"])))
        if hp.Boolean("dropout2_" + str(j)):
            model.add(Dropout(rate=0.1))

    model.add(Dense(units=output_size, activation='tanh'))

    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['accuracy'])

    return model


if __name__ == "__main__":

    print('Message 1:')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print('______')
    
    x_dimension = CNN_DIM
    img_resize_factor = 50
    epochs = 100

    # change num_sample = 2000 in practise (dont forget)
    X, y = load_training_data(num_sample=100, x_dimension=x_dimension, img_resize_factor=img_resize_factor,
                              shrinkx=False, stack=False)
    input_size, output_size = X.shape[1:], y.shape[1]

    if CNN_DIM == 3:
        project = f'3dcnn_all_gpu_{REMARK}'

    elif CNN_DIM == 2:
        project = f'2dcnn_all_gpu_{REMARK}'


    tuner = kt.Hyperband(build_model,
                         objective=kt.Objective("val_loss", direction="min"),
                         max_epochs=epochs,
                         factor=3,
                         directory='fyp_model_search',
                         project_name=project)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    #config = tf.compat.v1.ConfigProto(
    #    intra_op_parallelism_threads=1,
    #    inter_op_parallelism_threads=1
    #)
    
    tuner.search(x=X,
                 y=y,
                 epochs=epochs,
                 batch_size=32,
                 validation_split=0.2,
                 callbacks=[stop_early],
                 workers=NCPUS
                 )

    result = tuner.results_summary()

