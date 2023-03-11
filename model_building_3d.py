
import os
# Declare the global variable, NCPUS: number of cpus
NCPUS = 4

os.environ["OMP_NUM_THREADS"] = str(NCPUS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NCPUS)
os.environ["MKL_NUM_THREADS"] = str(NCPUS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NCPUS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NCPUS)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, AveragePooling3D
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.models import Model
import pickle
import numpy as np
import keras
import cv2 as cv
from utility_function import img_resize, model_namer, model_namer_description, save_ml_model, load_ml_model, load_training_data

def create_model_3D(input_size, output_size, model_num):

    model = Sequential()
    model.add(Input(input_size))

    if model_num == 1: # 3D_2450_o7167535
        model.add(Conv3D(filters=6, kernel_size=(4, 3, 3), strides=(1, 1, 1), activation='tanh'))
        model.add(Dropout(0.1))
        model.add(MaxPool3D())
        model.add(Flatten())
        model.add(Dense(units=2000, activation='tanh'))
        model.add(Dense(units=output_size, activation='tanh'))
        learning_rate = 0.00019617145132549103
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mean_squared_error',
                      metrics=[tf.keras.metrics.MeanSquaredError()])

    elif model_num == 2: # 3D_2000_o7158970
        model.add(Conv3D(filters=2, kernel_size=(5, 3, 3), strides=(1, 1, 1), activation='relu'))
        model.add(MaxPool3D())
        model.add(Flatten())
        model.add(Dense(units=1500, activation='softsign'))
        model.add(Dropout(0.1))
        model.add(Dense(units=output_size, activation='tanh'))
        learning_rate = 0.00011146694480809503
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mean_squared_error',
                      metrics=[tf.keras.metrics.MeanSquaredError()])

    elif model_num == 3: # 3D_2000_o7127128
        model.add(Conv3D(filters=10, kernel_size=(5, 3, 3), strides=(1, 1, 1), activation='relu'))
        model.add(MaxPool3D())
        model.add(Flatten())
        model.add(Dense(units=1500, activation='softsign'))
        model.add(Dense(units=output_size, activation='tanh'))
        learning_rate = 0.00017533560552392052
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mean_squared_error',
                      metrics=[tf.keras.metrics.MeanSquaredError()])

    elif model_num == 4:
        model = Sequential()
        model.add(Input(input_size))

        model.add(Conv3D(filters=5, kernel_size=(4, 3, 3), strides=(1, 1, 1), activation='tanh'))
        model.add(Dropout(0.1))

        model.add(MaxPool3D())

        model.add(Flatten())
        model.add(Dense(units=1000, activation='softsign'))
        model.add(Dropout(0.1))
        model.add(Dense(units=output_size, activation='tanh'))

        learning_rate = 0.0001735701471314462
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mean_squared_error',
                      metrics=[tf.keras.metrics.MeanSquaredError()])

    return model


if __name__ == "__main__":

    x_dimension = 3
    img_resize_factor = 50
    epochs = 1

    X, y = load_training_data(num_sample=10, x_dimension=x_dimension, img_resize_factor=img_resize_factor,
                              shrinkx=False, stack=False)

    model_num_list = [1,2,3,4]
    model_name_dict = {1: '3D_2450_o7167535',
                       2: '3D_2000_o7158970',
                       3: '3D_2000_o7127128',
                       4: '3D_2000_o7121864'}

    for i in model_num_list:
        modelname = model_name_dict[i]
        print(f'The model name is {modelname}')

        # Build model.
        input_size, output_size = X.shape[1:], y.shape[1]
        print(f'Input_size: {input_size}; Output_size: {output_size}')

        model = create_model_3D(input_size, output_size, model_num = i)
        print(model.summary())

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)

        # Fit data to model
        history = model.fit(X, y,
                            batch_size=32,
                            epochs=epochs,
                            verbose=1,
                            validation_split=0.2,
                            callbacks=[callback],
                            workers=NCPUS)

        # save_ml_model(model, modelname)
