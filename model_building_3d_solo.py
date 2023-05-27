# Last update: 26/5/2023
# This file train a single model without cross-validation


import os
import sys

HPC = False
print('cmd entry:', sys.argv)
# Declare the global variable, NCPUS: number of cpus

if HPC == True:
    NCPUS = int(sys.argv[1])
    MODEL_NUM = int(sys.argv[2])
    save = True
else:
    NCPUS = 4
    MODEL_NUM = 1

os.environ["OMP_NUM_THREADS"] = str(NCPUS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NCPUS)
os.environ["MKL_NUM_THREADS"] = str(NCPUS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(NCPUS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NCPUS)

import pandas as pd
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
from sklearn.model_selection import KFold
from utility_function import img_resize, model_namer, model_namer_description, save_ml_model, load_ml_model, load_training_data


def create_model_3D(input_size, output_size, model_num):
    model = Sequential()
    model.add(Input(input_size))

    if model_num == 1:  # 3D_2450_o7167535
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

    elif model_num == 2:  # 3D_2000_o7158970
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

    elif model_num == 3:  # 3D_2000_o7127128
        model.add(Conv3D(filters=10, kernel_size=(5, 3, 3), strides=(1, 1, 1), activation='relu'))
        model.add(MaxPool3D())
        model.add(Flatten())
        model.add(Dense(units=1500, activation='softsign'))
        model.add(Dense(units=output_size, activation='tanh'))
        learning_rate = 0.00017533560552392052
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mean_squared_error',
                      metrics=[tf.keras.metrics.MeanSquaredError()])

    elif model_num == 4:  # 3D_2000_o7121864
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

    elif model_num == 5:  # 3D_2000_o7127128-2
        model.add(Conv3D(filters=4, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='tanh'))
        model.add(Dropout(0.1))
        model.add(MaxPool3D())
        model.add(Flatten())
        model.add(Dense(units=1000, activation='softsign'))
        model.add(Dense(units=output_size, activation='tanh'))
        learning_rate = 0.0001960411264334168
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='mean_squared_error',
                      metrics=[tf.keras.metrics.MeanSquaredError()])

    return model


if __name__ == "__main__":

    x_dimension = 3
    img_resize_factor = 50
    epochs = 100

    X, y = load_training_data(num_sample=2450, x_dimension=x_dimension, img_resize_factor=img_resize_factor,
                              shrinkx=False, stack=False)

    print('')

    model_num_list = [MODEL_NUM]
    model_name_dict = {1: '3D_2450_o7167535',
                       2: '3D_2000_o7158970',
                       3: '3D_2000_o7127128',
                       4: '3D_2000_o7121864',
                       5: '3D_2000_o7127128-2'}


    for i in model_num_list:
        modelname = model_name_dict[i]
        print(f'Model {i}: The model name is {modelname}')

        # Build model.
        input_size, output_size = X.shape[1:], y.shape[1]
        print(f'Input_size: {input_size}; Output_size: {output_size}')

        final_loss_lst, final_val_loss_lst = [], []
        result_dict = {}

        fold_no = 1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


        print(f'Train_size: {X_train.shape}; Test_size: {X_test.shape}')

        model = create_model_3D(input_size, output_size, model_num = i)
        # print(model.summary())

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)

        # Generate a print
        print('------------------------------------------------------------------------')

        # Fit data to model
        train_history = model.fit(X_train, y_train,
                            batch_size=32,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(X_test, y_test),
                            callbacks=[callback],
                            workers=NCPUS)

        # Generate generalization metrics
        scores = model.evaluate(X_test, y_test, verbose=0)
        print(f'The score is {scores}')
        loss = train_history.history['loss'][-1]
        val_loss = train_history.history['val_loss'][-1]

        result_dict[f'loss_{fold_no}'] = train_history.history['loss']
        result_dict[f'Val_loss_{fold_no}'] = train_history.history['val_loss']

        final_loss_lst.append(loss)
        final_val_loss_lst.append(val_loss)
        print(f'Score_for_fold_{fold_no} - loss: {loss} - val_loss: {val_loss}')

        result_df = pd.DataFrame(result_dict)
        filename = f'{modelname}_kv_training_log_solo.csv'
        result_df.to_csv(f'NN_model/{filename}')

        print(f'Model_{modelname} - average_loss: {np.mean(final_loss_lst)} - average_val_loss: {np.mean(final_val_loss_lst)}')
        print('') # print an empty line

        # Save the model
        if save == True:
            save_ml_model(model, modelname)


