from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
import keras_tuner as kt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, AveragePooling3D
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.optimizers import Adadelta
from keras.models import Model
import pickle
import numpy as np
import keras
import cv2 as cv
from utility_function import img_resize, model_namer, model_namer_description, save_ml_model, load_ml_model, load_training_data


def build_model(hp):
    input_size, output_size = (865, 16, 16, 1), 7200
    model = Sequential()
    model.add(Input(input_size))
    for i in range(hp.Int('layers', 1, 2)):

        model.add(Conv3D(filters=hp.Int('filters_' + str(i), min_value=1, max_value=16, step=1),
                         kernel_size=(hp.Int('kernels_' + str(i), min_value=3, max_value=10, step=1), 3, 3),
                         activation=hp.Choice("activation_", ["relu", "tanh"])))

        if hp.Boolean("dropout"):
            model.add(Dropout(rate=0.1))

        if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
            model.add(MaxPool3D())
        else:
            model.add(AveragePooling3D())

    model.add(Flatten())
    model.add(Dense(units=hp.Int('dense_units_', min_value=1000, max_value=2000, step=500), activation='softsign'))
    if hp.Boolean("dropout2"):
        model.add(Dropout(rate=0.1))

    model.add(Dense(units=output_size, activation='tanh'))

    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['accuracy'])

    return model


if __name__ == "__main__":
    x_dimension = 3
    img_resize_factor = 50
    epochs = 50

    X, y = load_training_data(num_sample=100, x_dimension=x_dimension, img_resize_factor=img_resize_factor,
                              shrinkx=True, stack=False)
    input_size, output_size = X.shape[1:], y.shape[1]

    tuner = kt.Hyperband(build_model,
                         objective=kt.Objective("val_loss", direction="min"),
                         max_epochs=100,
                         factor=3,
                         directory='fyp_model_search',
                         project_name='3dcnn_1conv_all')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    tuner.search(x=X,
                 y=y,
                 epochs=epochs,
                 batch_size=32,
                 validation_split=0.2,
                 callbacks=[stop_early]
                 )

    result = tuner.results_summary()
