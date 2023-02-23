from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
import keras_tuner as kt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.optimizers import Adadelta
from matplotlib.pyplot import cm
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
    model.add(Conv3D(filters=2, kernel_size=(5, 3, 3), strides=(1, 1, 1), activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(units=1000, activation='softsign'))
    # model.add(Dropout(0.4))

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

    X, y = load_training_data(num_sample=1000, x_dimension=x_dimension, img_resize_factor=img_resize_factor,
                              shrinkx=True, stack=False)
    input_size, output_size = X.shape[1:], y.shape[1]

    tuner = kt.Hyperband(build_model,
                         objective=kt.Objective("loss", direction="min"),
                         max_epochs=100,
                         factor=3,
                         directory='fyp_model_search',
                         project_name='3dcnn_1conv_lr')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    tuner.search(x=X,
                 y=y,
                 epochs=epochs,
                 batch_size=32,
                 validation_split=0.2,
                 callbacks=[stop_early]
                 )

    result = tuner.results_summary()
