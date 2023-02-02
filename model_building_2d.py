
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, AveragePooling2D
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

def create_model_2D(input_size, output_size):

    model = Sequential()
    model.add(Input(input_size))
    model.add(Dropout(0.2))
    model.add(Conv2D(1, (5, 3), activation='tanh'))
    # model.add(Dropout(0.1))
    # model.add(Conv2D(16, (3, 3), activation='tanh'))
    model.add(Dropout(0.1))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=200, activation='softsign'))
    model.add(Dropout(0.4))

    model.add(Dense(units=output_size, activation='tanh'))

    return model


if __name__ == "__main__":

    x_dimension = 2
    img_resize_factor = 50
    epochs = 100

    X, y = load_training_data(num_sample=1000, x_dimension=x_dimension, img_resize_factor=img_resize_factor,
                              shrinkx=False, stack=False)
    input_size, output_size = X.shape[1:], y.shape[1]

    modelname = model_namer(dimension=x_dimension, num_sample=len(X), sub_sample=5, fmc_scaler=1.75e-14,
                           img_resize=img_resize_factor, remark='pooling', epochs=epochs, version=4)

    print(f'The model name is {modelname}')
    # Build model.
    model = create_model_2D(input_size, output_size)
    print(model.summary())

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    # Compile the model
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    # Fit data to model
    history = model.fit(X, y,
                        batch_size=30,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.3,
                        callbacks=[callback])

    save_ml_model(model, modelname)