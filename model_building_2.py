import os
import zipfile
import numpy as np
import tensorflow as tf
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization

def create_model2():

    sample_shape = (895, 16, 16, 1)
    # Create the model
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # model.add(BatchNormalization(center=True, scale=True))
    model.add(Dropout(0.5))
    # model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # model.add(BatchNormalization(center=True, scale=True))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    # model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(units=180 * 240, activation='sigmoid'))

    return model


if __name__ == "__main__":
    # parameters
    # epochs = 100
    # batch_size = 30
    # validation_split = 0.2

    pickle_in_x = open("data_subsampled_no_backwall_200_X.pickle", "rb")
    pickle_in_y = open("data_subsampled_no_backwall_200_y.pickle", "rb")
    X = pickle.load(pickle_in_x)
    y = pickle.load(pickle_in_y)
    # try training 50 fmc first
    X = X.reshape(-1, 895, 16, 16, 1)
    X = X / 1.75e-14
    # y = y / 255
    y = y.reshape(len(y), -1)

    print(X.shape)
    print(y.shape)

    # Build model.
    model = create_model2()
    print(model.summary())

    # Compile the model
    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])

    model.summary()
    # Fit data to model
    history = model.fit(X, y,
                        batch_size=30,
                        epochs=50,
                        verbose=1,
                        validation_split=0.3)

    model.save('3d_cnn4_no_img_scale.h5')


