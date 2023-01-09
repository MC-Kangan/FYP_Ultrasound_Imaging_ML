import os
import zipfile
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import pickle


def get_model(width=895, height=16, depth=16):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    # x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPool3D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPool3D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)
    #
    # x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPool3D(pool_size=2)(x)
    # x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=180 * 240, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


if __name__ == "__main__":
    # parameters
    epochs = 100
    batch_size = 5
    validation_split = 0.2

    pickle_in = open("training_data_subsampled_scaled.pickle", "rb")
    train_data = np.array(pickle.load(pickle_in), dtype=object)
    # try training 500 fmc first
    train_data = train_data[:500]

    X = train_data[:, 0].tolist()
    print(train_data[:, 0].shape)
    print(train_data[:, 0][0].shape)
    y = train_data[:, 1]
    print(train_data[:, 1].shape)
    y = y / 255
    y = y.tolist()

    # # Build model.
    # model = get_model(width=895, height=16, depth=16)
    # print(model.summary())
    #
    # # Compile model.
    # initial_learning_rate = 0.0001
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    # )
    # model.compile(
    #     loss="mean_squared_error",
    #     optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    #     metrics=["acc"],
    # )
    #
    # # Define callbacks.
    # checkpoint_cb = keras.callbacks.ModelCheckpoint(
    #     "3d_image_classification.h5", save_best_only=True
    # )
    # early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)
    #
    # # Train the model, doing validation at the end of each epoch
    # model.fit(
    #     X, y,
    #     validation_split=validation_split,
    #     epochs=epochs,
    #     shuffle=True,
    #     verbose=2,
    #     callbacks=[checkpoint_cb, early_stopping_cb],
    # )
    #
    # model.save('3d_cnn.h5')
