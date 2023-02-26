
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.models import Model
import pickle
import numpy as np
import keras
import cv2 as cv
from utility_function import img_resize, model_namer, model_namer_description, save_ml_model, load_ml_model, load_training_data

def create_model_3D(input_size, output_size):

    model = Sequential()
    model.add(Input(input_size))
    # model.add(Dropout(0.2))
    model.add(Conv3D(filters=2, kernel_size=(5, 3, 3), strides=(1, 1, 1), activation='tanh'))
    model.add(Dropout(0.1))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    # model.add(Conv3D(16, (3, 3), activation='tanh'))
    # model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(units=1000, activation='softsign'))
    # model.add(Dropout(0.4))

    model.add(Dense(units=output_size, activation='tanh'))

    return model


if __name__ == "__main__":

    x_dimension = 3
    img_resize_factor = 50
    epochs = 50

    X, y = load_training_data(num_sample=3200, x_dimension=x_dimension, img_resize_factor=img_resize_factor,
                              shrinkx=True, stack=False)

    # change the type (in order to build tf pipeline)
    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Build tensorflow data pipeline for better efficiency
    # ref: https://github.com/codebasics/deep-learning-keras-tf-tutorial/blob/master/43_distributed_training/dgx_benchamrking_tf_mirrored_stratergy.ipynb
    train_tf_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_tf_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    strategy = tf.distribute.MirroredStrategy()

    BATCH_SIZE_PER_REPLICA = 70
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    train_dataset = train_tf_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_tf_dataset.batch(BATCH_SIZE)

    modelname = model_namer(dimension=x_dimension, num_sample=len(X), sub_sample=5, fmc_scaler=1.75e-14,
                            img_resize=img_resize_factor, remark='pooling-lr', epochs=epochs, version=3)

    print(f'The model name is {modelname}')

    # Build model.
    input_size, output_size = X_train.shape[1:], y_train.shape[1]

    model = create_model_3D(input_size, output_size)
    print(model.summary())

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)

    opt = tf.keras.optimizers.Adam(learning_rate=0.00019409069392840677)

    # Compile the model
    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    with strategy.scope():
        # Fit data to model
        history = model.fit(train_dataset,
                            batch_size=BATCH_SIZE,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[callback])


    save_ml_model(model, modelname)
