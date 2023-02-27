
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

    opt = tf.keras.optimizers.Adam(learning_rate=0.00019409069392840677)

    # Compile the model
    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    return model

def get_dataset(num_sample, x_dimension, img_resize_factor):

    batch_size = 32

    X, y = load_training_data(num_sample=num_sample, x_dimension=x_dimension, img_resize_factor=img_resize_factor,
                              shrinkx=True, stack=False)

    # change the type (in order to build tf pipeline)
    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    num_val_samples = int(len(x_train) * 0.2)

    # Reserve num_val_samples samples for validation
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    input_size, output_size = x_train.shape[1:], y_train.shape[1]
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
        input_size,
        output_size
    )


if __name__ == "__main__":
    num_sample = 3200
    x_dimension = 3
    img_resize_factor = 50
    epochs = 50

    # Build tensorflow data pipeline for better efficiency

    modelname = model_namer(dimension=x_dimension, num_sample=num_sample, sub_sample=5, fmc_scaler=1.75e-14,
                            img_resize=img_resize_factor, remark='pooling-lr', epochs=epochs, version=3)

    print(f'The model name is {modelname}')


    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Load datasets
    train_dataset, val_dataset, test_dataset, input_size, output_size = get_dataset(num_sample, x_dimension,
                                                                                    img_resize_factor)
    # Open a strategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        model = create_model_3D(input_size, output_size)

    # Train the model on all available devices.
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
    model.fit(train_dataset,
              epochs=epochs,
              validation_data=val_dataset,
              verbose=1,
              callbacks=[callback])


    save_ml_model(model, modelname)
