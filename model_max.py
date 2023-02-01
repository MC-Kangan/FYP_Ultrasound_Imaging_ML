
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, AveragePooling2D
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
# from plotly.offline import iplot, init_notebook_mode
# from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
# import plotly.graph_objs as go
from matplotlib.pyplot import cm
from keras.models import Model
import pickle
import numpy as np
import keras
import cv2 as cv
from utility_function import img_resize
from utility_function import model_namer
from utility_function import model_namer_description

def create_model_max(input_size, output_size):
    ## input layer

    model = Sequential()
    model.add(Input(input_size))
    model.add(Dropout(0.2))

    model.add(Conv2D(1, 3, activation='tanh'))
    model.add(Dropout(0.1))

    model.add(MaxPool2D(2))
    model.add(Dropout(0.4))

    model.add(Conv2D(1, 3, activation='tanh'))
    model.add(Dropout(0.1))

    model.add(AveragePooling2D(2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(units=200, activation='softsign'))
    model.add(Dropout(0.4))

    model.add(Dense(units=output_size, activation='tanh'))

    return model


if __name__ == "__main__":

    pickle_in_x = open("training_data_subsampled_X.pickle", "rb")
    pickle_in_y = open("training_data_subsampled_no_backwall_y.pickle", "rb")
    X = pickle.load(pickle_in_x)[:1000]
    y = pickle.load(pickle_in_y)[:1000]

    X = X.reshape(-1, 895, 16*16, 1)
    X = X / 1.75e-14
    print(X.shape)
    # resize y by 50%
    img_resize_factor = 100
    y = np.array([img_resize(i, img_resize_factor) for i in y])
    y = y.reshape(len(y), -1)
    y = y/255

    print(X.shape)
    print(y.shape)
    input_size, output_size = X.shape[1:], y.shape[1]

    # Build model.
    model = create_model_max(input_size, output_size)
    print(model.summary())

    # Compile the model
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    # Fit data to model
    epochs = 100
    history = model.fit(X, y,
                        batch_size=30,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.3)

    modelname = model_namer(dimension=2, num_sample=len(X), sub_sample=5, fmc_scaler=1.75e-14,
                           img_resize=img_resize_factor, img_scaler=1, remark='tanh_max', epochs=epochs)

    model.save(modelname)
