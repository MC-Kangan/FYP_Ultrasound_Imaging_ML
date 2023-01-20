import os
import zipfile
import numpy as np
import tensorflow as tf
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


if __name__ == "__main__":
    model = load_model(f'3d_cnn.h5', compile=False)

    pickle_in_x = open("training_data_subsampled_scaled_X.pickle", "rb")
    pickle_in_y = open("training_data_subsampled_scaled_y.pickle", "rb")
    X = pickle.load(pickle_in_x)
    y = pickle.load(pickle_in_y)
    # try training 50 fmc first

    X = X[51]
    X = X.reshape(-1, 895, 16, 16, 1)

    print(X.shape)
    y = y[51]
    # y = y / 255
    # y = y.reshape(-1, -1)

    y = model.predict(X)
    img = y.reshape(180, 240)
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.axis('off')
    print(img.shape)

