
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
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

def create_model3():
    ## input layer
    input_layer = Input((895, 16, 16, 1))

    ## convolutional layers
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)

    ## add max pooling to obtain the most imformatic features
    pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
    conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)
    pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)

    ## perform batch normalization on the convolution outputs before feeding it to MLP architecture
    # pooling_layer2 = BatchNormalization()(pooling_layer2)
    flatten_layer = Flatten()(pooling_layer2)

    ## create an MLP architecture with dense layers : 4096 -> 512 -> 10
    ## add dropouts to avoid overfitting / perform regularization
    # dense_layer1 = Dense(units=2048, activation='relu')(flatten_layer)
    # dense_layer1 = Dropout(0.4)(dense_layer1)
    # dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
    # dense_layer2 = Dropout(0.4)(dense_layer2)
    # output_layer = Dense(units=43200, activation='relu')(dense_layer2)

    output_layer = Dense(units=43200, activation='relu')(flatten_layer)

    ## define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)

    return model



if __name__ == "__main__":

    pickle_in_x = open("training_data_subsampled_X.pickle", "rb")
    pickle_in_y = open("training_data_subsampled_y.pickle", "rb")
    X = pickle.load(pickle_in_x)
    y = pickle.load(pickle_in_y)
    # try training 50 fmc first
    X = X.reshape(-1, 895, 16, 16, 1)
    X = X[:500]
    X = X / 1.75e-14
    y = y[:500]
    y = y / 255
    y = y.reshape(500, -1)

    print(X.shape)
    print(y.shape)

    # Build model.
    model = create_model3()
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

    model.save('3d_cnn3.h5')