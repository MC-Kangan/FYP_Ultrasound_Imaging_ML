import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling1D, Flatten, Dropout
import tensorflow as tf


# define rgb-grayscale function
def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def limits(low, up):
    nlim = up - low
    return low, up, nlim


def imshrink(original, N):
    s = original.shape
    nshape = (int(s[0] / N), int(s[1] / N))
    new = np.zeros(nshape)
    new = original[0:s[0]:N, 0:s[1]:N]
    return new


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


# set n as number of data samples
n = 2000

# set x and y limits to prevent white border on images
lowlim_xi, uplim_xi, nxi = limits(130, 600)
lowlim_yi, uplim_yi, nyi = limits(200, 1000)

# shrinking factor
N = 5
nxi = int(nxi / N)
nyi = int(nyi / N)

for i in range(0, n):
    # load in all jpg as images and csv as FMCs
    im = open('Image_{0}.jpg'.format(i))
    ima = pl.imread(im.name)
    fm = open('FMC_{0}.csv'.format(i))
    fmc = pd.read_csv(fm.name)

    # first time only, to get shapes etc
    if i == 0:
        # find shapes of FMC and images
        npxi, npyi = ima.shape[:2]
        npxf, npyf = fmc.shape[:2]

        # set up inputs/output
        X = np.zeros((n, npxf, npyf, 1))
        Y = np.zeros((n, nxi, nyi))

    # set up arrays to store images and FMCs
    bwimages = np.zeros((npxi, npyi))
    FMCs = np.zeros((npxf, npyf))

    # add FMC to FMCs
    FMCs[:, :] = fmc
    # convert to input
    # X[i,:,:] = np.expand_dims(FMCs.T.reshape((npxf*npyf)),1)
    X[i, :, :, :] = np.expand_dims(FMCs, 2)

    # convert image to bw
    bwimages = rgb2gray(ima)
    # convert to output
    bwimages = bwimages[(lowlim_xi):(uplim_xi), (lowlim_yi):(uplim_yi)]
    bwimages = imshrink(bwimages, N)
    m = np.mean(bwimages)
    std = np.std(bwimages)

    bwimages[bwimages <= m + 3.2 * std] = 0
    bwimages[bwimages > m + 3.2 * std] = 1
    Y[i, :, :] = bwimages

# scaling param for FMC (X)
scal = np.sum(abs(X[0, np.nonzero(X[0, :])])) / np.count_nonzero(X[0, :])

# scale FMC (X) by scaling factor
X /= scal

# --------------------------------------------------------------
# Neural Network
# --------------------------------------------------------------


# parameters
epochs = 800
batch_size = 5
validation_split = 0.2

# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# set up a sequential neural network
model = Sequential()
# add layers
model.add(Input(X.shape[1:]))
model.add(Conv2D(1, (80, 57), activation='tanh'))
# model.add(MaxPooling2D((2,1)))
model.add(Conv2D(1, (40, 10), activation='tanh'))
model.add(Conv2D(1, (30, 20), activation='tanh'))
model.add(Conv2D(1, (26, 10), activation='tanh'))
model.add(Conv2D(1, (14, 4), activation='tanh'))

# #compile the model
model.compile(loss="mean_squared_error", optimizer='adam')
# summary of layers/parameters
model.summary()
# # history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[callback], verbose=1, shuffle=True)
history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0,
                    shuffle=True)
# model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1, shuffle=True)

# Save the model
# model.save_weights('FYP.h5')


Y_predict = model.predict(X)

archi = '2C 3D xlim 600, Conv2d'

# Figure Plotting time
pl.figure()
pl.plot(history.history['loss'])
pl.plot(history.history['val_loss'])
pl.title('Model loss')
pl.ylabel('Loss')
pl.xlabel('Epoch')
pl.legend(['Train', 'Test'], loc='upper right')
pl.savefig('Model Loss' + archi + ' BS' + str(batch_size) + ' E' + str(epochs) + '.png')

# Figure Plotting time
pl.figure()
pl.plot(smooth(history.history['loss'], 0.9), '--')
pl.plot(smooth(history.history['val_loss'], 0.9), '--')
pl.title('Model loss smoothed')
pl.ylabel('Loss')
pl.xlabel('Epoch')
pl.legend(['Train', 'Test'], loc='upper right')
pl.savefig('Smooth Model' + archi + ' BS' + str(batch_size) + ' E' + str(epochs) + '.png')

# change these to compare different parts of testing/valitdation set
sta = 8  # where to start
va = (1 - validation_split) * n  # end of training set (start of validation)
step1 = (va - sta) / 8  # choose step so it plots 8 images

actual = np.zeros((nxi, nyi))  # set up arrays to hold actual/predicted images
pred = np.zeros((nxi, nyi))

# selection of indices of images in validation set
selection = np.array([1842, 1846, 1847, 1848, 1849, 1851, 1861, 1871, 1886, 1964, 1998])

# plot actual and predicted images from the slections of the training and
# validation sets
for test in range(int(sta), int(va), int(step1)):
    actual = Y[test, :].T.reshape(nxi, nyi)

    pl.figure()
    pl.imshow(actual)
    pl.imsave('Actual_E' + str(epochs) + '_{0}.png'.format(test), actual)

    pred = Y_predict[test, :, 0]
    pl.figure()
    pl.imshow(pred)
    pl.imsave('Predict_E' + str(epochs) + '_{0}.png'.format(test), pred, cmap='inferno')

for test in range(len(selection)):
    actual = Y[selection[test], :].T.reshape(nxi, nyi)

    pl.figure()
    pl.imshow(actual)
    pl.imsave('Val_Actual_E' + str(epochs) + '_{0}.png'.format(test), actual)

    pred = Y_predict[selection[test], :, 0]

    pl.figure()
    pl.imshow(pred)
    pl.imsave('Val_Predicted_E' + str(epochs) + '_{0}.png'.format(test), pred, cmap='inferno')