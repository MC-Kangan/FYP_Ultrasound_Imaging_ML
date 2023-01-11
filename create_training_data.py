import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import cv2 as cv
from PIL import Image
import pickle
from sklearn.preprocessing import MinMaxScaler


def read_gray_image(dirName, imageName, plotting=False):
    image = cv.imread(f'{dirName}/{imageName}', cv.IMREAD_GRAYSCALE)
    # print(f'The shape of the GREYSCALE image is {image.shape}')
    if plotting == True:
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()
    return image


def create_training_data(subsample = True, scale = False, picklename = 'training_data'):
    training_data = []
    dirName_fmc = '/Users/chenkangan/Desktop/ME4_FYP/imageGenerate_2022/FMC_variable'
    dirName_pic = '/Users/chenkangan/PycharmProjects/ME4_FYP_py/py_output_fig'

    for index in range(1, 2001):
        filename = f'fmc_{index}.mat'
        imgName = f'defect_{index}_yshift_60.png'

        mat = scipy.io.loadmat(f'{dirName_fmc}/{filename}')
        fmc = mat['timeTraces']

        if subsample == True:
            fmc = fmc[0::5]

        if scale == True:
            scaler = MinMaxScaler()
            fmc = scaler.fit_transform(fmc.reshape(-1, fmc.shape[-1])).reshape(fmc.shape)

        image = read_gray_image(dirName_pic, imgName, plotting=False)
        # I need to append ([fmc, label])
        training_data.append([fmc, image, index])
        print(f'{round(index/2000, 3)*100}% Completed')

    # pickle_out = open(f"{picklename}.pickle", "wb")
    # pickle.dump(training_data, pickle_out)
    # pickle_out.close()

    return training_data


if __name__ == "__main__":
    # subsample every 5th data
    # training_data = create_training_data(subsample=True, scale = True, picklename = 'training_data_subsampled_scaled')


    # The below code seperate training data into X and y and stored in different pickle files
    pickle_in = open("training_data_subsampled_scaled.pickle", "rb")
    train_data = np.array(pickle.load(pickle_in), dtype=object)

    X, y = [], []
    for features, label, index in train_data:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, 895, 16, 16)
    y = np.array(y).reshape(-1, 180, 240, 1)


    pickle_out = open("training_data_subsampled_scaled_X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("training_data_subsampled_scaled_y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()