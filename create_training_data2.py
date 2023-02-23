import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import cv2 as cv
from PIL import Image
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

'''
Date: 30/1/23
NOTE! This file is the most updated version of training data creation script. Ignore create_training_data.py

'''

def read_gray_image(dirName, imageName, plotting=False):
    image = cv.imread(f'{dirName}/{imageName}', cv.IMREAD_GRAYSCALE)
    # print(f'The shape of the GREYSCALE image is {image.shape}')
    if plotting == True:
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()
    return image


def create_training_data(subsample=True, picklename='training_data', saveX=True, savey=True, data_range = 3500):

    '''
    :param subsample: Subsample the fmc matrix by taking the every 5th data
    :param scale: Apply scaling on the fmc matrix
    :param picklename: The output pickle name
    :param saveX: Save X in a pickle, default true
    :param savey: Save y in a pickle, default true
    :return:
    '''
    dirName_fmc = '/Users/chenkangan/Desktop/ME4_FYP/imageGenerate_2022/FMC_variable'
    dirName_pic = '/Users/chenkangan/PycharmProjects/ME4_FYP_py/py_output_fig_no_backwall_crop'
    print(f'The fmc directory is {dirName_fmc.split("/")[-1]}')
    print(f'The image directory is {dirName_pic.split("/")[-1]}')

    X, y = [], []
    for index in range(1, data_range+1):

        if saveX:
            filename = f'fmc_{index}.mat'
            mat = scipy.io.loadmat(f'{dirName_fmc}/{filename}')
            fmc = mat['timeTraces']

            if subsample == True:
                fmc = fmc[0::5]
            X.append(fmc)
            
        if savey:
            imgName = f'defect_{index}_yshift_60.png'
            image = read_gray_image(dirName_pic, imgName, plotting=False)
            nx, ny = image.shape
            y.append(image)

        print(f'{round(index/data_range, 3)*100}% Completed')

    if len(X) != 0:
        X = np.array(X).reshape(-1, 895, 16, 16, 1)
        pickle_out = open(f"{picklename}_X.pickle", "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

    if len(y) != 0:
        y = np.array(y).reshape(-1, nx, ny)
        pickle_out = open(f"{picklename}_y.pickle", "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

    return X, y


if __name__ == "__main__":
    # subsample every 5th data
    train_data = create_training_data(subsample=True,
                                      picklename='data_subsampled_no_backwall_crop_3500',
                                      saveX=False,
                                      savey=True,
                                      data_range=3500)

