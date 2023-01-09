import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import cv2 as cv
from PIL import Image
import pickle

def read_gray_image(dirName, imageName, plotting=False):
    image = cv.imread(f'{dirName}/{imageName}', cv.IMREAD_GRAYSCALE)
    # print(f'The shape of the GREYSCALE image is {image.shape}')
    if plotting == True:
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()
    return image


def create_training_data(subsample=True):
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
        else:
            pass

        image = read_gray_image(dirName_pic, imgName, plotting=False)
        # I need to append ([fmc, label])
        training_data.append([fmc, image, index])
        print(f'{round(index/2000, 3)*100}% Completed')

    pickle_out = open("training_data_subsampled.pickle", "wb")
    pickle.dump(training_data, pickle_out)
    pickle_out.close()

    return training_data


if __name__ == "__main__":
    # subsample every 5th data
    training_data = create_training_data(subsample=True)