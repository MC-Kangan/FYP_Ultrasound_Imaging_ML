import cv2 as cv
import pickle
import numpy as np

def img_resize(img, scale_percent = 50):
    """

    :param img: image array
    :param scale_percent: percent of original size
    :return:

    """
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img_resize = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    print(f'Image size after is {img_resize.shape} after resizing {scale_percent}%')

    return img_resize

def model_namer(dimension, num_sample, sub_sample, fmc_scaler, img_resize, img_scaler, activation, epochs):
    '''

    :param dimension: dimension of the cnn
    :param num_sample: number of sample trained
    :param sub_sample: fmc (X) sub-sampling frequency. Default, taking every 5th data
    :param fmc_scaler: scaler on fmc (X) value. Default, it is 1.75e-14
    :param img_resize: img (y) resize percentage, maximum 100%, minimum 0%
    :param img_scaler: scaler on img (y) value. Default, it is 1, unscaled
    :param activation: activation function used
    :param epochs: number of epochs
    :return:
    '''

    D0 = dimension
    D1 = num_sample
    D2 = sub_sample
    D3 = fmc_scaler
    D4 = img_resize
    D5 = img_scaler
    D6 = activation
    D7 = epochs

    filename = f'cnn{D0}d_{D1}_{D2}_{D3}_{D4}_{D5}_{D6}_{D7}.h5'
    return filename

def model_namer_description(filename):
    filename = filename[:-3]
    [name, D1, D2, D3, D4, D5, D6, D7] = filename.split('_')
    message = f'''Model description:\n 
                - Dimension: {name[-2]}\n 
                - Number of samples: {D1}\n
                - FMC subsampling frequency: {D2}\n
                - FMC scaler: {D3}\n
                - Img resize factor: {D4}\n
                - Img scaler: {D5}\n
                - Activation functions: {D6}\n
                - Epochs: {D7}\n
    '''
    print(message)
    return {'Dimension': name[-2],
            'num_sample': D1,
            'sub_sample': D2,
            'fmc_scaler': D3,
            'img_resize': D4,
            'img_scaler': D5,
            'activation': D6,
            'epochs': D7}

if __name__ == "__main__":
    # pickle_in_y = open("data_subsampled_no_backwall_200_y.pickle", "rb")
    # y = pickle.load(pickle_in_y)
    #
    # y = np.array([img_resize(i, 50) for i in y])
    # print(y.shape)
    a = model_namer_description('cnn3d_100_5_1.75e-14_100_1_relu_50.h5')
    print(a)