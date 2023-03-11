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
    # print(f'Image size after is {img_resize.shape} after resizing {scale_percent}%')

    return img_resize

def model_namer(dimension, num_sample, sub_sample, fmc_scaler, img_resize, remark, version, epochs):
    '''

    :param dimension: dimension of the cnn
    :param num_sample: number of sample trained
    :param sub_sample: fmc (X) sub-sampling frequency. Default, taking every 5th data
    :param fmc_scaler: scaler on fmc (X) value. Default, it is 1.75e-14
    :param img_scaler: scaler on img (y) value. Default, it is 1, unscaled
    :param version: version number
    :param remark: any remarks
    :param epochs: number of epochs
    :return:
    '''

    D0 = dimension
    D5 = version
    D1 = num_sample
    D2 = sub_sample
    D3 = fmc_scaler
    D4 = img_resize
    D6 = remark
    D7 = epochs

    filename = f'cnn{D0}d_v{D5}_{D1}_{D2}_{D3}_{D4}_{D6}_{D7}'
    return filename

def model_namer_description(filename):
    filename = filename[:-4]
    [name, D5, D1, D2, D3, D4, D6, D7] = filename.split('_')
    message = f'''Model description:\n 
                - Dimension: {name[-2]}\n 
                - Version: {D5}\n
                - Number of samples: {D1}\n
                - FMC subsampling frequency: {D2}\n
                - FMC scaler: {D3}\n
                - Img resize factor: {D4}\n
                - Remark: {D6}\n
                - Epochs: {D7}\n
    '''
    print(message)
    return {'Dimension': name[-2],
            'version': D5,
            'num_sample': D1,
            'sub_sample': D2,
            'fmc_scaler': D3,
            'img_resize': D4,
            'remark': D6,
            'epochs': D7}

def save_ml_model(model, filename) -> None:

    # Prepare versioned save file name
    save_file_name = f"{filename}.pkl"
    save_path = f'/Users/chenkangan/PycharmProjects/ME4_FYP_py/NN_model/{save_file_name}'

    pickle_out = open(save_path, "wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()
    return None

def load_ml_model(filename) -> None:

    save_path = f'/Users/chenkangan/PycharmProjects/ME4_FYP_py/NN_model/{filename}'

    pickle_in = open(save_path, "rb")
    model = pickle.load(pickle_in)
    return model

def load_training_data(num_sample = 2000, x_dimension = 3, img_resize_factor = 50, shrinkx = False, stack = False):

    pickle_in_x = open("data_subsampled_no_backwall_crop_3500_X.pickle", "rb")
    pickle_in_y = open("data_subsampled_no_backwall_crop_3500_y.pickle", "rb")
    X = pickle.load(pickle_in_x)[:num_sample]
    y = pickle.load(pickle_in_y)[:num_sample]
    if x_dimension == 3:
        X = X.reshape(-1, 895, 16, 16, 1)
    elif x_dimension == 2:
        if stack == False:
            X = X.reshape(-1, 895, 16*16, 1)
        else:
            X = X.reshape(-1, 895, 16, 16)

    if shrinkx == True:
        print(f'The original shape of X is {X.shape}')
        X = X[:, 30:]
        print(f'The shape of X after shrinking is {X.shape}')
    else:
        print(f'The shape of X is {X.shape}')

    X = X / 1.75e-14
    # resize y by img_resize_factor
    y = np.array([img_resize(i, img_resize_factor) for i in y])
    y = y.reshape(len(y), -1).astype('int')
    # make defect 1, make non-defect -1
    y[y == 0] = -1
    y[y > 0] = 1
    print(f'The shape of y is {y.shape}')

    return X, y

if __name__ == "__main__":
    # print('hello')
    # filename = model_namer(dimension=2, num_sample=1000, sub_sample=5, fmc_scaler=1, img_resize=1, remark='no', version=90, epochs=100)
    # filename = filename + '.pkl'
    # print(filename)
    # model_namer_description(filename)
    X, y = load_training_data(num_sample=3500, x_dimension=3, img_resize_factor=50, shrinkx=True, stack=False)
