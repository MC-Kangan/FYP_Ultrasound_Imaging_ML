import cv2 as cv
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import math

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


def plot_defect_zoom(defect, index=-999, plot=False):
    '''
    Date: 22/3/23
    This function assumes array = False
    y_shift = 60,
    backwall = False,
    save = False,
    crop = True

    '''
    y_shift = 60
    scale_percent = 50

    # y_shift controls the offset of the image (shift up)

    color = (255, 255, 255)  # White color

    # Extract defect data from the array
    defWidth, defHeight, defPosx, defPosy, defAngR = defect

    # define an image
    # np.zeros(height,width)

    # Assumed the image is croped
    img = np.zeros((120, 240, 3), np.uint8)

    # Transform defect size to pixel size
    pixel_step = 0.5e-3
    pixel_width = defWidth / pixel_step
    pixel_height = defHeight / pixel_step
    x = np.arange(-0.06, 0.06, pixel_step)  # length = 240
    y = np.arange(0, 0.06, pixel_step)  # length = 180

    # Function that takes in an array and a target value,
    # return the index of an element that is nearest to the target value
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        # Return index
        return idx

    pixel_xc = find_nearest(x, defPosx)
    pixel_yc = 180 - find_nearest(y, defPosy)

    # add a negative sign to convert from clockwise to anticlockwise
    defAngR = -defAngR

    # compute sine and cosine
    cos_theta, sin_theta = np.cos(defAngR), np.sin(defAngR)

    # let w and h be half of the width and height
    w, h = pixel_width / 2, pixel_height / 2

    # Coordinate: check the notes for details
    # left up corner
    x1, y1 = (pixel_xc - w * cos_theta - h * sin_theta), (pixel_yc - w * sin_theta + h * cos_theta - y_shift)
    # right up corner
    x2, y2 = (pixel_xc + w * cos_theta - h * sin_theta), (pixel_yc + w * sin_theta + h * cos_theta - y_shift)
    # right down corner
    x3, y3 = (pixel_xc + w * cos_theta + h * sin_theta), (pixel_yc + w * sin_theta - h * cos_theta - y_shift)
    # left down corner
    x4, y4 = (pixel_xc - w * cos_theta + h * sin_theta), (pixel_yc - w * sin_theta - h * cos_theta - y_shift)

    pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
    # print(pts)

    # reformat the points
    pts = pts.reshape((-1, 1, 2))

    img = cv.fillPoly(img, [pts], color)
    # print(img.shape)

    #     if plot == True:
    #     # Plot the image
    #         fig, ax = plt.subplots()
    #         ax.imshow(img)
    #         plt.title(f'Shape of the image: {img.shape}')
    #         plt.axis('off')

    # Plot the defect of the resize image
    img_defect = cv.cvtColor(img, cv.IMREAD_GRAYSCALE)[int(min([y1, y2, y3, y4]) - 2): int(max([y1, y2, y3, y4]) + 2),
                 int(min([x1, x2, x3, x4]) - 2): int(max([x1, x2, x3, x4]) + 2)]

    #     if plot == True:
    #         fig, ax = plt.subplots()
    #         ax.imshow(img_defect)
    #         plt.title(f'Shape of the defect: {img_defect.shape}')
    #         plt.axis('off')

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize image
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    if plot == True:
        fig, ax = plt.subplots()
        ax.imshow(resized)
        plt.title(f'Shape of the resized image: {resized.shape}')
        plt.axis('off')

    # Plot the defect of the resize image
    yleft, yright = int(math.ceil(min(y1, y2, y3, y4) / 2) - 2), int(math.ceil(max(y1, y2, y3, y4) / 2) + 2)
    xleft, xright = int(math.ceil(min(x1, x2, x3, x4) / 2) - 2), int(math.ceil(max(x1, x2, x3, x4) / 2) + 2)
    img_defect_resized = cv.cvtColor(resized, cv.IMREAD_GRAYSCALE)[yleft: yright, xleft: xright]
    if plot == True:
        fig, ax = plt.subplots()
        ax.imshow(img_defect_resized)
        plt.title(f'Shape of the defect resized: {img_defect_resized.shape}')
        plt.axis('off')

    return img_defect_resized, [yleft, yright, xleft, xright]


# Crop the ultrasound array and the backwall and scale it to 0-1

def DAS_image_resize(image_das_fmc):
    if image_das_fmc.shape == (875, 1167):
        cropped_image = image_das_fmc[125:575, 155:1055] / 255
    else:
        cropped_image = image_das_fmc[85:430, 116:806] / 255
    scale_percent = 50
    width = int(240 * scale_percent / 100)
    height = int(120 * scale_percent / 100)
    dim = (width, height)
    das_img_resize = cv.resize(cropped_image, dim, interpolation=cv.INTER_AREA)

    return das_img_resize


if __name__ == "__main__":
    # print('hello')
    # filename = model_namer(dimension=2, num_sample=1000, sub_sample=5, fmc_scaler=1, img_resize=1, remark='no', version=90, epochs=100)
    # filename = filename + '.pkl'
    # print(filename)
    # model_namer_description(filename)
    # X, y = load_training_data(num_sample=3500, x_dimension=3, img_resize_factor=50, shrinkx=True, stack=False)
    # model testing
    start = time.time()

    # filename = '3D_2000_o7127128-2.pkl' # Old Best 3d model
    # filename = 'Best_models/3D_2000_o7121864.pkl' # Best 3d model
    filename = 'Best_models/2D_2450_o7165279-1.pkl'  # Best 2d model

    # info = model_namer_description(filename)
    model = load_ml_model(filename)
    model.summary()

    # Computed the time taken
    end = time.time()
    print(end - start)
