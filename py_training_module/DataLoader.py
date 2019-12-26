import os
import cv2
from .__Imports import *


def load_data(path):
    data_x, data_y = [], []
    for image_file in os.listdir(path):

        img_data = cv2.imread(path + '/' + image_file, cv2.IMREAD_COLOR)

        data_x.append(img_data)

        img_label = int(image_file[:1])
        img_label = img_label
        data_y.append(img_label)

    return np.array(data_x), np.array(data_y)
