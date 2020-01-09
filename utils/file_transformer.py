import os
import json
from config_utils import *
import cv2
import numpy as np

RAW_DATA_PATH = '../resources/face_data/raw'


def img_transform_blur(img, blur=(5, 5)):
    return cv2.blur(img, blur)


# THIS IS TRICKY WERIFY FIRST
def img_transform_rotate(img, angle=0):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (h, w))


def img_transform_flip_horizontal(img):
    return cv2.flip(img, 1)


def img_transform_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def __transform_single_student_folder(student_folder_path):
    for img_file_name in os.listdir(student_folder_path):
        img = cv2.imread(student_folder_path + '/' + img_file_name)

        img_blured = img_transform_blur(img)
        img_blured_name = img_file_name[:-4] + '_blured.png'
        cv2.imwrite(student_folder_path + '/' + img_blured_name, img_blured)

        # -20, -15, -10, -5, 5, 10, 15, 20
        for angle in [-20, -15, -10, -5, 5, 10, 15, 20]:
            img_rotated = img_transform_rotate(img, angle)
            img_rotated_name = img_file_name[:-
                                             4] + f'_rotated_{str(angle)}.png'
            cv2.imwrite(student_folder_path + '/' +
                        img_rotated_name, img_rotated)

        img_flipped = img_transform_flip_horizontal(img)
        img_flipped_name = img_file_name[:-4] + '_flipped.png'
        cv2.imwrite(student_folder_path + '/' + img_flipped_name, img_flipped)

        img_gray = img_transform_gray(img)
        img_gray_name = img_file_name[:-4] + '_gray.png'
        cv2.imwrite(student_folder_path + '/' + img_gray_name, img_gray)


def main_transform_raw_data(raw_folder_path=RAW_DATA_PATH):
    for dir_name in os.listdir(raw_folder_path):
        student_folder_path = raw_folder_path + '/' + dir_name
        if os.path.isdir(student_folder_path) and dir_name == 'MateuszKitlinski':
            __transform_single_student_folder(student_folder_path)


main_transform_raw_data(RAW_DATA_PATH)
