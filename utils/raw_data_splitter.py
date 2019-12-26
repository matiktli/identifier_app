import os
import cv2
import numpy as np
from id_assigner import get_config_json
import random

RESOURCE_PATH = '../resources'

RAW_PATH = RESOURCE_PATH + '/face_data/raw'
TEST_PATH = RESOURCE_PATH + '/face_data/test'
TRAIN_PATH = RESOURCE_PATH + '/face_data/train'
CONFIG_PATH = RESOURCE_PATH + '/config/config_ids.JSON'


def __load_single_student_images(path):
    result = []
    for f in os.listdir(path):
        img_data = cv2.imread(path + '/' + f)
        result.append(img_data)
    return result


def load_all_students_images(path, student_data):
    result = {}
    print(f'\n--Loading students data:')
    for student in student_data:
        student_folder_path = path + '/' + student['name']

        student_photos_data = __load_single_student_images(student_folder_path)
        result[student['id']] = student_photos_data
        print(
            f'Student: {student["name"]} with id: {student["id"]} - {len(student_photos_data)} photos.')
    return result


def __save_images_batch_to_folder(image_batch, student_id, path):
    counter = 0
    for img_data in image_batch:
        img_name = str(student_id) + '_' + str(counter) + '.png'
        cv2.imwrite(path + '/' + img_name, img_data)
        counter += 1


def split_images_for_train_and_test(data, train_path, test_path, vector=0.8):
    print(f'\n--Splitting students data:')
    whole_size = len(data[list(data.keys())[0]])
    train_size = int(whole_size * vector)
    test_size = whole_size - train_size
    print(
        f'Train size per student: {train_size}\nTest size per student: {test_size}')

    # Shuffle photos to get rid of simillar postures
    for student_id in data:
        random.shuffle(data[student_id])
        student_train_imgs = data[student_id][:train_size]
        student_test_imgs = data[student_id][-test_size:]
        __save_images_batch_to_folder(
            student_train_imgs, student_id, train_path)
        __save_images_batch_to_folder(
            student_test_imgs, student_id, test_path)


STUDENTS = get_config_json(CONFIG_PATH)['students']

data = load_all_students_images(RAW_PATH, STUDENTS)

split_images_for_train_and_test(data, TRAIN_PATH, TEST_PATH, 0.8)
