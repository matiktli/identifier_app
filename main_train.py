import py_training_module.Learner as L
import py_training_module.Models as M
import py_training_module.DataLoader as DL
import cv2
import numpy as np

RESOURCE_PATH = 'resources'
TRAIN_PATH = RESOURCE_PATH + '/face_data/train'
TEST_PATH = RESOURCE_PATH + '/face_data/test'
MODELS_PATH = RESOURCE_PATH + '/models'

MODELS_PATH = MODELS_PATH + '/model_4_gray.h5'


# Main func of file
def main_train(train_folder_path=TRAIN_PATH, test_folder_path=TEST_PATH, model_path=MODELS_PATH, possible_choices=9, epochs=20, steps=8):
    train_x, train_y = DL.load_data(train_folder_path)
    test_x, test_y = DL.load_data(test_folder_path)

    print(len(train_x), '-', len(train_y), '-', len(test_x), '-', len(test_y))
    print(f'\nInput shape: {train_x.shape}')

    model = M.BASE_MODEL(possible_choices, (250, 250, 3))
    learner = L.Learner()
    learner.attach_model(model)
    learner.attach_train_data(train_x, train_y)
    learner.attach_test_data(test_x, test_y)
    learner.train_model(epochs=10, steps_per_epoch=16)
    learner.evaluate_model()
    learner.save_model(model_path)


main_train()
