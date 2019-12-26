import py_training_module.Learner as L
import py_training_module.Models as M
import py_training_module.DataLoader as DL
import cv2
import numpy as np

RESOURCE_PATH = 'resources'
TRAIN_PATH = RESOURCE_PATH + '/face_data/train'
TEST_PATH = RESOURCE_PATH + '/face_data/test'
MODELS_PATH = RESOURCE_PATH + '/models'

train_x, train_y = DL.load_data(TRAIN_PATH)
test_x, test_y = DL.load_data(TEST_PATH)

print(len(train_x), '-', len(train_y), '-', len(test_x), '-', len(test_y))
print(f'\nInput shape: {train_x.shape}')

model = M.BASE_MODEL(8, (250, 250, 3))
learner = L.Learner()
learner.attach_model(model)
learner.attach_train_data(train_x, train_y)
learner.attach_test_data(test_x, test_y)
learner.train_model(epochs=20, steps_per_epoch=10)
learner.evaluate_model()
learner.save_model(MODELS_PATH, 'model_1.h5')
