import py_training_module.Models as M
import py_training_module.DataLoader as DL
import cv2
import numpy as np
import tensorflow as tf

RESOURCE_PATH = 'resources'
MODELS_PATH = RESOURCE_PATH + '/models'
TEST_PATH = RESOURCE_PATH + '/face_data/test'


test_x, test_y = DL.load_data(TEST_PATH)
model = M.load_model(MODELS_PATH + '/model_1.h5')

counter = 0
for x, y in zip(test_x, test_y):
    x = tf.expand_dims(x, axis=0)
    x = tf.cast(x, tf.float32)
    predictions = model.predict(x)
    predicted_y = np.where(predictions == np.amax(predictions))[1][0]
    print(f'It should be: [{y}] -> [{predicted_y}]')

# Just to be sure
model.evaluate(test_x, test_y)
