from .__Imports import *


def BASE_MODEL(result_options, input_shape=(250, 250, 3)):
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu',
                            input_shape=input_shape, padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(result_options, activation='softmax'))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer='nadam', metrics=['accuracy'])
    return model


def YOLOv3(result_options, input_shape=(250, 250, 3)):
    return None


def load_model(path):
    return tf.keras.models.load_model(path)
