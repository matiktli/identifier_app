from .Models import *


class Learner():

    def __init__(self):
        pass

    def attach_model(self, model):
        self.model = model

    def attach_train_data(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def attach_test_data(self, test_x, test_y):
        self.test_x = test_x
        self.test_y = test_y

    def train_model(self, epochs=20, steps_per_epoch=50):
        self.model.fit(self.train_x, self.train_y, epochs=epochs,
                       steps_per_epoch=steps_per_epoch)

    def test_model(self, test_x, test_y):
        pass

    def evaluate_model(self):
        self.model.evaluate(self.test_x, self.test_y)

    def save_model(self, path):
        self.model.save(path)
