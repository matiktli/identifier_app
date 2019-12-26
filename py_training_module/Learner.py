from .Models import *


class Learner():

    def __init__(self):
        self.model = None

    def attach_model(self, model):
        self.model = model

    def train_model(self, train_x, train_y, epochs=20, steps_per_epoch=60):
        self.model.fit(train_x, train_y, epochs=epochs,
                       steps_per_epoch=steps_per_epoch)

    def evaluate_model(self, test_x, test_y):
        self.model.evaluate(test_x, test_y)
