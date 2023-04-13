from os.path import join as pjoin
import os


class Config:

    def __init__(self, model_path, model_classes, model_shape):
        # setting CNN (graphic elements) model
        self.image_shape = model_shape
        self.CNN_PATH = model_path
        self.element_class = model_classes
        self.class_number = len(self.element_class)
