from tensorflow.keras.models import Model,load_model
import numpy as np
import cv2
from django.utils.translation import gettext_lazy as _

from apps.featureextraction.SOM.CNN.ConfigCNN import Config

from tqdm import tqdm

class CompDetCNN:
    def __init__(self, model_path, model_classes, model_shape):
        cfg = Config(model_path, model_classes, model_shape)
        self.model = load_model(cfg.CNN_PATH)
        self.class_map = cfg.element_class
        self.image_shape = cfg.image_shape

    def preprocess_img(self, image):
        image = cv2.resize(image, self.image_shape[:2])
        x = (image / 255).astype('float32')
        x = np.array([x])
        return x

    def predict(self, imgs):
        if self.model is None:
            print(_("*** No model loaded ***"))
            return
        result = []
        for i in range(len(imgs)):
            try:
                X = self.preprocess_img(imgs[i])
                Y = self.class_map[np.argmax(self.model.predict(X, verbose=0))]
                result.append(Y)
            except Exception as e:
                result.append('unknown')
        return result
            