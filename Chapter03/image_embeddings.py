from data_preparation import train_image_paths, test_image_paths
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model
import pickle
import os


class ImageModel:
    def __init__(self):
        vgg_model = VGG16(weights='imagenet', include_top=True)
        self.model = Model(input=vgg_model.input,
                           output=vgg_model.get_layer('fc2').output)

    @staticmethod
    def load_preprocess_image(image_path):
        image_array = image.load_img(image_path, target_size=(224, 224))
        image_array = image.img_to_array(image_array)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        return image_array

    def extract_feature_from_image_path(self, image_path):
        image_array = self.load_preprocess_image(image_path)
        features = self.model.predict(image_array)
        return features.reshape((4096, 1))

    def extract_feature_from_image_paths(self, work_dir, image_names):
        features = []
        for image_name in image_names:
            image_path = os.path.join(work_dir, image_name)
            feature = self.extract_feature_from_image_path(image_path)
            features.append(feature)
        return features

    def extract_features_and_save(self, work_dir, image_names, file_name):
        features = self.extract_feature_from_image_paths(work_dir, image_names)
        with open(file_name, 'wb') as p:
            pickle.dump(features, p)


I = ImageModel()
I.extract_features_and_save(b'Flicker8k_Dataset',train_image_paths, 'train_image_features.p')
I.extract_features_and_save(b'Flicker8k_Dataset',test_image_paths, 'test_image_features.p')


