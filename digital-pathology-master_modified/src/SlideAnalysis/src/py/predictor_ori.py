import os
import sys

import numpy as np
import warnings

#stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')
from keras import backend as K
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing import image
#sys.stderr = stderr

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# dimensions of our images
img_width, img_height = 50, 50
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

THRESHOLD = 0.05


class Predict:


    def __init__(self):
        # build model

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        # load model
        model.load_weights(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model', 'model.h5'))
        self.model = model

    def predict_img_class(self, img):
        result = self.model.predict(img)
        if result < THRESHOLD:
            classification = "ecell"
        else:
            classification = "non-ecell"
        return classification

    def predict_img_class_folder(self, path):
        classifications = {}

        for img in os.listdir(path):
            imgpath = path + "/" + img
            test_image = image.load_img(imgpath, target_size=(img_width, img_height))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = self.model.predict(test_image)
            if result < THRESHOLD:
                classification = "ecell"

            else:
                classification = "non-ecell"

            classifications[img] = classification

        return classifications

    
    #image_list - the list of 50 by 50 pixel images as nd.arrays()
    # center list - list of tuples of int, the centers (x,y) of each image
    def filter_positive_classifications(self, image_list, center_list):
        ecell_center_list = []
        for i, img in enumerate(image_list):
            if img.shape != (50,50,3):
                continue
            test_image = img.astype(np.float32)
            test_image = np.expand_dims(test_image, axis=0)
            result = self.model.predict(test_image)
            if result < THRESHOLD:
                ecell_center_list.append(center_list[i])
        return ecell_center_list


if __name__ == '__main__':
    p = Predict()
    classes = p.predict_img_class_folder("cells")
    print(classes)
