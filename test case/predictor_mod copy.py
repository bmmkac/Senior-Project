import os
import sys


import numpy as np
import warnings
import tensorflow as tf
#stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')
from keras import backend as K
from keras.utils import plot_model

from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing import image
#sys.stderr = stderr
my_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + "/../digital-pathology-master/src/SlideAnalysis/src/py/model/model.h5"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logdir = 'logs/predictor/'

img_width, img_height = 50, 50
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

THRESHOLD = 0.95


def predict_model():
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
    model.load_weights(my_path)
    
    return model


model = predict_model()
plot_model(model, to_file='model.png')
