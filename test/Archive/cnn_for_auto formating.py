#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().system('ls ./data')


# In[4]:


#get_ipython().run_line_magic('load_ext', 'klab-autotime')


# In[5]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import glob, os, random

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# In[6]:

nb_train_samples = 2000
nb_validation_samples = 2000
epochs = 100  #
batch_size = 32



class FormatPredictor:

    def __init__(self):
        
        
        model = Sequential()
        # -----------------------------------------------------
        # Input layer: first layer
        # first conv layer and max pool layer
        model.add(Conv2D(filters=32,   # 32 filter
                 kernel_size=(3, 3),   # kernel size 3 x 3
                 input_shape=input_shape,  # input shape
                 activation='relu'))  # 'relu' activation
        model.add(MaxPooling2D(pool_size=(2, 2)))  # pool kernel size 2 x 2

        # ----------------------------------------------------
        #    hidden layer
        # Add second Conv and optional pooling layer
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Add third Conv and optional pooling layer
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten layer
        model.add(Flatten())  #
        model.add(Dense(units=64, activation='relu'))  # full connect layer 64 neurons
        model.add(Dropout(0.3))  # Prevent overfitting

        # ---------------------------------------------------
        # Output layer
        model.add(Dense(units=10, activation='sigmoid'))  # activation function

        model.summary()


        # In[19]:


        model.compile(loss='categorical_crossentropy',  #
              optimizer='rmsprop',  #
              metrics=['accuracy'])

    def train(self,train_data_dir,validation_data_dir,img_width,img_height):
        train_datagen = ImageDataGenerator(rescale=1. / 255,  # convert to [0,1]
                                               shear_range=0.2,  # shear range
                                               zoom_range=0.2,  # random zoom
                                               horizontal_flip=True,  # random horizontal flip
                                               vertical_flip = True #random vertical flip
                                               )
        train_generator = train_datagen.flow_from_directory(train_data_dir,  # training data directory
                                                    target_size=(img_width, img_height),  # size of image
                                                    batch_size=batch_size,  # batch size
                                                    class_mode='categorical'  # binary
                                                    )
        test_datagen = ImageDataGenerator(rescale=1. / 255,
                                                  shear_range=0.2,  # shear range
                                                  zoom_range=0.2,  # random zoom
                                                  horizontal_flip=True,# random horizontal flip
                                                  vertical_flip = True) #random vertical flip
        validation_generator = test_datagen.flow_from_directory(validation_data_dir,  # validation path
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical'  # binary
                                                        )
        history = model.fit_generator(train_generator,
                                              steps_per_epoch=nb_train_samples // batch_size,
                                              epochs=epochs,
                                              validation_data=validation_generator,
                                              validation_steps=nb_validation_samples // batch_size
                                              )
        train_acc = history.history['acc']
        test_acc = history.history['val_acc']
        epoch_counts = range(1, len(train_acc)+1)
        plt.plot(epoch_counts, train_acc, 'r--', marker='^')
        plt.plot(epoch_counts, test_acc, linestyle='-', marker='o', color='y')
        plt.title('accuracy condition')
        plt.legend(['train_acc', 'test_acc'])
        plt.xlabel('epochs')
        plt.ylabel('acc')

        model.save("model.h5")

