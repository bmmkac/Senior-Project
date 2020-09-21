from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model

import numpy as np
import os

import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing import image
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow


import keras.backend as K


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# dimensions of our images
img_width, img_height = 50, 50
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

THRESHOLD = 0.5


K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
   
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
       
    return X

def convolutional_block(X, f, filters, stage, block, s = 2):    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


def ResNet50(input_shape=(64, 64, 3), classes=6):
# Define the input as a tensor with shape input_shape'

    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2,2), name="avg_pool")(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


class Predict:


    def __init__(self):
        # build model
        model = self.resnet_v1((50,50,3
                                ))
        model.add(Lambda(lambda x: x * 1./255., input_shape=input_shape, output_shape=input_shape))

        model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
        
        # load model
        self.model = model

    def lr_schedule(self,epoch):
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr



    def resnet_layer(self,inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
        conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))
        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    def resnet_v1(self,input_shape):
        num_filters = 16
        num_res_blocks = int((20 - 2) / 6)
    
        inputs = Input(shape=input_shape)
        x = self.resnet_layer(inputs=inputs)
        y = x
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                
                if stack > 0 and res_block == 0:
                    y = self.resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                    y = self.resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
                if stack > 0 and res_block == 0:  
                    x = self.resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(1,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)
    
        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    
    def fitting_model(self, train_set, train_labels, test_set, test_labels, cross_val_num, epoch):
        history = self.model.fit(train_set, train_labels, epochs=epoch,validation_data=(test_set, test_labels))
        
        self.model.save(str(cross_val_num)+'_model.h5') 
 

        
        
    def S_fold(self, s, eo_path, non_eo_path, epoch):
        #Break up to 10 group, select 1 for testing and concatenate the rest 9 for training - Shuyi
        data_x, data_y = self.Convert_helper(eo_path, non_eo_path)
        #data_x = data[:, :data.shape[1]-1]
        #data_y = data[:, [data.shape[1]-1]]
        data_y.reshape((data_y.shape[0],1))
        data_x_split = []
        data_y_split = []
        if data_x.shape[0]%s==0:
            data_x_split = np.split(data_x,s)
            data_y_split = np.split(data_y,s)
        else:
            data_x_split = self.split_helper(data_x, s)
            data_y_split = self.split_helper(data_y, s)
        
        for i in range(0,s):
            c_data_x_test = data_x_split[i]
            c_data_y_test = data_y_split[i]
            c_data_x_train_list = data_x_split[:i] + data_x_split[i+1:]
            c_data_y_train_list = data_y_split[:i] + data_y_split[i+1:]
            c_data_x_train = self.concatination_helper(c_data_x_train_list)
            c_data_y_train = self.concatination_helper(c_data_y_train_list)
            self.fitting_model(c_data_x_train,c_data_y_train,c_data_x_test,c_data_y_test,i, epoch)
        
    def concatination_helper(self, data_list):
        #concatinate all group of matrix vertically in data_list - Shuyi
        result = data_list[0]
        for i in range(1, len(data_list)):
            result = np.concatenate((result, data_list[i]),axis=0)
        return result 
    
    def split_helper(self, data, s):
        #split the dataset to s groups, this will only be called when we can't evenly split the dataset - Shuyi
        result = []
        s_size = data.shape[0]//s
        #print(s_size)
        for i in range(0,s):
            if i!= s-1:
                current = data[i*s_size:(i+1)*s_size]

                result.append(current)

            else:
                current = data[i*s_size:]
                result.append(current)
        return result

    
    
    def Convert_helper(self, eo_path, non_eo_path): # - By Joe Amon
        data_x = []  # images (4000,50,50,3)
        data_y = []  # labels (4000,)
        total_files = 2000; # Get 2000 random files from both dirs

        # for files in eo_path dir
        eo_folder = os.listdir(eo_path)
        np.random.shuffle(eo_folder);
        for i, img in enumerate(eo_folder):
            if i == total_files:
                break;
            img_path = eo_path + "/" + img
            eo_image = image.load_img(img_path, target_size=(img_width, img_height))
            eo_image = image.img_to_array(eo_image)
            data_x.append(eo_image)
            data_y.append(1)

        # for files in non_eo_path dir
        non_eo_folder = os.listdir(non_eo_path)
        np.random.shuffle(eo_folder);
        for i, img in enumerate(non_eo_folder):
            if i == total_files:
                break;
            img_path = non_eo_path + "/" + img
            non_eo_image = image.load_img(img_path, target_size=(img_width, img_height))
            non_eo_image = image.img_to_array(non_eo_image)
            data_x.append(non_eo_image)
            data_y.append(0)

        # randomize rows of data_x and data_y the same way
        p = np.random.permutation(len(data_y))

        return np.array(data_x)[p], np.array(data_y)[p]
    
    
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

    def print_model_structure(self):
        self.model.summary()
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
    
    



    
    p.S_fold(2, './cellData/eosinophil', './cellData/non-eosinophil',20)
    ##print("S-fold ends")
    print("accuracy test start")
    classes1 = p.predict_img_class_folder("./cellData/eosinophil")
    classes2 = p.predict_img_class_folder("./cellData/non-eosinophil")
    #print(classes)
    correct_rate_ecell = len([k for k,v in classes1.items() if v == 'ecell'])/len(classes1.keys())
    print('Rate of correctly classify eosinophil cell: '+str(correct_rate_ecell))
    correct_rate_nonecell = len([k for k,v in classes2.items() if v == 'non-ecell'])/len(classes2.keys())
    print('Rate of correctly classify non-eosinophil cell: '+str(correct_rate_nonecell))


    for i in [x * 0.01 for x in range(1,100,5)]:
        THRESHOLD = i
        classes1 = p.predict_img_class_folder("./cellData/eosinophil")
        classes2 = p.predict_img_class_folder("./cellData/non-eosinophil")
        #print(classes)
        TP = len([k for k,v in classes1.items() if v == 'ecell'])
        FN = len([k for k,v in classes1.items() if v == 'non-ecell'])
        FP = len([k for k,v in classes2.items() if v == 'ecell'])
        TN = len([k for k,v in classes2.items() if v == 'non-ecell'])
        total_precision = []
        total_recall = []
        total_f_measure
        precision = None
        recall = None
        f_measure = None
        if TP+FP!=0:
            precision = TP/(TP+FP)
            total_precision.append(precision)
            p_x.append(i)
            
               
            
        if TP+FN !=0:
            recall = TP/(TP+FN)
            total_recall.append(recall)
            r_x.append(i)
            
                
        if precision!=None and recall != None:
            
            f_measure = (2*precision*recall)/(precision+recall)
            total_f_measure.append(f_measure)
            f_x.append(i)
        total_precision.append(precision)
        total_recall.append(recall)
      

    
    
    plt_dynamic(p_x, total_precision, ax,['r'])
