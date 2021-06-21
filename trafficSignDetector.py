# -*- coding: utf-8 -*-

"""Detection and Prediction of Traffic Sign"""
"""MAIN MODEL MAKING FILE"""
# Importing the Packages for Image Processing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

class signDetector:
    def build(width,height,depth,classes):
        
        #Initializing the model with a given input shape
        cnnmodel = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        '''Concepts used:
            Convolution -> Activation -> Batch Normalization-> Max Pooling
            '''
        cnnmodel.add(Conv2D(8, (5,5), padding="same", input_shape = inputShape))
        cnnmodel.add(Activation("relu"))
        cnnmodel.add(BatchNormalization(axis = chanDim))
        cnnmodel.add(MaxPooling2D(pool_size = (2,2)))
        
        #Adding Layer 1
        cnnmodel.add(Conv2D(16, (3, 3), padding="same"))
        cnnmodel.add(Activation("relu"))
        cnnmodel.add(BatchNormalization(axis=chanDim))
        cnnmodel.add(Conv2D(16, (3, 3), padding="same"))
        cnnmodel.add(Activation("relu"))
        cnnmodel.add(BatchNormalization(axis=chanDim))
        cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
        
        #Adding Layer 2
        cnnmodel.add(Conv2D(32, (3, 3), padding="same"))
        cnnmodel.add(Activation("relu"))
        cnnmodel.add(BatchNormalization(axis=chanDim))
        cnnmodel.add(Conv2D(32, (3, 3), padding="same"))
        cnnmodel.add(Activation("relu"))
        cnnmodel.add(BatchNormalization(axis=chanDim))
        cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
        
        #Flattening Layers
        cnnmodel.add(Flatten())
        cnnmodel.add(Dense(128))
        cnnmodel.add(Activation("relu"))
        cnnmodel.add(BatchNormalization())
        cnnmodel.add(Dropout(0.5))
		
        cnnmodel.add(Flatten())
        cnnmodel.add(Dense(128))
        cnnmodel.add(Activation("relu"))
        cnnmodel.add(BatchNormalization())
        cnnmodel.add(Dropout(0.5))
		
        #Softmax classifier
        cnnmodel.add(Dense(classes))
        cnnmodel.add(Activation("softmax"))
        
        return cnnmodel




