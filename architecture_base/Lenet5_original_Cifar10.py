#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.layers import Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Dropout,BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
from time import time

from tensorflow.keras.datasets import cifar10


# Loading the dataset

# In[2]:


# setting class names
class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#loading the dataset
(train_x,train_y),(test_x,test_y)=cifar10.load_data()


# Normalizing the Images

# In[3]:


train_x=train_x/255.0
train_x.shape


# In[4]:


test_x=test_x/255.0
test_x.shape


# Randomly Checking a image

# In[5]:


plt.imshow(test_x[215])


# In[6]:


val_x = train_x[:5000] 
val_y = train_y[:5000]


# In[7]:


# Building a Convolutional Neural Network
def getmodel(input_shape):
    # Input 
    X_input = Input(input_shape)
    X = Conv2D(filters=32,kernel_size=3,padding="same", activation="relu")(X_input)
    X = MaxPool2D(pool_size=2,strides=2,padding='valid')(X)
    X = Conv2D(filters=64,kernel_size=3,padding="same", activation="relu")(X)
    X = MaxPool2D(pool_size=2,strides=2,padding='valid')(X)
    X = Flatten()(X)
    X = Dense(units=128,activation='relu')(X)
    X = Dense(units=84,activation='relu')(X)
    X = Dense(units=10,activation='softmax')(X)
    
    model = Model(inputs=X_input, outputs=X, name='CNN')
    
    model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    print(model.summary())
    return model


# In[8]:


CNN_model = getmodel(train_x[0].shape)


# In[9]:


es = tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=1, restore_best_weights=True, patience=1)
list_cb = [es]


# In[10]:


start = time()
CNN_model.fit( train_x , train_y , epochs=50, batch_size=1024, validation_split=0.3, callbacks=list_cb)
training_time = time()-start


# In[11]:


print(CNN_model.evaluate(test_x, test_y))


# In[12]:


print(CNN_model.evaluate(train_x, train_y))


# In[13]:


print(training_time)


# In[ ]:




