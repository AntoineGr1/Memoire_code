#!/usr/bin/env python
# coding: utf-8

# # Import

# In[1]:


import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from time import time

from tensorflow.keras.datasets import cifar10


# # DenseNet 121

# In[2]:


def densenet(input_shape, n_classes, filters = 32):
    
    #batch norm + relu + conv
    def bn_rl_conv(x,filters,kernel=1,strides=1):
        
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel, strides=strides,padding = 'same')(x)
        return x
    
    def dense_block(x, repetition):
        
        for _ in range(repetition):
            y = bn_rl_conv(x, 4*filters)
            y = bn_rl_conv(y, filters, 3)
            x = concatenate([y,x])
        return x
        
    def transition_layer(x):
        
        x = bn_rl_conv(x, K.int_shape(x)[-1] //2 )
        x = AvgPool2D(2, strides = 2, padding = 'same')(x)
        return x
    
    input = Input (input_shape)
    x = Conv2D(64, 7, strides = 2, padding = 'same')(input)
    x = MaxPool2D(3, strides = 2, padding = 'same')(x)
    
    for repetition in [6,12,24,16]:
        
        d = dense_block(x, repetition)
        x = transition_layer(d)
    x = GlobalAveragePooling2D()(d)
    output = Dense(n_classes, activation = 'softmax')(x)
    
    model = Model(input, output)
    return model


# In[3]:


input_shape = 32, 32, 3
n_classes = 10


# In[4]:


model = densenet(input_shape,n_classes)
model.compile( loss=tf.keras.losses.sparse_categorical_crossentropy ,optimizer=tf.keras.optimizers.Adam( lr=0.0001 ) ,metrics=[ 'acc' ])
model.summary()
print(len(model.layers))


# # Data

# In[5]:


(train_x, train_y), (test_x, test_y) = cifar10.load_data()

# normalize to range 0-1
train_x = train_x / 255.0
test_x = test_x / 255.0

val_x = train_x[:5000]
val_y = train_y[:5000]


# # Train model

# In[6]:


es = tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=1, restore_best_weights=True, patience=1)
list_cb = [es]


# In[7]:



start = time()
model.fit( train_x , train_y , epochs=50, batch_size=1024, validation_split=0.3) #, callbacks=list_cb)
training_time = time()-start


# # Evaluate

# In[8]:


print(model.evaluate(test_x, test_y))


# In[9]:


print(model.evaluate(train_x, train_y))


# In[10]:


print(training_time)

