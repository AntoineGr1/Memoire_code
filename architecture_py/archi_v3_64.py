
import numpy as np
import os
from keras import backend as K
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPool2D, Concatenate, Dropout
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import sys
import traceback
import csv
from time import time


type_archi = 'RESNET'
epsilon = 0.0
dropout_rate = 0.8
axis = 3
compress_factor = 0.5


# load dataset
(train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()

# normalize to range 0-1
train_x = train_x / 255.0
test_x = test_x / 255.0

val_x = train_x[:5000]
val_y = train_y[:5000]
    


# init training time
training_time = 0
# init result test/train
test_result_loss = ""
test_result_acc = ""

train_result_loss = ""
train_result_acc = ""

nb_layers = "not build"


def id_block(X, f, filters, activation):

    X_shortcut = X

    X = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    if epsilon != 0:
        X = BatchNormalization(epsilon = epsilon, axis=axis)(X)
    X = Activation(activation)(X)


    X = Conv2D(filters=filters, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    if epsilon != 0:
        X = BatchNormalization(epsilon = epsilon, axis=axis)(X)

    X = Add()([X, X_shortcut])# SKIP Connection
    X = Activation(activation)(X)

    return X
    
def conv_block(X, f, filters, activation, s=2):

    X_shortcut = X

    X = Conv2D(filters=filters, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    if epsilon != 0:
        X = BatchNormalization(epsilon = epsilon, axis=axis)(X)
    X = Activation(activation)(X)

    X = Conv2D(filters=filters, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    if epsilon != 0:
        X = BatchNormalization(epsilon = epsilon, axis=axis)(X)

    X_shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    if epsilon != 0:
        X_shortcut = BatchNormalization(epsilon = epsilon, axis=axis)(X_shortcut)


    X = Add()([X, X_shortcut])
    X = Activation(activation)(X)

    return X
    
try:
    def getModel():
        X_input = X = Input([32, 32, 3])
        X = conv_block(X, 2, 18, 'relu', 2)
        X = id_block(X, 4, 18, 'relu')
        X = id_block(X, 2, 18, 'selu')
        X = MaxPooling2D(pool_size=3, strides=1, padding='same')(X)
        X = Conv2D(36, kernel_size=7, strides=2, activation='selu', padding='valid')(X)
        X = id_block(X, 4, 36, 'relu')
        X = conv_block(X, 2, 72, 'selu', 1)
        X = id_block(X, 5, 72, 'relu')
        X = id_block(X, 6, 72, 'tanh')
        X = GlobalMaxPooling2D()(X)
        X = Dense(10, activation='softmax')(X)
        model = Model(inputs=X_input, outputs=X)
        return model

    model = getModel()
    plot_model(model, show_shapes=True, to_file="../architecture_img/archi_v3_64.png")
    model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    start = time()
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=1)
    list_cb = [es]
    history = model.fit(train_x, train_y, epochs=50, validation_split=0.1, callbacks=list_cb)
    training_time = time()-start
    print(model.evaluate(test_x, test_y))

    log_file = open("../architecture_log/archi_v3_64.log" , "w")
    
    # save test result
    log_file.write('test result : ' + str(model.evaluate(test_x, test_y)))
    test_result_loss = model.evaluate(test_x, test_y)[0]
    test_result_acc = model.evaluate(test_x, test_y)[1]
    
    # save train result
    log_file.write('train result : ' + str(model.evaluate(test_x, test_y)))
    log_file.write('History train result : ' + str(history.history))
    train_result_loss = model.evaluate(train_x, train_y)[0]
    train_result_acc = model.evaluate(train_x, train_y)[1]
    
    print('OK: file ../architecture_log/archi_v3_64.log has been create')
    
    nb_layers = len(model.layers)
    log_file.close()
except:
    print('error: file ../architecture_log/archi_v3_64_error.log has been create')
    error_file = open("../architecture_log/archi_v3_64_error.log" , "w")
    traceback.print_exc(file=error_file)
    result_loss = "Error"
    result_acc = "Error"
    error_file.close()
finally:
    file = open('../architecture_results_v3.csv', 'a', newline ='')
    with file: 

        # identifying header   
        header = ['file_name', 'training_time(s)', 'test_result_loss', 'test_result_acc', 'train_result_acc', 'train_result_loss', 'nb_layers', 'epochs', 'type_archi'] 
        writer = csv.DictWriter(file, fieldnames = header) 
      
        # writing data row-wise into the csv file 
        # writer.writeheader() 
        writer.writerow({'file_name' : 'archi_v3_64',  
                         'training_time(s)': training_time,  
                         'test_result_loss': test_result_loss,
                         'test_result_acc': test_result_acc,
                         'train_result_acc': train_result_acc,
                         'train_result_loss': train_result_loss,
                         'nb_layers': nb_layers,
                         'epochs' : len(history.history['loss']),
                         'type_archi': type_archi}) 
        print('add line into architecture_results_v3.csv')
    file.close()
    