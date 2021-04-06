
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
import sys
import traceback
import csv
from time import time
(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()

# normaliser les pixel 0-255 -> 0-1
train_x = train_x / 255.0
test_x = test_x / 255.0

train_x = tf.expand_dims(train_x, 3)
test_x = tf.expand_dims(test_x, 3)

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

try:
    
    def id_block(X, f, filters):
   
        X_shortcut = X

        X = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
        # X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)


        X = Conv2D(filters=filters, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
        # X = BatchNormalization(axis=3)(X)

        X = Add()([X, X_shortcut])# SKIP Connection
        X = Activation('relu')(X)

        return X
    
    def conv_block(X, f, filters, s=2):
    
        X_shortcut = X

        X = Conv2D(filters=filters, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
        # X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=filters, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
        # X = BatchNormalization(axis=3)(X)

        X_shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        # X_shortcut = BatchNormalization(axis=3)(X_shortcut)


        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X
    def ResNet():
        X_input = X = Input([28, 28, 1])
        X = Conv2D(18, kernel_size=7, strides=2, activation='selu', padding='valid')(X)
        X = AveragePooling2D(pool_size=3, strides=2, padding='same')(X)
        X = conv_block(X, 3, 36, 2)
        X = id_block(X, 3, 36)
        X = conv_block(X, 3, 72, 2)
        model = Model(inputs=X_input, outputs=X, name='ResNet18')
        return model

    Input = ResNet()
    head_model = Input.output
    head_model = Flatten()(head_model)
    head_model = Dense(85, activation='selu')(head_model)
    head_model = Dense(40, activation='relu')(head_model)
    head_model = Dense(10, activation='softmax')(head_model)
    model = Model(inputs=Input.input, outputs=head_model)
    plot_model(model, show_shapes=True, to_file="../architecture_img/archi_resnet_5.png")
    model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    start = time()
    model.fit(train_x, train_y, epochs=5, validation_data=(val_x, val_y))
    training_time = time()-start
    print(model.evaluate(test_x, test_y))

    log_file = open("../architecture_log/archi_resnet_5.log" , "w")
    
    # save test result
    log_file.write('test result : ' + str(model.evaluate(test_x, test_y)))
    test_result_loss = model.evaluate(test_x, test_y)[0]
    test_result_acc = model.evaluate(test_x, test_y)[1]
    
    # save train result
    log_file.write('train result : ' + str(model.evaluate(test_x, test_y)))
    train_result_loss = model.evaluate(train_x, train_y)[0]
    train_result_acc = model.evaluate(train_x, train_y)[1]
    
    print('OK: file ../architecture_log/archi_resnet_5.log has been create')
    
    nb_layers = len(model.layers)
    log_file.close()
except:
    print('error: file ../architecture_log/archi_resnet_5_error.log has been create')
    error_file = open("../architecture_log/archi_resnet_5_error.log" , "w")
    traceback.print_exc(file=error_file)
    result_loss = "Error"
    result_acc = "Error"
    error_file.close()
finally:
    file = open('../architecture_results_resnet.csv', 'a', newline ='')
    with file: 

        # identifying header   
        header = ['file_name', 'training_time(s)', 'test_result_loss', 'test_result_acc', 'train_result_acc', 'train_result_loss', 'nb_layers'] 
        writer = csv.DictWriter(file, fieldnames = header) 
      
        # writing data row-wise into the csv file 
        # writer.writeheader() 
        writer.writerow({'file_name' : 'archi_resnet_5',  
                         'training_time(s)': training_time,  
                         'test_result_loss': test_result_loss,
                         'test_result_acc': test_result_acc,
                         'train_result_acc': train_result_acc,
                         'train_result_loss': train_result_loss,
                         'nb_layers': nb_layers}) 
        print('add line into architecture_results.csv')
    file.close()
    