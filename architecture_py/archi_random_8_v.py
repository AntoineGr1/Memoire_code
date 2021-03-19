
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
import numpy as np
from tensorflow.keras.utils import plot_model
import sys
import traceback
import csv
from time import time

# load dataset
(train_x, train_y), (test_x, test_y) = cifar10.load_data()

# normalize to range 0-1
train_x = train_x / 255.0
test_x = test_x / 255.0

val_x = train_x[:5000]
val_y = train_y[:5000]
    


# init training time
training_time = 0
# init result
result_loss = ""
result_acc = ""
try:
    model = keras.models.Sequential([
		keras.layers.Input([32, 32, 3]),
		keras.layers.Conv2D(18, kernel_size=3, strides=2, activation='tanh', padding='valid'),
		keras.layers.MaxPooling2D(pool_size=5, strides=1, padding='same'),
		keras.layers.Flatten(),
		keras.layers.Dense(1215, activation='tanh'),
		keras.layers.Dense(850, activation='selu'),
		keras.layers.Dense(10, activation='softmax'),

	])
    plot_model(model, show_shapes=True, to_file="../architecture_img/archi_random_8_v.png")
    model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    start = time()
    model.fit(train_x, train_y, epochs=5, validation_data=(val_x, val_y))
    training_time = time()-start
    print(model.evaluate(test_x, test_y))

    print('OK: file ../architecture_log/archi_random_8_v.log has been create')
    log_file = open("../architecture_log/archi_random_8_v.log" , "w")
    log_file.write(str(model.evaluate(test_x, test_y)))
    result_loss = model.evaluate(test_x, test_y)[0]
    result_acc = model.evaluate(test_x, test_y)[1]
    nb_layers = len(model.layers)
    log_file.close()
except:
    print('error: file ../architecture_log/archi_random_8_v_error.log has been create')
    error_file = open("../architecture_log/archi_random_8_v_error.log" , "w")
    traceback.print_exc(file=error_file)
    result_loss = "Error"
    result_acc = "Error"
    error_file.close()
finally:
    file = open('../architecture_results.csv', 'a', newline ='')
    with file: 

        # identifying header   
        header = ['file_name', 'training_time(s)', 'result_loss', 'result_acc', 'nb_layers'] 
        writer = csv.DictWriter(file, fieldnames = header) 
      
        # writing data row-wise into the csv file 
        # writer.writeheader() 
        writer.writerow({'file_name' : 'archi_random_8_v',  
                         'training_time(s)': training_time,  
                         'result_loss': result_loss,
                         'result_acc': result_acc,
                         'nb_layers': nb_layers}) 
        print('add line into architecture_results.csv')
    