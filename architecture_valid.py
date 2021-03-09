
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.utils import plot_model
import sys
(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()

# normaliser les pixel 0-255 -> 0-1
train_x = train_x / 255.0
test_x = test_x / 255.0

train_x = tf.expand_dims(train_x, 3)
test_x = tf.expand_dims(test_x, 3)

val_x = train_x[:5000]
val_y = train_y[:5000]

try:
    model = keras.models.Sequential([
		keras.layers.Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=train_x[0].shape, padding='same'),
		keras.layers.AveragePooling2D(pool_size=2, strides=None, padding='valid'),
		keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'),
		keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='valid'),
		keras.layers.Flatten(),
		keras.layers.Dense(120, activation='tanh'),
		keras.layers.Dense(84, activation='tanh'),
		keras.layers.Dense(10, activation='softmax'),

	])
    plot_model(model, show_shapes=True, to_file="architecture_valid.png")
    model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=5, validation_data=(val_x, val_y))
    print(model.evaluate(test_x, test_y))

    print('OK: file architecture_valid.log has been create')
    log_file = open("architecture_valid.log" , "w")
    log_file.write(str(model.evaluate(test_x, test_y)))
    log_file.close()
except:
    print('error: file architecture_valid_error.log has been create')
    error_file = open("architecture_valid_error.log" , "w")
    error_file.write(str(sys.exc_info()))
    error_file.close()
