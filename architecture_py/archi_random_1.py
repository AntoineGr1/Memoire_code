
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.utils import plot_model
import sys
import traceback
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
		keras.layers.Dense(84, activation='sigmoid'),
		keras.layers.Conv2D(16, kernel_size=5, strides=5, activation='relu', padding=[2,2,2,2]),
		keras.layers.MaxPooling2D(pool_size=1, strides=2, padding='same'),

	])
    plot_model(model, show_shapes=True, to_file="archi_random_1.png")
    model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=5, validation_data=(val_x, val_y))
    print(model.evaluate(test_x, test_y))

    print('OK: file archi_random_1.log has been create')
    log_file = open("archi_random_1.log" , "w")
    log_file.write(str(model.evaluate(test_x, test_y)))
    log_file.close()
except:
    print('error: file archi_random_1_error.log has been create')
    error_file = open("archi_random_1_error.log" , "w")
    traceback.print_exc(file=error_file)
    error_file.close()
