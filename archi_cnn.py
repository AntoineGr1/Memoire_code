import keras
import numpy as np
from keras.utils import plot_model
model = keras.models.Sequential([
	keras.layers.Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=[28, 28, 1], padding='same'),
	keras.layers.AveragePooling2D(pool_size=2, strides=None, padding='valid'),
	keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'),
	keras.layers.MaxPooling2D(pool_size=2, strides=None, padding='valid'),
	keras.layers.Flatten(),
	keras.layers.Dense(120, activation='tanh'),
	keras.layers.Dense(84, activation='tanh'),
	keras.layers.Dense(10, activation='softmax'),

])
plot_model(model, show_shapes=True, to_file='model_V.png')