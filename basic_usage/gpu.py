import tensorflow as tf
from tensorflow import keras
import os
import numpy as np


tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(gpus[0:3], 'GPU')   #指定几个可用的gpu
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


train_dataset = None
x_train_scaled = None
batch_size = None

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3,
                                  padding='same',
                                  activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=3,
                                  padding='same',
                                  activation='relu'))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=3,
                                  padding='same',
                                  activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=3,
                                  padding='same',
                                  activation='relu'))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=3,
                                  padding='same',
                                  activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer = "sgd",
                  metrics = ["accuracy"])
history = model.fit(train_dataset,
                    steps_per_epoch = x_train_scaled.shape[0] // batch_size,
                    epochs=10)