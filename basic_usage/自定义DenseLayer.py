import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
from sklearn.model_selection import train_test_split
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data,housing.target,random_state=7)
x_train, x_valid, y_train,y_valid = train_test_split(x_train_all,y_train_all,random_state=11)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)


customized_softplus = keras.layers.Lambda(lambda x:tf.nn.softplus(x))

class CustomizedDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.units),
                                      initializer='uniform', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.units, ), initializer='zeors', trainable=True)
        super(CustomizedDenseLayer, self).build(input_shape)

    def call(self, x):
        return self.activation(x @ self.kernel + self.bias)

model = tf.keras.models.Sequential([
    CustomizedDenseLayer(30, activation='relu', input_shape=x_train.shape[1:]),
    keras.layers.Dense(1, activation='softplus')
])

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mean_squared_error'])
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
history = model.fit(x_train_scaled, y_train, validation_data=(x_valid_scaled, y_valid), epochs=100, callbacks=callbacks)
