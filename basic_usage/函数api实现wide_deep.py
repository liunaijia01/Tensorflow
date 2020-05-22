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
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=7)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, random_state=11)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_val_scaled = scaler.transform(x_val)

input = tf.keras.layers.Input(shape=x_train.shape[1:])
hidden1 = tf.keras.layers.Dense(30, activation='sigmoid')(input)
hidden2 = tf.keras.layers.Dense(30, activation='sigmodi')(hidden1)
concat = tf.keras.layers.concatenate([input, hidden2])
output = tf.keras.layers.Dense(1)(concat)

model = tf.keras.models.Model(inputs=[input], outputs=[output])
model.compile(loss='mean_squared_error', optimizer='adam')
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]

history = model.fit(x_train_scaled, y_train, validation_data=(x_val_scaled, y_val), epochs=10, callbacks=callbacks)

model.evaluate(x_test_scaled, y_test)


