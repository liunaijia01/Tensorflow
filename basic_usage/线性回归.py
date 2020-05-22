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
x_train, x_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.3, random_state=7)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=11)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

model = tf.keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=x_train.shape[1:]),
    keras.layers.Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='sgd')
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
history = model.fit(x_train_scaled, y_train, validation_data=(x_val_scaled, y_val), epochs=100, callbacks=callbacks)