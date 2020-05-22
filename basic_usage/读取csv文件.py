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

train_filenames = ["file1.csv", "file2.csv", "file3.csv", "file4.csv"]
#os.listdir(in_path)
filename_dataset = tf.data.Dataset.list_files(train_filenames)
n_reader = 5
dataset = filename_dataset.interleave(
    lambda filename:tf.data.TextLineDataset(filename),
    cycle_length=n_reader
)

def parse_csv_line(line, n_field):
    defs = [tf.constant(np.nan)] * n_field
    parsed_field = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(parsed_field[0:-1])
    y = tf.stack(parsed_field[-1:])
    return x, y


def csv_reader_dataset(filenames, n_reader=5, batch_size=32, n_parse_threads=5, shuffle_buffer_size=1000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset.interleave(
        lambda filename:tf.data.TextLineDataset(filename).skip(1),
        cycle_length=n_reader
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset.map(parse_csv_line, num_parallel_calls=n_parse_threads)
    dataset.batch(batch_size)
    return dataset

batch_size = 32
train_set = csv_reader_dataset(train_filename,batch_size=batch_size)
valid_set = csv_reader_dataset(valid_filename,batch_size=batch_size)
test_set = csv_reader_dataset(test_filename,batch_size=batch_size)

model = tf.keras.models.Sequential([
    keras.layers.Dense(30,activation='relu',input_shape=[8]),
    keras.layers.Dense(1)
])
model.compile(loss = "mean_squared_error",optimizer='sgd')
callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-2)]
history = model.fit(train_set, validation_data=valid_set, steps_per_epoch = 11160// batch_size,
                    validation_steps = 3870//batch_size, epochs=100, callbacks=callbacks)
model.evaluate(test_set, steps=5160//batch_size)
