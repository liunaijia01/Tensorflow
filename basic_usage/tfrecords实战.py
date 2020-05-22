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

expect_feature = {
    "input_features": tf.io.FixedLenFeature([8], dtype=tf.float32),
    "label": tf.io.FixedLenFeature([1], dtype=tf.float32)
}

def parse_example(serialized_example):
    example = tf.io.parse_single_example(serialized_example, expect_feature)
    return example['input_features'], example['label']

def tfrecords_reader_dataset(filename, n_reader=5, batch_size=32, n_parse_threads=5, shuffle_buffer_size=1000):
    dataset = tf.data.Dataset.list_files(filename)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename:tf.data.TFRecordDataset(filename, compression_type="GZIP"),
        cycle_length=n_reader
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_example, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset


