import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
favorite_books = [name.encode('utf-8') for name in ['machine learning', 'cc150']]
favorite_books_bytelist = tf.train.BytesList(value=favorite_books)
hours_floatlist = tf.train.FloatList(value=[15.5,9.5,70,80])

age_int64list = tf.train.Int64List(value=[42])

features = tf.train.Features(
    feature={
        "favorite_books": tf.train.Feature(bytes_list=favorite_books_bytelist),
        "hours": tf.train.Feature(float_list=hours_floatlist),
        "age": tf.train.Feature(int64_list=age_int64list)
    }
)
example = tf.train.Example(features=features)
serialized_expample = example.SerializeToString()

# 写
output_dir="./"
filename = "basic_usage.tfrecords"
filename_path = os.path.join(output_dir,filename)
with tf.io.TFRecordWriter(filename_path) as writer:
    for i in range(3):
        writer.write(serialized_expample)

#读
dataset = tf.data.TFRecordDataset([filename_path])
for serialized_expample_tensor in dataset:
    print (serialized_expample_tensor)



#生成tfrecords文件
def serialize_example(x, y):
    input_features = tf.train.FloatList(value=x)
    label = tf.train.FloatList(value=y)
    features = tf.train.Features(
        features={
            "input_feature": tf.train.Feature(float_list=input_features),
            "label": tf.train.Feature(float_list=label)
        }
    )
    example = tf.train.Example(features=features)
    return example.SerializeToString()

def csv_dataset_to_tfrecords(base_filename, dataset, n_shards, steps_per_shads, compression_type=None):
    options = tf.io.TFRecordOptions(compression_type=compression_type)
    all_filename = []
    for shard_id in range(n_shards):
        filename_fullpath = '{}_{:05d}-of-{:05d}', format(
            base_filename, shard_id, n_shards)
        with tf.io.TFRecordWriter(filename_fullpath, options) as writer:
            for x_batch, y_batch in dataset.take(steps_per_shads):
                for x_example, y_example in zip(x_batch, y_batch):
                    writer.write(serialized_expample(x_example, y_example))
        all_filename.append(filename_fullpath)
    return all_filename




