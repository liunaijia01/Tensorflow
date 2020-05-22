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

s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]], values=[1, 2, 3], dense_shape=[3, 4])
print (s)
print (tf.sparse.to_dense(s))

s4 = tf.constant([[10, 20],
                  [30, 40],
                  [50, 60],
                  [70, 80]])
print(tf.sparse.sparse_dense_matmul(s, s4))

