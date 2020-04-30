import tensorflow as tf


def construct_model(n_feature):
    model = tf.keras.layers.Dense(1, input_shape=(n_feature,), activation=tf.nn.sigmoid)
    return model