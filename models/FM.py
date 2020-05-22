import numpy as np
import tensorflow as tf


class FM(tf.keras.models.Model):
    def __init__(self, num_feature, reg_l2=0.01, k=10):
        super().__init__()
        print ("k value is ", k)
        self.num_feature = num_feature
        self.bias = tf.Variable([0.0])
        self.linear_weight = tf.keras.layers.Embedding(num_feature, 1,
                                                       embeddings_initializer='glorot_uniform',
                                                       embeddings_regularizer=tf.keras.regularizers.l2(reg_l2))
        self.feature_embedding = tf.keras.layers.Embedding(num_feature, k,
                                                           embeddings_initializer='glorot_uniform',
                                                           embeddings_regularizer=tf.keras.regularizers.l2(reg_l2))

    def call(self, feature_value):
        #linear part
        feature_index = np.arange(self.num_feature).astype(np.float32)
        feature_value = tf.expand_dims(feature_value, axis=-1)
        linear_weight = self.linear_weight(feature_index)
        wx = tf.math.multiply(linear_weight, feature_value)
        wx_sum = tf.math.reduce_sum(tf.squeeze(wx, axis=-1), axis=1)

        #cross part
        v = self.feature_embedding(feature_index)
        vx = tf.math.multiply(v, feature_value)
        square_of_sum = tf.square(tf.math.reduce_sum(vx, axis=1))
        sum_of_square = tf.reduce_sum(tf.square(vx), axis=1)
        cross_part = 0.5 * tf.reduce_sum(tf.math.subtract(square_of_sum, sum_of_square), axis=1)
        output = self.bias + wx_sum + cross_part
        return output

def construct_model(num_feature):
    return FM(num_feature=num_feature, k=11)

