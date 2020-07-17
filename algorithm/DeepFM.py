import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


def load_data(file_name):
    df = pd.read_csv(file_name)
    target = df.pop('target').tolist()
    for col in df.columns:
        lbe_encoder = LabelEncoder()
        df[col] = lbe_encoder.fit_transform(df[col])

    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    data = one_hot_encoder.fit_transform(df).toarray()
    x_train, y_train, x_test, y_test = train_test_split(data, target, test_size=0.3, random_state=1)
    return x_train, y_train, x_test, y_test

class DeepFM(tf.keras.Model):
    def __init__(self,
                 num_feature,
                 num_field,
                 dropout_fm,
                 dropout_deep,
                 deep_activation='relu',
                 layer_sizes=[200, 100],
                 embedding_size=10):
        super(DeepFM, self).__init__()
        self.num_feature = num_feature
        self.num_field = num_field
        self.dropout_fm = dropout_fm
        self.dropout_deep = dropout_deep
        self.layer_sizes = layer_sizes
        self.embedding_size = embedding_size

        self.linear_weights = tf.keras.layers.Embedding(num_feature, 1, embeddings_initializer='uniform')
        self.feature_embedding = tf.keras.layers.Embedding(num_feature, embedding_size, embeddings_initializer='uniform')

        for i in range(len(layer_sizes)):
            setattr(self, 'dense_' + str(i), tf.keras.layers.Dense(layer_sizes[i]))
            setattr(self, 'batchNorm_' + str(i), tf.keras.layers.BatchNormalization())
            setattr(self, 'activation_' + str(i), tf.keras.layers.Activation(deep_activation))
            setattr(self, 'dropout_' + str(i), tf.keras.layers.Dropout(dropout_deep[i + 1]))

        self.fc = tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True)


    def call(self, feature_index, feature_value, use_dropout=True):
        feature_value = tf.expand_dims(feature_value, axis=-1)
        #linear part
        linear_weights = self.linear_weights(feature_index)
        linear_wx = tf.math.multiply(linear_weights, feature_value)
        linear_out = tf.math.reduce_sum(linear_wx, axis=2) # None * num_field
        if use_dropout:
            linear_out = tf.keras.layers.Dropout(self.dropout_fm[0])(linear_out)

        #fm interaction part
        feature_embedding = self.feature_embedding(feature_index)
        feature_embedding_value = tf.math.multiply(feature_embedding, feature_value) # None *  num_field * embedding_size

        interaction_part1 = tf.math.pow(tf.math.reduce_sum(feature_embedding_value, axis=1), 2)
        interaction_part2 = tf.math.reduce_sum(tf.math.pow(feature_embedding_value, 2), axis=1)
        interaction_out = 0.5 * tf.math.subtract(interaction_part1, interaction_part2) # None * embedding_size

        #DNN part
        dnn_out = tf.reshape(feature_embedding_value, (-1, self.num_feature * self.embedding_size))
        if use_dropout:
            dnn_out = tf.keras.layers.Dropout(self.dropout_deep[0])(dnn_out)
        for i in range(len(self.layer_sizes)):
            dnn_out = getattr(self, 'dense_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            if use_dropout:
                dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)

        concat_input = tf.concat((linear_out, interaction_out, dnn_out), axis=1)
        output = self.fc(concat_input)
        return output

if __name__ == '__main__':
    file_name = "../data/CTR/train.csv"
    x_train, y_train, x_test, y_test = load_data(file_name)
    num_feature = 672225
    num_field = 110

    deep_fm_model = DeepFM(num_feature=num_feature,
                           num_field=num_field,
                           dropout_fm=[0.0, 0.0],
                           dropout_deep=[0.5, 0.5, 0.5, 0.5],
                           layer_sizes=[200, 100],
                           embedding_size=10)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(num_feature,), name='inputs'))
    model.add(deep_fm_model)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

    model.fit(x_train, y_train, batch_size=128, epochs=2, validation_data=(x_test, y_test))
    model.save("./model_file/")


