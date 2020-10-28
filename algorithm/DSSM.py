import tensorflow as tf
import numpy as np
from data.data_reader import movie_lens_reader

def load_dataset(batch_size):
    x_list, y_list = movie_lens_reader.load_rating_as_binary()
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    #dataset = tf.data.Dataset.from_tensor_slices((x_list, y_list))
    dataset = tf.data.Dataset.from_tensor_slices(((x_list[:,0], x_list[:,1]), y_list))
    #dataset = dataset.shuffle(len(x_list)).batch(batch_size)
    dataset = dataset.batch(batch_size)
    steps_per_epochs = len(x_list) // batch_size
    return dataset, steps_per_epochs


class DSSM(tf.keras.models.Model):
    def __init__(self, user_cnt, item_cnt, embedding_size, vector_size, hidden_units=1024):
        super(DSSM, self).__init__()

        self.user_embedding = tf.keras.layers.Embedding(user_cnt, embedding_size, dtype=tf.float32)
        self.item_embedding = tf.keras.layers.Embedding(item_cnt, embedding_size, dtype=tf.float32)

        self.user_hidden_layer = tf.keras.layers.Dense(hidden_units, name='user_hidden', dtype=tf.float32)
        self.item_hidden_layer = tf.keras.layers.Dense(hidden_units, name='item_hidden', dtype=tf.float32)

        self.user_vector_layer = tf.keras.layers.Dense(vector_size, name='user_vector', dtype=tf.float32)
        self.item_vector_layer = tf.keras.layers.Dense(vector_size, name='item_vector', dtype=tf.float32)

        self.cosine = tf.keras.layers.Dot(axes=-1, normalize=True, name='cosine', dtype=tf.float32)
        self.out_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output', dtype=tf.float32)


    def call(self, inputs, training=None, mask=None):
        user_input, item_input = inputs[:, 0], inputs[:, 1]

        user_embed = self.user_embedding(user_input)
        item_embed = self.item_embedding(item_input)

        user_dense = self.user_hidden_layer(user_embed)
        user_vector = self.user_vector_layer(user_dense)

        item_dense = self.item_hidden_layer(item_embed)
        item_vector = self.item_vector_layer(item_dense)

        cosine_value = self.cosine([user_vector, item_vector])
        out = self.out_layer(cosine_value)
        return out

    def get_user_vector(self, user_input):
        user_embed = self.user_embedding(user_input)
        user_dense = self.user_hidden_layer(user_embed)
        user_vector = self.user_vector_layer(user_dense)
        return user_vector

    def get_item_vector(self, item_input):
        item_embed = self.item_embedding(item_input)
        item_dense = self.item_hidden_layer(item_embed)
        item_vector = self.item_vector_layer(item_dense)
        return item_vector

def method_1():
    def train_one_batch(x, y):
        with tf.GradientTape() as tape:
            preds = model(x)
            batch_loss = losses(y, preds)
        varaibles = model.trainable_variables
        grads = tape.gradient(batch_loss, varaibles)
        optimizer.apply_gradients(zip(grads, varaibles))
        return batch_loss

    model = DSSM(user_cnt=user_cnt, item_cnt=item_cnt, embedding_size=10, vector_size=10, hidden_units=1024)
    optimizer = tf.keras.optimizers.Adam()
    losses = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=losses)
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(ds.take(steps_per_epochs)):
            batch_loss = train_one_batch(x, y)
            # if batch_idx % 10 == 0:
            #     print ("Epoch {} batch_idx {} batch_loss {}".format(epoch, batch_idx, batch_loss))

    print(model.get_user_vector(test_input))
    model.save("./test_dssm")


def model_in_functional_api(user_cnt, item_cnt, embedding_size, vector_size, hidden_units=1024):
    user_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int64, name='user_input')
    item_input = tf.keras.Input(shape=(1,), dtype=tf.int64, name='item_input')

    user_embedding = tf.keras.layers.Embedding(user_cnt, embedding_size, dtype=tf.float32)
    item_embedding = tf.keras.layers.Embedding(item_cnt, embedding_size, dtype=tf.float32)

    user_hidden_layer = tf.keras.layers.Dense(hidden_units, name='user_hidden', dtype=tf.float32)
    item_hidden_layer = tf.keras.layers.Dense(hidden_units, name='item_hidden', dtype=tf.float32)

    user_vector_layer = tf.keras.layers.Dense(vector_size, name='user_vector', dtype=tf.float32)
    item_vector_layer = tf.keras.layers.Dense(vector_size, name='item_vector', dtype=tf.float32)


    user_embed = user_embedding(user_input)
    item_embed = item_embedding(item_input)

    user_dense = user_hidden_layer(user_embed)
    user_vector = user_vector_layer(user_dense)

    item_dense = item_hidden_layer(item_embed)
    item_vector = item_vector_layer(item_dense)

    cosine_value = cosine([user_vector, item_vector])
    out = out_layer(cosine_value)

    model = tf.keras.Model(inputs=[user_input, item_input], outputs=out)
    model.__setattr__("user_input", user_input)
    model.__setattr__("item_input", item_input)
    model.__setattr__("user_vector", user_vector)
    model.__setattr__("item_vector", item_vector)
    return model


def method_2():
    model = model_in_functional_api(user_cnt, item_cnt, embedding_size=10, vector_size=10, hidden_units=1024)
    optimizer = tf.keras.optimizers.Adam()
    losses = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=losses)
    model.fit(ds, epochs=2)
    model.save("./test_dssm")
    return model

if __name__ == '__main__':
    batch_size = 512
    epochs = 1
    ds, steps_per_epochs = load_dataset(batch_size)
    user_cnt = len(movie_lens_reader.user_idx_dict)
    item_cnt = len(movie_lens_reader.movie_idx_dict)

    test_input = tf.convert_to_tensor([1])

    tf.random.set_seed(2020)

    model = method_2()
    user_embedding_model = tf.keras.models.Model(inputs=model.user_input, outputs=model.user_vector)
    item_embedding_model = tf.keras.models.Model(inputs=model.item_input, outputs=model.item_vector)

    user_embs = user_embedding_model.predict(test_input)
    print(user_embs)

    user_embedding_model.save("./test_user_vector")
    user_embedding_model = tf.keras.models.load_model("./test_user_vector")
    user_embs = user_embedding_model.predict(test_input)
    print (user_embs)






