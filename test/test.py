import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

click_seqs_path = "/tf/data/album_click_seqs.txt"
vocab_path = "/tf/data/album_ids.txt"

# 构建词典
def create_mapping_dict(vocab_file):
    item_list = []
    with open(vocab_file, "r") as f:
        for line in f:
            item_id = line.strip()
            item_list.append(item_id)
    item_list = set(item_list)
    vocab_size = len(item_list)
    id_item_dict = dict([(idx, item)for idx, item in enumerate(item_list)])
    item_id_dict = dict([(item, idx) for idx, item in id_item_dict.items()])
    return id_item_dict, item_id_dict, vocab_size


id_item_dict, item_id_dict, vocab_size = create_mapping_dict(vocab_path)

def load_data(input_file, item_id_dict, min_seq_length=3, max_seq_length=200):
    data_list = []
    with open(input_file, "r") as f:
        for line in f:
            babyid, albums  = line.strip().split("\t")
            if babyid.strip() == '0':
                continue
            album_list = albums.strip().split(",")
            album_list = [i for i in album_list if i in item_id_dict]
            if len(album_list) > min_seq_length and len(album_list) < max_seq_length:
                data_list.append(album_list)
    return data_list

def prepare_dataset(input_file, item_id_dict, min_seq_length=3, max_seq_length=200):
    inputs, targets = [], []
    data_list = load_data(input_file, item_id_dict, min_seq_length, max_seq_length)
    for row in data_list:
        row_as_int = []
        for item in row:
            row_as_int.append(item_id_dict[item])
        inputs.append(row_as_int[:-1])
        targets.append(row_as_int[1:])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_seq_length, padding='post', value=0.0)
    targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=max_seq_length, padding='post', value=0.0)
    return inputs, targets

max_seq_length = 20
inputs, targets = prepare_dataset(click_seqs_path, item_id_dict, min_seq_length=3, max_seq_length=20)

# 取部分数据用作测试流程   正式训练需要去除
inputs = inputs[0:50000, ]
targets = inputs[0:50000, ]
print (inputs.shape)

x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.3)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))




embedding_dim = 64
rnn_units = 64
EPOCHS = 10


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.GRU(rnn_units, return_sequences=True,recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy)

checkpoint_dir = "./RNN_generate_checkpoint"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

# history = model.fit(train_ds,
#                     epochs=EPOCHS,
#                     validation_data=val_ds,
#                     validation_freq=1,
#                     callbacks=[checkpoint_callback],
#                     use_multiprocessing=False
#                    )
model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=val_ds,
                    validation_freq=1,
                    use_multiprocessing=False
                   )
model.save("./test_rnn")


def generate_text(model, start_string, item_id_dict):
    num_generate = 10
    input_eval = [item_id_dict[i] for i in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 1.0
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(id_item_dict[predicted_id])
    return ",".join(text_generated)

# predict阶段 重新加载模型 以便于更改预测的batch size
BATCH_SIZE = 1
model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([BATCH_SIZE, None]))


start_string = ['108']
generate_text(model, start_string, item_id_dict)


model = tf.keras.models.load_model("./test_rnn")
start_string = ['108','108']
generate_text(model, start_string, item_id_dict)



