import tensorflow as tf
import numpy as np

def positional_encoding(pos, d_model):
    def get_angles(position, i):
        return position / np.power(10000, 2 * (i // 2) / np.float(d_model))

    angle_rates = get_angles(np.arange(pos)[:, np.newaxis],
                             np.arange(d_model)[np.newaxis, :])

    pe_sin = np.sin(angle_rates[:, 0::2])
    pe_cos = np.cos(angle_rates[:, 1::2])
    pos_encoding = np.concatenate([pe_sin, pe_cos], axis=-1)
    pos_encoding = tf.cast(pos_encoding[np.newaxis, ], tf.float32) # (1, 50, 512)
    return pos_encoding


'********* 第一部分：scaled dot-product attention *********'
def scaled_dot_product_attention(q, k, v, mask):
    """
        attention(Q,K,V) = softmax(Q * K^T / sqrt(dk)) * V
        q k v 是embedding向量
        dk embedding 的维度
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(q)[-1], tf.float32)
    scaled_attention = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention += mask * -1e-9  #将mask的token乘-1e-9 与attention相加后 mask的位置经过softmax后为0
    attention_weights = tf.nn.softmax(scaled_attention)
    outputs = tf.matmul(attention_weights, v)
    return outputs, attention_weights

"""******  multi-head attention"""
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        # d_model必须可以正确分成多个头
        assert d_model % num_heads == 0
        # 分头之后维度
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # 分头，将头个数的维度，放到seq_len前面 x输入shape=[batch_size, seq_len, d_model]
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        # 分头前的前向网络，根据q,k,v的输入，计算Q, K, V语义
        q = self.wq(q)  # shape=[batch_size, seq_len_q, d_model]
        k = self.wq(k)
        v = self.wq(v)
        # 分头
        q = self.split_heads(q, batch_size)  # shape=[batch_size, num_heads, seq_len_q, depth]
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # 通过缩放点积注意力层
        # scaled_attention shape=[batch_size, num_heads, seq_len_q, depth]
        # attention_weights shape=[batch_size, num_heads, seq_len_q, seq_len_k]
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) # shape=[batch_size, seq_len_q, num_heads, depth]
        # 把多头合并
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) # shape=[batch_size, seq_len_q, d_model]
        # 全连接重塑
        output = self.dense(concat_attention)
        return output, attention_weights

class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1:],
            initializer=tf.ones_initializer(),
            trainable=True
        )
        self.beta = self.add_weight(
            name='gamma',
            shape=input_shape[-1:],
            initializer=tf.ones_initializer(),
            trainable=True
        )
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


def point_wise_feed_forward(d_model, diff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(diff, activation=tf.nn.relu),
        tf.keras.layers.Dense(d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward(d_model, dff) # ff feed forward
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training, mask):
        att_output, _ = self.mha(inputs, inputs, inputs, mask)
        att_output = self.dropout1(att_output, training=training)
        output1 = self.layernorm1(inputs + att_output)
        ffn_output = self.ffn(output1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layernorm2(output1 + ffn_output)
        return output2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_layers, num_heads, dff, input_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.emb = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)
        self.encoder_layer = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training, mask):
        seq_len = inputs.shape[1]
        word_embedding = self.emb(inputs)
        word_embedding *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        emb = word_embedding + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(emb, training=training)
        for i in range(self.num_layers):
            x = self.encoder_layer[i](x, training, mask)
        return x


def create_padding_mask(seq):
    """
        避免输入中padding的token对句子的意思产生影响，需要将padding位mark掉
        encoder和decoder过程中都会用到
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, np.newaxis, np.newaxis, :]

def create_look_ahead_mask(size):
    """
        用于对未预测的token进行掩码
        要预测第三个单词时，只会用到第一个和第二个词
        要预测第四个单词时，只会用到第一、二、三个单词
        只有decoder过程用到
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_mask(inputs, targets):
    encoder_padding_mask = create_padding_mask(inputs)
    decoder_padding_mask = create_padding_mask(inputs)

    seq_mask = create_look_ahead_mask(tf.shape(targets)[1])
    decoder_targets_padding_mask = create_padding_mask(targets)

    look_ahead_mask = tf.maximum(decoder_targets_padding_mask, seq_mask)
    return encoder_padding_mask, look_ahead_mask, decoder_padding_mask

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward(d_model, dff)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, encoder_out, training, look_ahead_mask, padding_mask):
        # masked multi-head attention: Q = K = V
        att_out1, att_weight1 = self.mha1(inputs, inputs, inputs, look_ahead_mask)
        att_out1 = self.dropout1(att_out1, training=training)
        att_out1 = self.layernorm1(inputs + att_out1)

        # multi-head attention: Q=att_out1, K = V = encoder_out
        att_out2, att_weight2 = self.mha2(att_out1, encoder_out, encoder_out, padding_mask)
        att_out2 = self.dropout2(att_out2, training=training)
        att_out2 = self.layernorm2(att_out1 + att_out2)

        # feed forward network
        ffn_out = self.ffn(att_out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        output = self.layernorm3(att_out2 + ffn_out)
        return output, att_weight1, att_weight2

class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_layers, num_heads, dff, target_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.seq_len = tf.shape
        self.d_model = d_model
        self.num_layers = num_layers
        self.word_embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)
        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, encoder_out, training, look_ahead_mask, padding_mask):
        seq_len = inputs.shape[1]
        attention_weights = {}
        word_embedding = self.word_embedding(inputs)
        word_embedding *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        emb = word_embedding + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(emb, training=training)
        for i in range(self.num_layers):
            x, att1, att2 = self.decoder_layers[i](x, encoder_out, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_att_w1'.format(i+1)] = att1
            attention_weights['decoder_layer{}_att_w2'.format(i+1)] = att2
        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, d_model, num_layers, num_heads, dff, input_vocab_size, target_vocab_size, max_seq_len, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, num_layers, num_heads, dff, input_vocab_size, max_seq_len, dropout_rate)
        self.decoder = Decoder(d_model, num_layers, num_heads, dff, target_vocab_size, max_seq_len, dropout_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, targets, training, encoder_padding_mask, look_ahead_mask, decoder_padding_mask):
        encoder_output = self.encoder(inputs, training, encoder_padding_mask)
        decoder_output, att_weights = self.decoder(targets, encoder_output, training, look_ahead_mask, decoder_padding_mask)

        final_out = self.final_layer(decoder_output)
        return final_out, att_weights


if __name__ == '__main__':
    sample_transformer = Transformer(num_layers=2, d_model=512, num_heads=8,
                                     dff=1024, input_vocab_size=8500,
                                     target_vocab_size=8000, max_seq_len=120)
    temp_input = tf.random.uniform((64, 62))
    temp_target = tf.random.uniform((64, 26))
    fn_out, att = sample_transformer(temp_input, temp_target, training=False, encoder_padding_mask=None,
                                     look_ahead_mask=None, decoder_padding_mask=None)
    #print(fn_out)







