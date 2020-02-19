from utils.config import save_wv_model_path, vocab_path
import tensorflow as tf
from utils.gpu_utils import config_gpu
import tensorflow as tf
from data_process.wv_loader import load_embedding_matrix, Vocab


class Encoder(tf.keras.Model):
    def __init__(self, embedding_matrix, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units

        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.bidirectional_gru = tf.keras.layers.Bidirectional(self.gru)

    def __call__(self, x, hidden):
        x = self.embedding(x)
        # output, hidden = self.gru(x, initial_state=hidden)
        output, forward_state, backward_state = self.bidirectional_gru(x, initial_state=[hidden, hidden])
        hidden = tf.keras.layers.concatenate([forward_state, backward_state], axis=-1)
        return output, hidden

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


def masked_attention(enc_padding_mask, attn_dist):
    """Take softmax of e then apply enc_padding_mask and re-normalize"""
    attn_dist = tf.squeeze(attn_dist, axis=2)
    mask = tf.cast(enc_padding_mask, dtype=attn_dist.dtype)
    attn_dist *= mask  # apply mask
    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
    attn_dist = attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize
    attn_dist = tf.expand_dims(attn_dist, axis=2)
    return attn_dist


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W_s = tf.keras.layers.Dense(units)
        self.W_h = tf.keras.layers.Dense(units)
        self.W_c = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, dec_hidden, enc_output, enc_pad_mask, use_coverage=False, prev_coverage=None):
        """
         calculate attention and coverage from dec_hidden enc_output and prev_coverage
         one dec_hidden(word) by one dec_hidden
         dec_hidden or query is [batch_sz, enc_unit], enc_output or values is [batch_sz, max_train_x, enc_units],
         prev_coverage is [batch_sz, max_len_x, 1]
         dec_hidden is initialized as enc_hidden, prev_coverage is initialized as None
         output context_vector [batch_sz, enc_units] attention_weights & coverage [batch_sz, max_len_x, 1]
         """
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)

        if use_coverage and prev_coverage is not None:
            # Multiply coverage vector by w_c to get coverage_features.
            # self.W_s(values) [batch_sz, max_len, units] self.W_h(hidden_with_time_axis) [batch_sz, 1, units]
            # self.W_c(prev_coverage) [batch_sz, max_len, units]  score [batch_sz, max_len, 1]
            score = self.V(tf.nn.tanh(self.W_s(enc_output) + self.W_h(hidden_with_time_axis) + self.W_c(prev_coverage)))
            # attention_weights shape (batch_size, max_len, 1)

            # attention_weights sha== (batch_size, max_length, 1)
            mask = tf.cast(enc_pad_mask, dtype=score.dtype)
            masked_score = tf.squeeze(score, axis=-1) * mask
            masked_score = tf.expand_dims(masked_score, axis=2)
            attention_weights = tf.nn.softmax(masked_score, axis=1)
            attention_weights = masked_attention(enc_pad_mask, attention_weights)
            coverage = attention_weights + prev_coverage
        else:
            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is (batch_size, max_length, units)
            # 计算注意力权重值
            score = self.V(tf.nn.tanh(
                self.W_s(enc_output) + self.W_h(hidden_with_time_axis)))

            mask = tf.cast(enc_pad_mask, dtype=score.dtype)
            masked_score = tf.squeeze(score, axis=-1) * mask
            masked_score = tf.expand_dims(masked_score, axis=2)
            attention_weights = tf.nn.softmax(masked_score, axis=1)
            attention_weights = masked_attention(enc_pad_mask, attention_weights)
            if use_coverage:
                coverage = attention_weights
            else:
                coverage = []

        # # 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights, coverage


class Decoder(tf.keras.Model):
    def __init__(self, embedding_matrix, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax)

    def __call__(self, dec_inp, hidden, enc_output, context_vector):
        # 使用上次的隐藏层（第一次使用编码器隐藏层）、编码器输出计算注意力权重
        # enc_output shape == (batch_size, max_length, hidden_size)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        # print('x:{}'.format(x))
        dec_inp = self.embedding(dec_inp)

        # 将上一循环的预测结果跟注意力权重值结合在一起作为本次的GRU网络输入
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        dec_x = tf.concat([tf.expand_dims(context_vector, 1), dec_inp], axis=-1)

        # passing the concatenated vector to the GRU
        output, dec_hidden = self.gru(dec_x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        dec_pred = self.fc(output)

        return dec_x, dec_pred, dec_hidden


class Pointer(tf.keras.layers.Layer):

    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def __call__(self, context_vector, dec_hidden, dec_inp):
        # change dec_inp_context to [batch_sz,embedding_dim+enc_units]
        dec_inp = tf.squeeze(dec_inp, axis=1)
        return tf.nn.sigmoid(self.w_s_reduce(dec_hidden) + self.w_c_reduce(context_vector) + self.w_i_reduce(dec_inp))


if __name__ == '__main__':
    # GPU资源配置
    config_gpu()
    # 读取vocab训练
    vocab = Vocab(vocab_path)
    # 计算vocab size
    vocab_size = vocab.count

    # 使用GenSim训练好的embedding matrix
    embedding_matrix = load_embedding_matrix()

    enc_max_len = 200
    dec_max_len = 41
    batch_size = 64
    embedding_dim = 300
    units = 1024

    # 编码器结构
    encoder = Encoder(vocab_size, embedding_dim, embedding_matrix, units, batch_size)
    # encoder input
    enc_inp = tf.ones(shape=(batch_size, enc_max_len), dtype=tf.int32)
    # decoder input
    dec_inp = tf.ones(shape=(batch_size, dec_max_len), dtype=tf.int32)
    # enc pad mask
    enc_pad_mask = tf.ones(shape=(batch_size, enc_max_len), dtype=tf.int32)

    # encoder hidden
    enc_hidden = encoder.initialize_hidden_state()

    enc_output, enc_hidden = encoder(enc_inp, enc_hidden)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(enc_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(enc_hidden.shape))

    attention_layer = BahdanauAttention(10)
    context_vector, attention_weights, coverage = attention_layer(enc_hidden, enc_output, enc_pad_mask)

    print("Attention context_vector shape: (batch size, units) {}".format(context_vector.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
    print("Attention coverage: (batch_size, ) {}".format(coverage))

    decoder = Decoder(vocab_size, embedding_dim, embedding_matrix, units, batch_size)

    dec_x, dec_out, dec_hidden, = decoder(tf.random.uniform((64, 1)),
                                          enc_hidden,
                                          enc_output,
                                          context_vector)
    print('Decoder output shape: (batch_size, vocab size) {}'.format(dec_out.shape))
    print('Decoder dec_x shape: (batch_size, 1,embedding_dim + units) {}'.format(dec_x.shape))

    pointer = Pointer()
    p_gen = pointer(context_vector, dec_hidden, dec_x)
    print('Pointer p_gen shape: (batch_size,1) {}'.format(p_gen.shape))
