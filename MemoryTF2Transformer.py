import numpy as np
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, embed_dim)

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        return pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='gelu'),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False, mask=None):
        attn_output = self.attention(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerEncoder(tf.keras.Model):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout_rate, output_dim=2):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.pos_encoding = PositionalEncoding(sequence_length=100, embed_dim=embed_dim)
        self.input_proj = tf.keras.layers.Dense(embed_dim)
        self.enc_layers = [
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='sigmoid')

    def call(self, inputs, training=False, mask=None):
        x = self.input_proj(inputs)
        x = self.pos_encoding(x)
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training, mask=mask)
        return self.output_layer(x)


class MemoryDNN:
    def __init__(self, net, max_users=None, learning_rate=0.001, memory_size=2000,
                 batch_size=128, training_interval=10, output_dim=2):
        self.net = net
        self.max_users = max_users if max_users is not None else net[1]
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.training_interval = training_interval
        self.kernal_size = int(net[0] / self.max_users)
        self.output_dim = output_dim

        self.model = TransformerEncoder(
            num_layers=4,
            embed_dim=128,
            num_heads=4,
            ff_dim=256,
            dropout_rate=0.1,
            output_dim=self.output_dim
        )
        self.model.build(input_shape=(None, self.max_users, self.kernal_size))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.memory_counter = 0
        self.cost_his = []  # Track training loss

    def remember(self, x, y):
        self.memory.append((x, y))
        self.memory_counter += 1

    def learn(self):
        if self.memory_counter < self.batch_size:
            return
        mini_batch = random.sample(self.memory, self.batch_size)
        h_train = np.array([data[0] for data in mini_batch])
        y_train = np.array([data[1] for data in mini_batch])

        h_train = h_train.reshape(-1, self.max_users, self.kernal_size)

        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy')
        history = self.model.fit(h_train, y_train, batch_size=self.batch_size, verbose=0)
        self.cost_his.append(history.history['loss'][0])

    def encode(self, x, y):
        self.remember(x, y)
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def decode(self, x, K=None, decoder_mode='OP'):
        if K is None:
            K = self.max_users
        x_reshaped = x.reshape(1, self.max_users, self.kernal_size).astype(np.float32)

        y_pred = self.model(x_reshaped, training=False).numpy()
        y_pred = y_pred[0]

        m_pred = np.zeros_like(y_pred)
        m_pred[np.arange(self.max_users), y_pred.argmax(axis=1)] = 1

        m_list = [m_pred]
        return m_pred, m_list

    def plot_cost(self):
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.title('Transformer MemoryDNN Training Cost')
        plt.grid(True)
        plt.show()
