# Copyright (c) 2022 Victor I. Afolabi
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import tensorflow as tf


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model: int, dff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout),
        ])
        self.add = tf.keras.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x
