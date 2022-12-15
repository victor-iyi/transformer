# Copyright (c) 2022 Victor I. Afolabi
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import tensorflow as tf


class FeedForward(tf.keras.layers.Layer):
    """Point-wise feed forward layer."""

    def __init__(self, d_model: int, dff: int, dropout: float = 0.1) -> None:
        """Feed forward layer.

        Arguments:
            d_model (int): Embedding dimension.
            dff (int): Number of neurons for the feed forward layer.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout),
        ])
        self.add = tf.keras.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Feed forward layer with a residual add layer and a normalization layer.

        Arguments:
            x (tf.Tensor): Word embedding of shape `(batch_size, seq_length, embed_dim)`.

        Returns:
            tf.Tensor - Returns the same shape as the input.
        """
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x
