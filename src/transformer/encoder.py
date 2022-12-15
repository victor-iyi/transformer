# Copyright (c) 2022 Victor I. Afolabi
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import tensorflow as tf
from transformer.attention import GlobalSelfAttention
from transformer.feed_forward import FeedForward


class EncoderLayer(tf.keras.layers.Layer):
    """The encoder contains a stack of `N` encoder layers. Each encoder layer
    contains `GlobalSelfAttention` and `FeedForward` layer."""

    def __init__(
        self, *,
        d_model: int, num_heads: int,
        dff: int, dropout: float = 0.1,
    ) -> None:
        """A single encoder layer.

        Arguments:
            d_model (int): Embedding dimension.
            num_heads (int): Number of MultiHeadAttention heads.
            dff (int): Number of neurons for the feed forward layer.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout,
        )
        self.ffn = FeedForward(d_model=d_model, dff=dff)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Encoder layer with self attention and feed forward layer.

        Arguments:
            x (tf.Tensor): An embedding tensor of shape
                `(batch_size, seq_length, embed_dim)`.

        Returns:
            tf.Tensor - A Tensor of shape `(batch_size, seq_length, embedding_dim)`.
        """
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
