# Copyright (c) 2022 Victor I. Afolabi
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import tensorflow as tf
from transformer.attention import GlobalSelfAttention
from transformer.embedding import PositionalEmbedding
from transformer.feed_forward import FeedForward


class EncoderLayer(tf.keras.layers.Layer):
    """The encoder contains a stack of `N` encoder layers.

    Each encoder layer contains `GlobalSelfAttention` and `FeedForward`
    layer.
    """

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


class Encoder(tf.keras.layers.Layer):
    """Transformer's Encoder layer."""

    def __init__(
        self, *,
        num_layers: int, d_model: int, num_heads: int,
        dff: int, vocab_size: int, dropout: float = 0.1,
    ) -> None:
        """Encoder half of the Transformer architecture.

        Arguments:
            num_layers (int): Number of `EncoderLayer`.
            d_model (int): Embedding dimension.
            num_heads (int): Number of `MultiHeadAttention` heads.
            dff (int): Number of neurons for the feed forward layer.
            vocab_size (int): Vocabulary size.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model,
        )

        self.encoder_layers = [
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Encoder transforms input of shape `(batch_size, seq_length)`.

        Arguments:
            x (tf.Tensor): Tensor of shape `(batch_size, seq_len)`.

        Returns:
            tf.Tensor - Tensor of shape `(batch_size, seq_len, d_model)`.
        """
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape (batch_size, seq_len, d_model)

        # Add dropout.
        x = self.dropout(x)

        # Pass embedding through a stack of encoder layers.
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x)

        # Shape: (batch_size, seq_len, d_model)
        return x
