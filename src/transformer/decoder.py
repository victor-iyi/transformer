# Copyright (c) 2022 Victor I. Afolabi
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import tensorflow as tf
from transformer.attention import CausalSelfAttention
from transformer.attention import CrossAttention
from transformer.embedding import PositionalEmbedding
from transformer.feed_forward import FeedForward


class DecoderLayer(tf.keras.layers.Layer):
    """Decoder layer consists of `CausalSelfAttention`, `CrossAttention` and
    `FeedForward` layer."""

    def __init__(
        self, *,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout: float = 0.1,
    ) -> None:
        """Decoder layer.

        Arguments:
            d_model (int): Embedding dimension.
            num_heads (int): Number of `MultiHeadAttention` heads.
            dff (int): Number of neurons for the feed forward layer.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()

        # Causal self attention for auto-regressive attention.
        self.causal_self_attn = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout,
        )

        # Cross attention between query & context vectors.
        self.cross_attn = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout,
        )

        # Feed Forward layer.
        self.ffn = FeedForward(d_model=d_model, dff=dff)

    def call(self, query: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        """A decoder layer.

        Arguments:
            query (tf.Tensor): Query vector of shape
                `(batch_size, target_seq_len, d_model)`.
            context (tf.Tensor): Context vector of shape
                `(batch_size, context_seq_len, d_model)`.

        Returns:
            tf.Tensor - Tensor of shape `(batch_size, target_seq_len, d_model)`.
        """
        x = query

        # Output shape: (batch_size, query_seq_len, embed_dim)
        x = self.causal_self_attn(query=x)
        # Output shape: (batch_size, query_seq_len, embed_dim)
        x = self.cross_attn(query=x, context=context)

        # Cache the last attention score for plotting layer.
        self.last_attn_scores = self.cross_attn.last_attn_scores

        # Output shape: (batch_size, query_seq_len, d_model)
        x = self.ffn(x)

        return x


class Decoder(tf.keras.layers.Layer):
    """Transformer's Decoder layer."""

    def __init__(
        self, *,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        vocab_size: int,
        dropout: float = 0.1,
    ) -> None:
        """Decoder half of the Transformer architecture.

        Arguments:
            num_layers (int): Number of `DecoderLayer`.
            d_model (int): Embedding dimension.
            num_heads (int): Number of `MultiHeadAttention` heads.
            dff (int): Number of neurons for the feed forward layer.
            vocab_size (int): Vocabulary size.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Positional embedding.
        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
        )
        self.dropout = tf.keras.layers.Dropout(dropout)

        # Stack of decoder layers.
        self.decoder_layers = [
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]
        self.last_attn_scores = None

    def call(self, query: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        """Decoder takes the target token with some context embeddings.

        Arguments:
            query (tf.Tensor): Tensor of shape `(batch_size, seq_len)`.
            context (tf.Tensor): Context embedding of shape
                `(batch_size, context_seq_len, d_model)`.

        Returns:
            tf.Tensor - A tensor of shape `(batch_size, seq_len, d_model)`.
        """
        # `query` is token-IDs shape (batch_size, target_seq_len)
        x = self.pos_embedding(query)  # (batch_size, target_seq_len, d_model).

        x = self.dropout(x)

        # Pass target & context embedding through decoder layers.
        for i in range(self.num_layers):
            x = self.decoder_layers[i](x, context)

        # Cache the last attention scores for plotting.
        self.last_attn_scores = self.decoder_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x
