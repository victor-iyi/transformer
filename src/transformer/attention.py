# Copyright (c) 2022 Victor I. Afolabi
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import tensorflow as tf


class BaseAttention(tf.keras.layers.Layer):
    """Base attention consists of a MultiHeadAttention, LayerNormalization and Add layer."""

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        Arguments:
            See arguments for `tf.keras.layers.MultiHeadAttention`.

        """
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    """The Cross Attention Layer connects the encoder and the decoder."""

    def call(self, query: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        """Perform a fuzzy, differentiable, vectorized lookup of a query within a context.

        Arguments:
            query (tf.Tensor): Query vector.
                Tensor of shape `(..., query_length, query_depth)`
            context (tf.Tensor): Context (or key) vector.
                Tensor of shape `(..., context_length, context_depth)`

        Returns:
            tf.Tensor - A tensor of shape `(batch_size, query_length, embdding_dim)`
        """
        attn_output, attn_scores = self.mha(
            query=query,
            key=context,
            value=context,
            return_attention_scores=True,
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([query, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    """Global self attention lets every sequence element directly access every
    other sequence element, with only a few operations, and all the outputs can
    be computed in parallel.

    The query `x`, is passed to the `MultiHeadAttention` layer as the `query`,
    `key` & `value`.
    """

    def call(self, query: tf.Tensor) -> tf.Tensor:
        """Self attention layer for query vector `x`.

        Arguments:
            query (tf.Tensor): Tensor or list of Tensor of shape
                (batch_size, seq_length, embedding_dim)

        Returns:
            tf.Tensor - A tensor of shape `(batch_size, seq_length, embedding_dim)`
        """
        # The query is passed as Q, K & V for self-attention.
        attn_output = self.mha(query=query, value=query, key=query)

        x = self.add([query, attn_output])
        x = self.layernorm(x)

        return x
