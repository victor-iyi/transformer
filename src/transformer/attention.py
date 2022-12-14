# Copyright (c) 2022 Victor I. Afolabi
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import tensorflow as tf


class BaseAttention(tf.keras.layers.Layer):
    """Base attention consists of a MultiHeadAttention, LayerNormalization and Add layer."""

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    """The Cross Attention Layer connects the encoder and the decoder."""

    def call(self, query: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        """Perform a fuzzy, differentiable, vectorized lookup of a query within a context.

        Arguments:
            query (tf.Tensor): Tensor of shape `(..., query_elements, query_depth)`
            context (tf.Tensor): Tensor of shape `(..., key_elements, key_depth)`

        Returns:
            tf.Tensor - A tensor of shape `(..., query_elements, key_depth)`
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
