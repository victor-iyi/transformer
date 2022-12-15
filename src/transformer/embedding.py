# Copyright (c) 2022 Victor I. Afolabi
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from typing import Optional

import tensorflow as tf
from transformer.utils import positional_encoding


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size: int, d_model: int, length: int = 2048) -> None:
        """Positional embedding layer that looks-up a token's embedding vector
        and adds the position vector.

        Arguments:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the embedding vector.
            length (int, optional): Length of the input sequence.
                Defaults to 2048.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, d_model, mask_zero=True,
        )
        self.pos_encoding = positional_encoding(length=length, depth=d_model)

    def compute_mask(self, *args: tf.Tensor, **kwargs: tf.Tensor) -> Optional[tf.Tensor]:
        """Computes an output mask tensor.

        Arguments:
          input (tf.Tensor): Tensor or list of Tensors.
          mask (tf.Tensor, optional): Tensor or list of Tensors.
            Defaults to None.

        Returns:
          tf.Tensor - None or a list of Tensor
            (one per output tensor of the layer).
        """
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Lookup a token's embedding vector and adds the position vector.

        Arguments:
            x (tf.Tensor): Word embeddings with shape
                (batch_size, n_tokens)

        Returns:
            tf.Tensor - Word embeddings with positional context.
                Returns same shape as input.
        """
        length = tf.shape(x)[1]
        x = self.embedding(x)

        # This factor sets the relative scale of the embedding
        # and positional_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]

        return x
