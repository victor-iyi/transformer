# Copyright (c) 2022 Victor I. Afolabi
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import tensorflow as tf
from transformer.decoder import Decoder
from transformer.encoder import Encoder


class Transformer(tf.keras.Model):
    """Transformer based on the paper "Attention is all you need" by.

    *Vaswani et al. (2017)*

    """

    def __init__(
        self, *,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        input_vocab_size: int,
        target_vocab_size: int,
        dropout: float = 0.1,
    ) -> None:
        """Transformer architecture.

        Arguments:
            num_layers (int): Number of encoder/decoder layers.
            d_model (int): Embedding dimension.
            num_heads (int): Number of MultiHeadAttention heads.
            dff (int): Number of neurons for the feed forward layer.
            input_vocab_size (int): Input vocabulary size.
            target_vocab_size (int): Target vocabulary size.
            dropout (float, optional): Dropout rate. Defaults to 0.1.

        """
        super().__init__()
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=input_vocab_size,
            dropout=dropout,
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=target_vocab_size,
            dropout=dropout,
        )

        # Final output layer.
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """End-to-end transformer architecture.

        Arguments:
            inputs (tf.Tensor): A tensor or list of Tensor, each with shape
                `(batch_size, seq_len)`.

        Returns:
            tf.Tensor - A tensor of shape
                `(batch_size, target_seq_length, target_vocab_size)`.

        """
        context, x = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_seq_len, d_model)

        # Final linear layer output.
        # (batch_size, target_seq_len, target_vocab_size)
        logits = self.final_layer(x)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits
