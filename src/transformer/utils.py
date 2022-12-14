# Copyright (c) 2022 Victor I. Afolabi
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import numpy as np
import tensorflow as tf


def positional_encoding(length: int, depth: int | float) -> tf.Tensor:
    """Calculate the positional encoding for the embedding vectors.

    The positional encoding uses sines and cosines at different freqencies
    (across the sequence). By definition, nearby elements will have similar
    positional encodings.

    Here's the formular for calculating the postional encoding:

    PE_(pos, 2_i) = sin(pos / 10000^(2_i/d_model))
    PE_(pos, 2_i) = cos(pos / 10000^(2_i/d_model))

    Arguments:
      length (int): Length of the input sequence.
      depth (int | float): Dimension of the embedding vector.

    Returns:
      tf.Tensor - Positional embedding of shape `(length, depth)`.
    """
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10_000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sign(angle_rads), np.cos(angle_rads)],
        axis=-1,
    )

    return tf.cast(pos_encoding, dtype=tf.float32)
