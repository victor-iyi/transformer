from typing import Callable

import pytest
import tensorflow as tf
from transformer.feed_forward import FeedForward


@pytest.mark.parametrize(
    'seq_length, embed_dim, dff',
    (
        (64, 256, 128),
        (512, 128, 512),
    ),
)
def test_feed_forward(
    get_embedding: Callable[..., tf.Tensor],
    seq_length: int,
    embed_dim: int,
    dff: int,
) -> None:
    """Check the shape of the cross attention layer's output."""

    # Create a dummy embedding.
    embedding = get_embedding(
        seq_len=seq_length,
        embed_dim=embed_dim,
    )

    # Create cross attention with 2 attention heads.
    sample_ffn = FeedForward(d_model=embed_dim, dff=dff)
    output = sample_ffn(embedding)

    expected_shape = tf.TensorShape((None, seq_length, embed_dim))
    assert output.shape == expected_shape
