from typing import Callable

import pytest
import tensorflow as tf
from transformer.encoder import Encoder
from transformer.encoder import EncoderLayer


@pytest.mark.parametrize(
    'seq_length, embed_dim, dff',
    (
        (50, 32, 64),
        (100, 128, 256),
    ),
)
def test_encoder_layer(
    get_embedding: Callable[..., tf.Tensor],
    seq_length: int,
    embed_dim: int,
    dff: int,
) -> None:
    """Test encoder layer's output shape."""
    # Sample embedding.
    embedding = get_embedding(
        seq_len=seq_length,
        embed_dim=embed_dim,
    )

    encoder_layer = EncoderLayer(
        d_model=embed_dim, num_heads=2, dff=dff,
    )

    # Expected output shape.
    expected_shape = tf.TensorShape((None, seq_length, embed_dim))
    assert encoder_layer(embedding).shape == expected_shape


@pytest.mark.parametrize(
    'num_layers, num_heads, seq_len, embed_dim, dff',
    (
        (1, 2, 64, 32, 256),
        (3, 6,  82, 64, 256),
    ),
)
def test_encoder(
    input_data: Callable[..., tf.Tensor],
    num_layers: int,
    num_heads: int,
    seq_len: int,
    embed_dim: int,
    dff: int,
) -> None:
    """Test encoder's output shape."""

    # Sample data (dummy token IDs)
    data = input_data(shape=(seq_len,), dtype=tf.float32)

    # Multiple encoder layers.
    encoder = Encoder(
        num_layers=num_layers,
        d_model=embed_dim,
        num_heads=num_heads,
        dff=dff,
        vocab_size=1_000,
    )
    output = encoder(data, training=False)

    # Expected output shape.
    expected_shape = tf.TensorShape((None, seq_len, embed_dim))
    assert output.shape == expected_shape
