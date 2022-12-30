from typing import Callable

import pytest
import tensorflow as tf
from transformer.decoder import Decoder
from transformer.decoder import DecoderLayer


@pytest.mark.parametrize(
    'target_seq_len, context_seq_len, embed_dim, dff',
    (
        (64, 64, 256, 512),
        (64, 32, 1024, 2048),
    ),
)
def test_decoder_layer(
    get_embedding: Callable[..., tf.Tensor],
    target_seq_len: int,
    context_seq_len: int,
    embed_dim: int,
    dff: int,
) -> None:
    """Test decoder layer's output shape."""

    # Sample input.
    query_embedding = get_embedding(
        seq_len=target_seq_len,
        embed_dim=embed_dim,
        dtype=tf.float32,
    )
    context_embedding = get_embedding(
        seq_len=context_seq_len,
        embed_dim=embed_dim,
        dtype=tf.float32,
    )

    decoder_layer = DecoderLayer(
        d_model=embed_dim,
        num_heads=2,
        dff=dff,
    )
    output = decoder_layer(query_embedding, context_embedding)

    # Expected output shape.
    expected_shape = tf.TensorShape((None, target_seq_len, embed_dim))
    assert output.shape == expected_shape


@pytest.mark.parametrize(
    'num_layers, target_seq_len, context_seq_len, embed_dim, dff',
    (
        (1, 32, 64, 256, 1024),
        (8, 64, 32, 512, 2048),
    ),
)
def test_decoder(
    input_data: Callable[..., tf.Tensor],
    get_embedding: Callable[..., tf.Tensor],
    num_layers: int,
    target_seq_len: int,
    context_seq_len: int,
    embed_dim: int,
    dff: int,
) -> None:
    """Test decoder's output shape."""
    # Hyperparameters.
    vocab_size = 1_000

    # Target tokens.
    target = input_data(shape=(target_seq_len,))

    # Context embedding.
    context_embedding = get_embedding(
        seq_len=context_seq_len,
        embed_dim=embed_dim,
        dtype=tf.float32,
        vocab_size=vocab_size,
    )

    # Decoder.
    decoder = Decoder(
        num_layers=num_layers, d_model=embed_dim,
        num_heads=2, dff=dff, vocab_size=vocab_size,
    )
    output = decoder(target, context_embedding)
    expected_shape = tf.TensorShape((None, target_seq_len, embed_dim))
    assert output.shape == expected_shape
