import numpy as np
import pytest
import tensorflow as tf
from transformer.decoder import Decoder
from transformer.decoder import DecoderLayer


@pytest.mark.parametrize(
    'target_seq_len,context_seq_len,embed_dim',
    (
        (128, 256, 1024),
        (64, 128, 2048),
    ),
)
def test_decoder_layer(target_seq_len: int, context_seq_len: int, embed_dim: int) -> None:
    """Test decoder layer's output shape."""
    # Hyperparameters.
    batch_size, dff = 32, 1024

    # Sample embedding.
    query_embedding = tf.random.normal(
        shape=(batch_size, target_seq_len, embed_dim),
        dtype=tf.float32,
    )
    context_embedding = tf.random.normal(
        shape=(batch_size, context_seq_len, embed_dim),
        dtype=tf.float32,
    )

    decoder_layer = DecoderLayer(d_model=embed_dim, num_heads=2, dff=dff)
    output = decoder_layer(query_embedding, context_embedding)
    assert output.shape == (batch_size, target_seq_len, embed_dim)


@pytest.mark.parametrize(
    'num_layers,target_seq_len,context_seq_len,embed_dim',
    (
        (1, 32, 64, 512),
        (8, 64, 32, 1024),
    ),
)
def test_decoder(
    num_layers: int,
    target_seq_len: int,
    context_seq_len: int,
    embed_dim: int,
) -> None:
    # Hyperparameters.
    batch_size, vocab_size, dff = 32, 10_000, 1024

    # Sample target.
    target = np.random.randint(
        vocab_size,
        size=(batch_size, target_seq_len),
    )
    context_embedding = tf.random.normal(
        shape=(batch_size, context_seq_len, embed_dim),
        dtype=tf.float32,
    )

    # Decoder.
    decoder = Decoder(
        num_layers=num_layers, d_model=embed_dim,
        num_heads=2, dff=dff, vocab_size=vocab_size,
    )
    output = decoder(target, context_embedding)
    assert output.shape == (batch_size, target_seq_len, embed_dim)
