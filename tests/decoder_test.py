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
    target_seq_len: int,
    context_seq_len: int,
    embed_dim: int,
    dff: int,
) -> None:
    """Test decoder layer's output shape."""

    # Sample input.
    query_embedding = tf.keras.Input(
        shape=(target_seq_len, embed_dim),
        dtype=tf.float32,
    )
    context_embedding = tf.keras.Input(
        shape=(context_seq_len, embed_dim),
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
    num_layers: int,
    target_seq_len: int,
    context_seq_len: int,
    embed_dim: int,
    dff: int,
) -> None:
    """Test decoder's output shape."""
    # Hyperparameters.

    # Sample target.
    target = tf.keras.Input(
        shape=[target_seq_len],
        dtype=tf.int32,
    )
    context_embedding = tf.keras.Input(
        shape=(context_seq_len, embed_dim),
        dtype=tf.float32,
    )

    # Decoder.
    decoder = Decoder(
        num_layers=num_layers, d_model=embed_dim,
        num_heads=2, dff=dff, vocab_size=1_000,
    )
    output = decoder(target, context_embedding)
    expected_shape = tf.TensorShape((None, target_seq_len, embed_dim))
    assert output.shape == expected_shape
