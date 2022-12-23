import pytest
import tensorflow as tf
from transformer.embedding import PositionalEmbedding
from transformer.utils import positional_encoding


@pytest.mark.parametrize(
    'seq_len, embed_dim',
    (
        (32, 128),
        (64, 512),
    ),
)
def test_positional_encoding(
    seq_len: int,
    embed_dim: int,
) -> None:
    """Positional encoding outputs the correct shape."""
    # Calculate the positional encoding given the sequence length and embedding dimension.
    pos_encoding = positional_encoding(length=seq_len, depth=embed_dim)
    assert pos_encoding.shape == (seq_len, embed_dim)


@pytest.mark.parametrize(
    'seq_len, embed_dim',
    (
        (78, 256),
        (128, 512),
    ),
)
def test_positional_embedding(
    input_data,
    seq_len: int,
    embed_dim: int,
) -> None:
    """Positional embedding outputs the correct shape."""

    # Create input of shape (None, seq_len).
    data = input_data(shape=(seq_len,))

    # Create positional encoding with vocab size of 1k.
    embed = PositionalEmbedding(vocab_size=1_000, d_model=embed_dim)

    # Expected output shape.
    expected_shape = tf.TensorShape((None, seq_len, embed_dim))

    assert embed(data).shape == expected_shape
