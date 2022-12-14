import pytest
import tensorflow as tf
from transformer.embedding import PositionalEmbedding
from transformer.utils import positional_encoding


@pytest.mark.parametrize(
    'seq_length,embed_dim',
    (
        (1024, 128),
        (2048, 512),
    ),
)
def test_positional_encoding(seq_length: int, embed_dim: int) -> None:
    """Positional encoding outputs the correct shape."""
    # Calculate the positional encoding given the sequence length and embedding dimension.
    pos_encoding = positional_encoding(length=seq_length, depth=embed_dim)
    assert pos_encoding.shape == (seq_length, embed_dim)


@pytest.mark.parametrize(
    'batch_size,seq_length,embed_dim',
    (
        (64, 78, 512),
        (32, 256, 4096),
    ),
)
def test_positional_embedding(batch_size: int, seq_length: int, embed_dim: int) -> None:
    """Positional embedding outputs the correct shape."""
    # Random fake data.
    data = tf.random.normal(shape=(batch_size, seq_length))

    # Create positional encoding with vocab size of 10k.
    embed = PositionalEmbedding(vocab_size=10_000, d_model=embed_dim)

    assert embed(data).shape == (batch_size, seq_length, embed_dim)
