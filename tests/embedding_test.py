import numpy as np
import pytest
from transformer.embedding import PositionalEmbedding
from transformer.utils import positional_encoding


@pytest.mark.parametrize(
    'seq_length,embed_dim',
    (
        (32, 128),
        (64, 512),
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
        (64, 78, 256),
        (32, 128, 512),
    ),
)
def test_positional_embedding(batch_size: int, seq_length: int, embed_dim: int) -> None:
    """Positional embedding outputs the correct shape."""
    vocab_size = 1_000
    # Random fake data.
    data = np.random.randint(vocab_size, size=(batch_size, seq_length))
    # Create positional encoding with vocab size of 10k.
    embed = PositionalEmbedding(vocab_size=vocab_size, d_model=embed_dim)

    assert embed(data).shape == (batch_size, seq_length, embed_dim)
