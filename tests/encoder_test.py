import numpy as np
import pytest
import tensorflow as tf
from transformer.encoder import Encoder
from transformer.encoder import EncoderLayer


@pytest.mark.parametrize(
    'embed_dim,dff',
    (
        (32, 64),
        (128, 256),
    ),
)
def test_encoder_layer(embed_dim: int, dff: int) -> None:
    """Test encoder layer's output shape."""
    # Hyperparameters.
    batch_size, seq_length = 32, 78

    # Sample embedding.
    embed_shape = (batch_size, seq_length, embed_dim)
    embedding = tf.random.normal(shape=embed_shape)

    encoder_layer = EncoderLayer(d_model=embed_dim, num_heads=2, dff=dff)
    assert encoder_layer(embedding).shape == embed_shape


@pytest.mark.parametrize(
    'num_layers,seq_len,embed_dim,dff',
    (
        (1, 64, 32, 256),
        (3, 82, 64, 256),
    ),
)
def test_encoder(
    num_layers: int,
    seq_len: int,
    embed_dim: int,
    dff: int,
) -> None:
    """Test encoder's output shape."""
    batch_size, num_heads, vocab_size = 32, 8, 1_000

    # Sample data (dummy token IDs)
    data = np.random.randint(vocab_size, size=(batch_size, seq_len))

    # Multiple encoder layers.
    encoder = Encoder(
        num_layers=num_layers,
        d_model=embed_dim,
        num_heads=num_heads,
        dff=dff,
        vocab_size=vocab_size,
    )
    output = encoder(data, training=False)
    assert output.shape == (batch_size, seq_len, embed_dim)
