import pytest
import tensorflow as tf
from transformer.encoder import EncoderLayer


@pytest.mark.parametrize(
    'embed_dim,dff',
    (
        (512, 2048),
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
