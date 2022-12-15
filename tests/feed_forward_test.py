import pytest
import tensorflow as tf
from transformer.feed_forward import FeedForward


@pytest.mark.parametrize(
    'seq_length,embed_dim',
    (
        (64, 256),
        (512, 128),
    ),
)
def test_feed_forward(seq_length: int, embed_dim: int) -> None:
    """Check the shape of the cross attention layer's output."""
    batch_size = 32

    # Create a dummy embedding.
    embedding = tf.random.normal(
        shape=(batch_size, seq_length, embed_dim),
    )

    # Create cross attention with 2 attention heads.
    sample_ffn = FeedForward(d_model=embed_dim, dff=512)
    output = sample_ffn(embedding)

    assert output.shape == (batch_size, seq_length, embed_dim)
