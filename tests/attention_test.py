import pytest
import tensorflow as tf
from transformer.attention import CrossAttention
from transformer.attention import GlobalSelfAttention


@pytest.mark.parametrize(
    'query_length,context_length,embed_dim',
    (
        (71, 78, 128),
        (128, 128, 512),
    ),
)
def test_cross_attention(query_length: int, context_length: int, embed_dim: int) -> None:
    """Check the shape of the cross attention layer's output."""
    batch_size = 32

    # Create dummy query embedding vector.
    query_embedding = tf.random.normal(
        shape=(batch_size, query_length, embed_dim),
        dtype=tf.float32,
    )
    # Create dummy context embedding vector.
    context_embedding = tf.random.normal(
        shape=(batch_size, context_length, embed_dim),
        dtype=tf.float32,
    )

    # Create cross attention with 2 attention heads.
    cross_attn = CrossAttention(num_heads=2, key_dim=embed_dim)
    output = cross_attn(query=query_embedding, context=context_embedding)

    assert output.shape == (batch_size, query_length, embed_dim)


@pytest.mark.parametrize(
    'seq_length,embed_dim',
    (
        (74, 512),
        (128, 1024),
    ),
)
def test_global_attention(seq_length: int, embed_dim: int) -> None:
    """Check the shape of the self attention layer's output."""
    batch_size = 32
    # Create dummy embedding.
    embedding = tf.random.normal((batch_size, seq_length, embed_dim))

    # Create self attention layer with 2 attention heads.
    gsa = GlobalSelfAttention(num_heads=2, key_dim=embed_dim)
    output = gsa(query=embedding)

    assert output.shape == (batch_size, seq_length, embed_dim)
