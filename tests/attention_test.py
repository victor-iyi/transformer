from typing import Callable

import pytest
import tensorflow as tf
from transformer.attention import CausalSelfAttention
from transformer.attention import CrossAttention
from transformer.attention import GlobalSelfAttention


@pytest.mark.parametrize(
    'query_length, context_length, embed_dim',
    (
        (71, 78, 128),
        (128, 128, 512),
    ),
)
def test_cross_attention(
    get_embedding: Callable[..., tf.Tensor],
    query_length: int,
    context_length: int,
    embed_dim: int,
) -> None:
    """Check the shape of the cross attention layer's output."""
    # vocab_size = 1000

    # Input query (without PositionalEncoding).
    query_embed = get_embedding(
        seq_len=query_length,
        embed_dim=embed_dim,
    )

    # Context query embedding (without PositionalEncoding).
    context_embed = get_embedding(
        seq_len=context_length,
        embed_dim=embed_dim,
    )

    # Create cross attention with 2 attention heads.
    cross_attn = CrossAttention(num_heads=2, key_dim=embed_dim)
    output = cross_attn(query=query_embed, context=context_embed)

    # Expected output shape.
    expected_shape = tf.TensorShape((None, query_length, embed_dim))
    assert output.shape == expected_shape


@pytest.mark.parametrize(
    'seq_len, embed_dim',
    (
        (74, 512),
        (128, 1024),
    ),
)
def test_global_attention(
    get_embedding: Callable[..., tf.Tensor],
    seq_len: int,
    embed_dim: int,
) -> None:
    """Check the shape of the self attention layer's output."""
    # Create dummy embedding.
    query_embed = get_embedding(
        seq_len=seq_len,
        embed_dim=embed_dim,
    )

    # Create self attention layer with 2 attention heads.
    gsa = GlobalSelfAttention(num_heads=2, key_dim=embed_dim)
    output = gsa(query=query_embed)

    # Expected output shape.
    expected_shape = tf.TensorShape((None, seq_len, embed_dim))
    assert output.shape == expected_shape


@pytest.mark.parametrize(
    'seq_len, embed_dim',
    (
        (64, 256),
        (512, 128),
    ),
)
def test_casual_attention(
    get_embedding: Callable[..., tf.Tensor],
    seq_len: int,
    embed_dim: int,
) -> None:
    """Test casual self-attention layer's output shape."""
    # Create a dummy embedding.
    query_embed = get_embedding(
        seq_len=seq_len,
        embed_dim=embed_dim,
    )

    # Create casual self attention layer with 2 attention heads.
    csa = CausalSelfAttention(num_heads=2, key_dim=embed_dim)
    output = csa(query=query_embed)

    # Expected output shape.
    expected_shape = tf.TensorShape((None, seq_len, embed_dim))
    assert output.shape == expected_shape
