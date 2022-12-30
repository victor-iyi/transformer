from typing import Callable

import pytest
import tensorflow as tf
from transformer.transformer import Transformer


@pytest.mark.parametrize(
    'input_vocab_size, target_vocab_size, input_seq_len, target_seq_len',
    (
        (500, 500, 100, 200),
        (100, 100, 60, 80),
    ),
)
def test_transformer(
    input_data: Callable[..., tf.Tensor],
    input_vocab_size: int,
    target_vocab_size: int,
    input_seq_len: int,
    target_seq_len: int,
) -> None:
    """Test transformer output shape."""
    # Hyperparameters.
    num_layers, num_heads, d_model, dff = 3, 3, 256, 512

    input_token = input_data(shape=(input_seq_len,))
    target_token = input_data(shape=(target_seq_len,))

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        dropout=0.2,
    )
    output = transformer((target_token, input_token))

    # Expected output shape.
    expected_shape = tf.TensorShape((None, input_seq_len, target_vocab_size))
    assert output.shape == expected_shape
