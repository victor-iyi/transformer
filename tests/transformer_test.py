import numpy as np
import pytest
from transformer.transformer import Transformer


@pytest.mark.parametrize(
    'input_vocab_size,target_vocab_size,input_seq_len,target_seq_len',
    (
        (500, 500, 100, 200),
        (100, 100, 60, 80),
    ),
)
def test_transformer(
        input_vocab_size: int,
        target_vocab_size: int,
        input_seq_len: int,
        target_seq_len: int,
) -> None:
    """Test transformer output shape."""
    batch_size, num_layers, num_heads, d_model, dff = 32, 3, 3, 256, 512

    input_token = np.random.randint(
        input_vocab_size,
        size=(batch_size, input_seq_len),
    )
    target_token = np.random.randint(
        target_vocab_size,
        size=(batch_size, target_seq_len),
    )

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
    assert output.shape == (batch_size, input_seq_len, target_vocab_size)
