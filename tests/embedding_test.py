import pytest
from transformer.utils import positional_encoding


@pytest.mark.parametrize(
    'length,depth',
    [
        (2048, 512),
        (2048, 128),
    ],
)
def test_positional_encoding(length: int, depth: int) -> None:
    pos_encoding = positional_encoding(length=length, depth=depth)
    assert pos_encoding.shape == (length, depth)
