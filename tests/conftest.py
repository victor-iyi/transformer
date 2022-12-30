from typing import Any
from typing import Callable

import pytest
import tensorflow as tf


@pytest.fixture
def input_data() -> Callable[..., tf.Tensor]:
    """Fixture for getting a `KerasInput`."""
    def _input(**kwargs: Any) -> tf.Tensor:
        shape = kwargs.pop('shape', (None,))
        dtype = kwargs.pop('dtype', tf.float32)
        return tf.keras.Input(shape=shape, dtype=dtype)

    return _input


@pytest.fixture
def get_embedding(input_data: Callable[..., tf.Tensor]) -> Callable[..., tf.Tensor]:
    """Fixture for returning an embedding vector."""
    def _embedding_fn(**kwargs: Any) -> tf.Tensor:
        seq_len = kwargs.pop('seq_len', 100)
        embed_dim = kwargs.pop('embed_dim', 512)
        vocab_size = kwargs.pop('vocab_size', 1000)

        embedding_input = input_data(shape=(seq_len,), dtype=tf.int32)
        embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_dim)
        embed = embedding_layer(embedding_input)

        return embed

    return _embedding_fn
