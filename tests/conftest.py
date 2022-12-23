import pytest
import tensorflow as tf


@pytest.fixture
def input_data():
    """Fixture for getting a `KerasInput`."""
    def _input(**kwargs):
        shape = kwargs.pop('shape', (None,))
        dtype = kwargs.pop('dtype', tf.float32)
        return tf.keras.Input(shape=shape, dtype=dtype)

    return _input


@pytest.fixture
def get_embedding(input_data):
    """Fixture for returning an embedding vector."""
    def _embedding_fn(**kwargs):
        seq_len = kwargs.pop('seq_len', 100)
        embed_dim = kwargs.pop('embed_dim', 512)
        vocab_size = kwargs.pop('vocab_size', 100)

        print('This is an embedding fixture and it works!')
        # vocab_size, embed_dim, seq_len = 1000, 512, 100

        embedding_input = input_data(shape=(seq_len,), dtype=tf.int32)
        # embedding_input = tf.keras.Input(shape=(seq_len,), dtype=tf.int32)
        embedding_layer = tf.keras.layers.Embedding(vocab_size, embed_dim)
        embed = embedding_layer(embedding_input)

        return embed

    return _embedding_fn


# @pytest.fixture
# def pos_embedding(input_data):
#     def _pos_embedding(**kwargs):
#         shape = kwargs.pop('shape', ())
#         dtype = kwargs.pop('dtype', tf.float32)
#         input_data(kwargs)
#     embed_input = input_data(shape=shape, dtype=tf.int32)
