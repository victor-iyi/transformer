from typing import Any
from typing import Tuple
from typing import Union

import pytest
import tensorflow as tf
from data import load_data
from data import load_tokenizer


@pytest.fixture
def tokenizer() -> Any:
    """Fixture to load Portuguese/English tokenizers."""
    return load_tokenizer()


@pytest.fixture
def data() -> Union[
    Tuple[Tuple[tf.data.Dataset, tf.data.Dataset], Any],
    Tuple[tf.data.Dataset, tf.data.Dataset],
]:
    """Fixture to load the train & validation dataset with tokenizers."""
    # Download and process data.
    (train_ds, val_ds), tokenizer = load_data()
    return (train_ds, val_ds), tokenizer


@pytest.mark.skip(reason='Untested behaviour')
def test_load_data() -> None:
    """Download and load dataset."""
    # Download and load dataset.
    batch_size = 64
    expected_pt_seq, expected_en_seq = 71, 78

    # Download and process data.
    (train_ds, val_ds), tokenizers = load_data(batch_size=batch_size)

    # Extract the vocabulary sizes.
    input_vocab_size = tokenizers.pt.get_vocab_size().numpy()
    target_vocab_size = tokenizers.en.get_vocab_size().numpy()

    # Check the vocab size.
    assert input_vocab_size == 7765
    assert target_vocab_size == 7010

    # Test train batches.
    for (pt, en), en_labels in train_ds.take(1):

        assert pt.shape == (batch_size, expected_pt_seq)
        assert en.shape == (batch_size, expected_en_seq)
        assert en_labels.shape == (batch_size, expected_en_seq)

        break

    # Test validation batches.
    for (pt, en), en_labels in val_ds.take(1):

        assert pt.shape == (batch_size, expected_pt_seq)
        assert en.shape == (batch_size, expected_en_seq)
        assert en_labels.shape == (batch_size, expected_en_seq)

        break
