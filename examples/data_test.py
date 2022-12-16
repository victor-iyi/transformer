import pytest
from data import load_data


@pytest.mark.skip(reason='Untested behaviour')
def test_load_data() -> None:
    """Download and load dataset."""
    # Download and load dataset.
    batch_size = 64
    expected_pt_seq, expected_en_seq = 71, 78

    # Download and process data.
    (
        (train_ds, val_ds),
        (input_vocab_size, target_vocab_size),
    ) = load_data(batch_size=batch_size)

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
