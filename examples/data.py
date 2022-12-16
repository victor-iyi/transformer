# Copyright (c) 2022 Victor I. Afolabi
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os
from typing import Any
from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

# Data path.
BASE_DIR: str = os.path.dirname(os.path.dirname(__file__))


def load_tokenizer(
    tokenizer_name: str = 'ted_hrlr_translate_pt_en_converter',
    cache_dir: str = 'data', cache_sub_dir: str = 'translate',
) -> Any:
    """Load tokenizer from tensorflow hub.

    Arguments:
        tokenizer_name (str, optional): Name of the tokenizer.
            Defaults to `'ted_hrlr_translate_pt_en_converter'`.
        cache_dir (str, optional): Base directory to download tokenizer.
            Defaults to `data/`.
        cache_sub_dir (str, optional): Sub directory. Usually a child of `cahce_dir`.
            Defaults to `translate/` i.e `{cache_dir/cache_sub_dir}`.

    Returns:

    """
    # Tokenizer path.
    origin = f'https://storage.googleapis.com/download.tensorflow.org/models/{tokenizer_name}.zip'

    # Downlaod the tokenizer.
    tokenizer_path = tf.keras.utils.get_file(
        fname=f'{tokenizer_name}.zip', origin=origin,
        cache_dir=cache_dir, cache_subdir=cache_sub_dir, extract=True,
    )

    tokenizer_path = os.path.join(
        os.path.dirname(tokenizer_path), tokenizer_name,
    )
    print(f'Loading model from {tokenizer_path}')

    # Load tokenizer.
    tokenizers = tf.saved_model.load(tokenizer_path)

    return tokenizers


def load_data(
    name: str = 'ted_hrlr_translate/pt_to_en',
    tokenizer_name: str = 'ted_hrlr_translate_pt_en_converter',
    cache_dir: str = 'data', cache_sub_dir: str = 'translate',
    max_tokens: int = 128, batch_size: int = 64, buffer_size: int = 20_000,
) -> Tuple[Tuple[tf.data.Dataset, tf.data.Dataset], Any]:
    """Download, tokenize and prepare data into train & validation batches.

    Arguments:
        name (str, optional): Name of dataset from tensorflow-datasets.
            Defaults to `'ted_hrlr_translate/pt_to_en'`.
        tokenizer_name (str, optional): Name of the tokenizer.
            Defaults to `'ted_hrlr_translate_pt_en_converter'`.
        cache_dir (str, optional): Base directory to download tokenizer.
            Defaults to `data/`.
        cache_sub_dir (str, optional): Sub directory. Usually a child of `cahce_dir`.
            Defaults to `translate/` i.e `{cache_dir/cache_sub_dir}`.
        max_tokens (int, optional): Maximum length of tokens.
            Defaults to 128.
        batch_size (int, optiona): Mini batch size.
            Defaults to 64.
        buffer_size (int, optional): Buffer size.
            Defaults to 20,000.

    Returns:
        ((tf.data.Dataset, tf.data.Dataset), (int, int)):
            (Train dataset, validation dataset), (input_vocab_size, target_vocab_size)
    """
    # Load dataset from tensorflow-datasets.
    examples, _ = tfds.load(name, with_info=True, as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    # Load tokenizer.
    tokenizers = load_tokenizer(
        tokenizer_name=tokenizer_name,
        cache_dir=cache_dir,
        cache_sub_dir=cache_sub_dir,
    )

    def prepare_batch(
        pt: tf.Tensor, en: tf.Tensor,
    ) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        """Prepare the Portuguese/English language pair.

        Arguments:
            pt (tf.Tensor): Portuguese examples.
            en (tf.Tensor): English examples.

        Returns:
            Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
                The portuguese tokens and the english token pair, and the
                english labels shifted by one.
        """
        pt = tokenizers.pt.tokenize(pt)  # Output is ragged.
        pt = pt[:, :max_tokens]  # Trim to max_token.
        pt_tokens = pt.to_tensor()  # Convert to 0-padded dense Tensor

        en = tokenizers.en.tokenize(en)
        en = en[:, :(max_tokens + 1)]
        en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens.
        en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens.

        return (pt_tokens, en_inputs), en_labels

    def make_batches(ds: tf.data.Dataset) -> tf.data.Dataset:
        return (
            ds
            .shuffle(buffer_size)
            .batch(batch_size)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

    # Make train & validation mini batches.
    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)

    return (train_batches, val_batches), tokenizers


if __name__ == '__main__':
    # Load train & validation data.
    (train_batches, val_batches), tokenizers = load_data()

    # Extract the vocabulary sizes.
    input_vocab_size = tokenizers.pt.get_vocab_size().numpy()
    target_vocab_size = tokenizers.en.get_vocab_size().numpy()

    print(f'Input vocab size: {input_vocab_size:,}')
    print(f'Target vocab size: {target_vocab_size}')

    print(f'{train_batches = }')
    print(f'{val_batches = }')

    train_ds = train_batches.take(1)
    val_ds = val_batches.take(1)

    print('\nTrain data:')
    print(list(train_ds.as_numpy_iterator()))
    print('\nValidation data:')
    print(list(val_ds.as_numpy_iterator()))
