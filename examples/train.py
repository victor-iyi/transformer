# Copyright (c) 2022 Victor I. Afolabi
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import os

import tensorflow as tf
from data import load_data
from metrics import CustomSchedule
from metrics import masked_accuracy
from metrics import masked_loss
from transformer import Transformer

# Tensorboard log dir.
BASE_DIR: str = os.path.dirname(os.path.dirname(__file__))
LOG_DIR: str = os.path.join(BASE_DIR, 'logs')


def train() -> int:
    """Train the transformer model for translation task."""
    # Hyperparameters.
    d_model, dff = 128, 512
    num_layers, num_heads = 4, 8
    batch_size, epochs, dropout = 64, 10, 0.1

    # Download and process data.
    (train_ds, val_ds), tokenizers = load_data(batch_size=batch_size)

    # Extract the vocabulary sizes.
    input_vocab_size = tokenizers.pt.get_vocab_size().numpy()
    target_vocab_size = tokenizers.en.get_vocab_size().numpy()

    # Optimizer.
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizer.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9,
    )

    # Define the model.
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        dropout=dropout,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
    )

    # Compile model with optimizer, loss & metrics.
    transformer.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy],
    )

    # Train the model.
    transformer.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
        ],
    )

    return 0


if __name__ == '__main__':
    raise SystemExit(train())
