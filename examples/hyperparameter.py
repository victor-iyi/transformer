import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from data import load_data
from metrics import CustomSchedule
from metrics import masked_accuracy
from metrics import masked_loss
from tensorboard.plugins.hparams import api as hp
from transformer import Transformer

# Hyperparameters.
HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([4, 8]))
HP_D_MODEL = hp.HParam('d_model', hp.Discrete([128, 256]))
HP_DFF = hp.HParam('dff', hp.Discrete([512, 1024]))
HP_NUM_HEADS = hp.HParam('num_heads', hp.Discrete([8, 16]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval([0.1, 0.2]))

BASE_DIR: str = os.path.dirname(os.path.dirname(__file__))
LOG_DIR: str = os.path.join(BASE_DIR, 'logs/hparam_tunning')
EPOCHS: int = 5
BATCH_SIZE: int = 64


def run(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    input_vocab_size: int,
    target_vocab_size: int,
    hparams: hp.HParams,
    run_dir: str,
) -> None:
    """Run training step for different parameter configuration."""
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial.
        epoch_loss, epoch_val_loss = train_test_model(
            train_ds=train_ds,
            val_ds=val_ds,
            input_vocab_size=input_vocab_size,
            target_vocab_size=target_vocab_size,
            hparams=hparams,
        )
        for i, (loss, val_loss) in enumerate(zip(epoch_loss, epoch_val_loss), start=1):
            tf.summary.scalar('loss', loss, step=i)
            tf.summary.scalar('val_loss', val_loss, step=i)


def train_test_model(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    input_vocab_size: int,
    target_vocab_size: int,
    hparams: hp.HParams,
) -> Tuple[np.float32, np.float32]:
    """Run a full training epoch."""
    learning_rate = CustomSchedule(hparams[HP_D_MODEL])
    optimizer = tf.keras.optimizer.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9,
    )

    transformer = Transformer(
        num_layers=hparams[HP_NUM_LAYERS],
        d_model=hparams[HP_D_MODEL],
        num_heads=hparams[HP_NUM_HEADS],
        dff=hparams[HP_DFF],
        dropout=hparams[HP_DROPOUT],
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
    )

    transformer.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[masked_accuracy],
    )

    history = transformer.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[
            tf.keras.callbacks.TensorBoard(LOG_DIR),  # log metrics.
            hp.KerasCallback(LOG_DIR, hparams),  # log hparams.
        ],
    )

    # Metrics over epochs.
    metrics = history.history

    loss, val_loss = metrics['loss'], metrics['val_loss']

    return loss, val_loss


def experiment() -> int:
    """Start experiment to for hyperparameter tunning."""
    # Create summary.
    with tf.summary.create_file_writer(LOG_DIR).as_default():
        hp.hparams_config(
            hparams=[
                HP_NUM_LAYERS, HP_D_MODEL,
                HP_DFF, HP_NUM_HEADS, HP_DROPOUT,
            ],
            metrics=[hp.Metric(masked_accuracy, display_name='Accuracy')],
        )

    # Download and process data.
    (train_ds, val_ds), tokenizers = load_data(batch_size=BATCH_SIZE)

    # Extract the vocabulary sizes.
    input_vocab_size = tokenizers.pt.get_vocab_size().numpy()
    target_vocab_size = tokenizers.en.get_vocab_size().numpy()

    # Start Hyperparameter tunning.
    session = 0
    for num_layers in HP_NUM_LAYERS.domain.values:
        for d_model in HP_D_MODEL.domain.values:
            for dff in HP_DFF.domain.values:
                for num_heads in HP_NUM_HEADS.domain.values:
                    for dropout in (HP_DROPOUT.domain.min_value, HP_DROPOUT.max_value):
                        hparams = {
                            HP_NUM_LAYERS: num_layers,
                            HP_D_MODEL: d_model,
                            HP_DFF: dff,
                            HP_NUM_HEADS: num_heads,
                            HP_DROPOUT: dropout,
                        }
                        run_name = f'run-{session}'
                        print(f'\n--- Starting trial: {run_name}')
                        print({h.name: hparams[h] for h in hparams})
                        run(
                            train_ds=train_ds, val_ds=val_ds,
                            input_vocab_size=input_vocab_size,
                            target_vocab_size=target_vocab_size,
                            run_dir=f'{LOG_DIR}/{run_name}', hparams=hparams,
                        )
                        session += 1

    print(f'Done after {session:,} sessions')

    return 0


if __name__ == '__main__':
    raise SystemExit(experiment())
