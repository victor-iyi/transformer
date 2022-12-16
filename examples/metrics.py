import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate scheduler according to the formula from the original
    Transformer paper."""

    def __init__(self, d_model: int, warmup_steps: int = 4000) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> float:
        """Custom scheduler."""

        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return float(tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2))


def masked_loss(label: tf.Tensor, pred: tf.Tensor) -> float:
    """Apply mask padding to loss.

    Arguments:
        label (tf.Tensor): Ground truth.
        pred (tf.Tensor): Predicted output.

    Returns:
        float - Masked loss.
    """
    mask = label != 0
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none',
    )
    loss = loss_obj(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return float(loss)


def masked_accuracy(label: tf.Tensor, pred: tf.Tensor) -> float:
    """Masked accuracy.

    Arguments:
        label (tf.Tensor): Ground truth.
        pred (tf.Tensor): Predicted output.

    Returns:
        float - Masked accuracy.
    """
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)

    match = label == pred
    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    return float(tf.reduce_sum(match) / tf.reduce_sum(mask))
