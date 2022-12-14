import tensorflow as tf
from transformer.utils import positional_encoding


class PositionalEmbedding(tf.keras.layers.Layer):
    def __int__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, d_model, mask_zero=True,
        )
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args: tf.Tensor, **kwargs: tf.Tenosr) -> tf.Tensor | None:
        """Computes an output mask tensor.

        Arguments:
          input (tf.Tensor): Tensor or list of Tensors.
          mask (tf.Tensor): Tensor or list of Tensors.

        Returns:
          tf.Tensor - None or a list of Tensor (one per output tensor of the layer).
        """
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        length = self.shape(x)[1]
        x = self.embedding(x)

        # This factor sets the relative scale of the embedding and positional_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]

        return x
