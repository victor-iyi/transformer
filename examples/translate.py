from typing import Any
from typing import Tuple
from typing import Union

import tensorflow as tf
from transformer import Transformer


class Translator(tf.Module):
    """Translate Portuguese text to English text."""

    def __init__(self, tokenizers: Any, transformer: Transformer) -> None:
        """Initialize a `Translator` from pre-trained tokenizers and
        transformer.

        Arguments:
            tokenizers (object): Pre-trained tokenizer for Portuguese/English.
            transformer (Transformer): The model to predict English text given
                Portuguese text.

        """
        super().__init__()
        self.tokenizers = tokenizers
        self.transformer = transformer

    def __call__(
        self,
        sentence: Union[str, tf.Tensor],
        max_length: int = 128,
    ) -> Tuple[Union[str, tf.Tensor], tf.Tensor, tf.Tensor]:
        """Predict English text given a Porutguese sentence.

        Arguments:
            sentence (str | tf.Tensor): Portuguese sentence.
            max_length (int, optional): Maximum length of sentence to generate.

        Returns:
            Tuple[str | tf.Tensor, tf.Tensor, tf.Tensor] -
               Predicted English text, output tokens, attention weights.

        """
        # The input sentence is Portuguese, hence adding the
        # `[START] and `[END]` tokens.

        # Convert str to tf.Tensor.
        if isinstance(sentence, str):
            sentence = tf.constant(sentence)

        assert isinstance(sentence, tf.Tensor), 'Sentence must be a tf.Tensor'

        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        # Tokenize the sentence. (batch_size, seq_len).
        sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

        encoder_input = sentence

        # As the output language is English, initialize the output with the
        # English `[START]` token.
        start_end = self.tokenizers.en.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a Python list), so that
        # the dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(
            dtype=tf.int64, size=0, dynamic_size=True,
        )
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer(
                (encoder_input, output), training=False,
            )

            # Select the last token from the `seq_len` dimension.
            # Shape: `(batch_size, 1, vocab_size)
            predictions = predictions[:, -1:, :]  # pyright: ignore

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input.
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == end:
                break
        output = tf.transpose(output_array.stack())
        # The output shape is `(1, tokens)`.
        text = self.tokenizers.en.detokenize(output)[0]  # Shape: `()`.

        tokens = self.tokneizers.en.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop.
        # So, recalculate them outside the loop.
        self.transformer([encoder_input, output[:, :-1]], training=False)
        attn_weights = self.transformer.decoder.last_attn_scores

        return text, tokens, attn_weights
