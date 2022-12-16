import pytest
import tensorflow as tf
from data import load_tokenizer
from transformer import Transformer
from translate import Translator


@pytest.mark.skip(reason='Not yet tested.')
def test_translate() -> None:
    tokenizers = load_tokenizer()
    transformer = Transformer(
        num_layers=2,
        num_heads=4,
        d_model=512,
        dff=512,
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    )
    translator = Translator(tokenizers=tokenizers, transformer=transformer)

    sentence = 'este é um problema que temos que resolver.'
    ground_truth = 'this is a problem we have to solve .'

    text, tokens, _ = translator(sentence=tf.constant(sentence))
    print_translation(text, tokens, ground_truth)


def print_translation(
    sentence: str, tokens: tf.Tensor, ground_truth: str,
) -> None:
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction:":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')


if __name__ == '__main__':
    test_translate()
