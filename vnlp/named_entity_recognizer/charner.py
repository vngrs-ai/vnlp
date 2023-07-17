from typing import List, Tuple

import pickle

import numpy as np
import tensorflow as tf

from ..tokenizer import WordPunctTokenize
from ..utils import check_and_download, load_keras_tokenizer
from .utils import ner_to_displacy_format
from ._charner_utils import create_charner_model

# Resolving parent dependencies
from inspect import getsourcefile
import os
import sys

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[: current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")

PROD_WEIGHTS_LOC = RESOURCES_PATH + "NER_CharNER_prod.weights"
EVAL_WEIGHTS_LOC = RESOURCES_PATH + "NER_CharNER_eval.weights"

PROD_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/NER_CharNER_prod.weights"
EVAL_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/NER_CharNER_eval.weights"

TOKENIZER_CHAR_LOC = RESOURCES_PATH + "CharNER_char_tokenizer.json"
TOKENIZER_LABEL_LOC = RESOURCES_PATH + "NER_label_tokenizer.json"

CHAR_VOCAB_SIZE = 150
SEQ_LEN_MAX = 256
OOV_TOKEN = "<OOV>"
PADDING_STRAT = "post"

EMBED_SIZE = 32
RNN_DIM = 128
NUM_RNN_STACKS = 5
MLP_DIM = 32
NUM_CLASSES = 5  # Equals to len(tokenizer_label.index_word) + 1. +1 is reserved to 0, which corresponds to padded values.
DROPOUT = 0.3


class CharNER:
    """
    CharNER Named Entity Recognizer.

    - This is an implementation of `CharNER: Character-Level Named Entity Recognition <https://aclanthology.org/C16-1087/>`_.
    - There are slight modifications to the original paper:
    - This version is trained for Turkish language only.
    - This version uses simple Mode operation among the character predictions of each token, instead of Viterbi Decoder
    - It achieves 0.9589 Accuracy and 0.9200 F1_macro_score.
    - Input data is processed by NLTK.tokenize.WordPunctTokenizer so that each punctuation becomes a new token.
    - Entity labels are: ['O', 'PER', 'LOC', 'ORG']
    - For more details about the training procedure, dataset and evaluation metrics, see `ReadMe <https://github.com/vngrs-ai/VNLP/blob/main/vnlp/named_entity_recognizer/ReadMe.md>`_.
    """

    def __init__(self, evaluate):
        self.model = create_charner_model(
            CHAR_VOCAB_SIZE,
            EMBED_SIZE,
            SEQ_LEN_MAX,
            NUM_RNN_STACKS,
            RNN_DIM,
            MLP_DIM,
            NUM_CLASSES,
            DROPOUT,
        )
        # Check and download model weights
        if evaluate:
            MODEL_WEIGHTS_LOC = EVAL_WEIGHTS_LOC
            MODEL_WEIGHTS_LINK = EVAL_WEIGHTS_LINK
        else:
            MODEL_WEIGHTS_LOC = PROD_WEIGHTS_LOC
            MODEL_WEIGHTS_LINK = PROD_WEIGHTS_LINK

        check_and_download(MODEL_WEIGHTS_LOC, MODEL_WEIGHTS_LINK)

        # Load Model weights
        with open(MODEL_WEIGHTS_LOC, "rb") as fp:
            model_weights = pickle.load(fp)

        # Set model weights
        self.model.set_weights(model_weights)

        tokenizer_char = load_keras_tokenizer(TOKENIZER_CHAR_LOC)
        tokenizer_label = load_keras_tokenizer(TOKENIZER_LABEL_LOC)

        self.tokenizer_char = tokenizer_char
        self.tokenizer_label = tokenizer_label

    def _predict_char_level(
        self, word_punct_tokenized: List[str]
    ) -> List[int]:
        """
        Returns char level predictions in integers, which will be passed to decoder.

        Args:
            word_punct_tokenized:
                List of tokens, tokenized by WordPunctTokenizer.

        Returns:
            List of integers, indicating entity classes for each character.
        """
        white_space_joined_word_punct_tokens = " ".join(word_punct_tokenized)
        white_space_joined_word_punct_tokens = [
            char for char in white_space_joined_word_punct_tokens
        ]
        sequences = self.tokenizer_char.texts_to_sequences(
            [white_space_joined_word_punct_tokens]
        )
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=SEQ_LEN_MAX, padding=PADDING_STRAT
        )
        raw_pred = self.model([padded]).numpy()
        arg_max_pred = np.argmax(raw_pred, axis=2).reshape(-1)

        return arg_max_pred

    def _charner_decoder(
        self, word_punct_tokenized: List[str], arg_max_pred: List[int]
    ) -> List[str]:
        """
        Args:
            word_punct_tokenized:
                List of tokens, tokenized by WordPunctTokenizer.
            arg_max_pred:
                List of integers, indicating entity classes for each character.

        Returns:
            decoded_entities: List of entities, one entity per token.
        """

        lens = [0] + [len(token) + 1 for token in word_punct_tokenized]
        cumsum_of_lens = np.cumsum(lens)

        decoded_entities = []
        for idx in range(len(cumsum_of_lens) - 1):
            lower_bound = cumsum_of_lens[idx]
            upper_bound = (
                cumsum_of_lens[idx + 1] - 1
            )  # minus one prevents including the whitespace after the token

            island = arg_max_pred[lower_bound:upper_bound]
            # Extracting mode value
            vals, counts = np.unique(island, return_counts=True)
            mode_value = vals[np.argmax(counts)]

            detokenized_pred = self.tokenizer_label.sequences_to_texts(
                [[mode_value]]
            )[0]
            decoded_entities.append(detokenized_pred)

        return decoded_entities

    def predict(
        self, text: str, displacy_format: bool = False
    ) -> List[Tuple[str, str]]:
        """
        Args:
            text:
                Input text.
            displacy_format:
                When set True, returns the result in spacy.displacy format to allow visualization.

        Returns:
            NER result as pairs of (token, entity).
        """
        word_punct_tokenized = WordPunctTokenize(text)

        # if len chars (including whitespaces) > sequence length, split it recursively
        len_text = len(list(" ".join(word_punct_tokenized)))
        if len_text > SEQ_LEN_MAX:
            num_tokens = len(word_punct_tokenized)

            first_half_result = self.predict(
                " ".join(word_punct_tokenized[: num_tokens // 2])
            )
            first_half_tokens = [pair[0] for pair in first_half_result]
            first_half_entities = [pair[1] for pair in first_half_result]

            second_half_result = self.predict(
                " ".join(word_punct_tokenized[(num_tokens // 2) :])
            )
            second_half_tokens = [pair[0] for pair in second_half_result]
            second_half_entities = [pair[1] for pair in second_half_result]

            word_punct_tokenized = first_half_tokens + second_half_tokens
            decoded_entities = first_half_entities + second_half_entities

        else:
            charlevel_pred = self._predict_char_level(word_punct_tokenized)
            decoded_entities = self._charner_decoder(
                word_punct_tokenized, charlevel_pred
            )

        ner_result = [
            (t, e) for t, e in zip(word_punct_tokenized, decoded_entities)
        ]

        if not displacy_format:
            return ner_result
        else:
            return ner_to_displacy_format(text, ner_result)
