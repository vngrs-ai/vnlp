from typing import List, Tuple

import pickle

import numpy as np

import sentencepiece as spm

from ..tokenizer import TreebankWordTokenize
from ..utils import check_and_download, load_keras_tokenizer
from .utils import ner_to_displacy_format
from ._spu_context_utils import (
    create_spucontext_ner_model,
    process_single_word_input,
)

# Resolving parent dependencies
from inspect import getsourcefile
import os
import sys

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[: current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")

PROD_WEIGHTS_LOC = RESOURCES_PATH + "NER_SPUContext_prod.weights"
EVAL_WEIGHTS_LOC = RESOURCES_PATH + "NER_SPUContext_eval.weights"
WORD_EMBEDDING_MATRIX_LOC = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "resources/SPUTokenized_word_embedding_16k.matrix",
    )
)
PROD_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/NER_SPUContext_prod.weights"
EVAL_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/NER_SPUContext_eval.weights"
WORD_EMBEDDING_MATRIX_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/SPUTokenized_word_embedding_16k.matrix"

SPU_TOKENIZER_WORD_LOC = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "resources/SPU_word_tokenizer_16k.model",
    )
)
TOKENIZER_LABEL_LOC = RESOURCES_PATH + "NER_label_tokenizer.json"

# Data Preprocessing Config
TOKEN_PIECE_MAX_LEN = 8
SENTENCE_MAX_LEN = 40

# Loading Tokenizers
spu_tokenizer_word = spm.SentencePieceProcessor(SPU_TOKENIZER_WORD_LOC)

tokenizer_label = load_keras_tokenizer(TOKENIZER_LABEL_LOC)

sp_key_to_index = {
    spu_tokenizer_word.id_to_piece(id): id
    for id in range(spu_tokenizer_word.get_piece_size())
}
sp_index_to_key = {
    id: spu_tokenizer_word.id_to_piece(id)
    for id in range(spu_tokenizer_word.get_piece_size())
}

LABEL_VOCAB_SIZE = len(tokenizer_label.word_index)
WORD_EMBEDDING_VOCAB_SIZE = len(sp_key_to_index)
WORD_EMBEDDING_VECTOR_SIZE = 128
WORD_EMBEDDING_MATRIX = np.zeros(
    (WORD_EMBEDDING_VOCAB_SIZE, WORD_EMBEDDING_VECTOR_SIZE)
)
NUM_RNN_STACKS = 2
RNN_UNITS_MULTIPLIER = 2
NUM_RNN_UNITS = WORD_EMBEDDING_VECTOR_SIZE * RNN_UNITS_MULTIPLIER
FC_UNITS_MULTIPLIER = (2, 1)
DROPOUT = 0.2


class SPUContextNER:
    """
    SentencePiece Unigram Context Named Entity Recognizer class.

    - This is a context aware Deep GRU based Named Entity Recognizer that uses `SentencePiece Unigram <https://arxiv.org/abs/1804.10959>`_ tokenizer and pre-trained Word2Vec embeddings.
    - It achieves 0.9928 Accuracy and 0.9833 F1 score on test sets of "wikiann", "gungor.ner" and "teghub-TurkishNER-BERT" datasets.
    - It achieves 0.9766 F1 score for "ORG", 0.9852 F1 score for "PER" and 0.9742 F1 score for "LOC" entities. (Treating entity of interest as positive, all others as negative class.)
    - For more details about the training procedure, dataset and evaluation metrics, see `ReadMe <https://github.com/vngrs-ai/VNLP/blob/main/vnlp/named_entity_recognizer/ReadMe.md>`_.
    """

    def __init__(self, evaluate):
        self.model = create_spucontext_ner_model(
            TOKEN_PIECE_MAX_LEN,
            SENTENCE_MAX_LEN,
            WORD_EMBEDDING_VOCAB_SIZE,
            LABEL_VOCAB_SIZE,
            WORD_EMBEDDING_VECTOR_SIZE,
            WORD_EMBEDDING_MATRIX,
            NUM_RNN_UNITS,
            NUM_RNN_STACKS,
            FC_UNITS_MULTIPLIER,
            DROPOUT,
        )
        # Check and download word embedding matrix and model weights
        check_and_download(
            WORD_EMBEDDING_MATRIX_LOC, WORD_EMBEDDING_MATRIX_LINK
        )
        if evaluate:
            MODEL_WEIGHTS_LOC = EVAL_WEIGHTS_LOC
            MODEL_WEIGHTS_LINK = EVAL_WEIGHTS_LINK
        else:
            MODEL_WEIGHTS_LOC = PROD_WEIGHTS_LOC
            MODEL_WEIGHTS_LINK = PROD_WEIGHTS_LINK

        check_and_download(MODEL_WEIGHTS_LOC, MODEL_WEIGHTS_LINK)

        # Load Word embedding matrix
        word_embedding_matrix = np.load(WORD_EMBEDDING_MATRIX_LOC)
        # Load Model weights
        with open(MODEL_WEIGHTS_LOC, "rb") as fp:
            model_weights = pickle.load(fp)
        # Insert word embedding weights to correct position (0 for SPUContextNER model)
        model_weights.insert(0, word_embedding_matrix)
        # Set model weights
        self.model.set_weights(model_weights)

        self.spu_tokenizer_word = spu_tokenizer_word
        self.tokenizer_label = tokenizer_label

    def predict(
        self, sentence: str, displacy_format: bool = False
    ) -> List[Tuple[int, str, int, str]]:
        """
        Args:
            sentence:
                Input sentence/text.
            displacy_format:
                When set True, returns the result in spacy.displacy format to allow visualization.

        Returns:
            NER result as pairs of (token, entity).
        """
        tokenized_sentence = TreebankWordTokenize(sentence)
        num_tokens_in_sentence = len(tokenized_sentence)

        int_preds = []
        for t in range(num_tokens_in_sentence):
            # t is the index of the token/word
            X = process_single_word_input(
                t,
                tokenized_sentence,
                self.spu_tokenizer_word,
                self.tokenizer_label,
                int_preds,
            )
            # Predicting
            logits = self.model(X).numpy()[0]

            int_pred = np.argmax(logits, axis=-1)
            int_preds.append(int_pred)

        preds = [
            self.tokenizer_label.sequences_to_texts([[int_pred]])[0]
            for int_pred in int_preds
        ]
        ner_result = [(t, e) for t, e in zip(tokenized_sentence, preds)]

        if not displacy_format:
            return ner_result
        else:
            return ner_to_displacy_format(sentence, ner_result)
