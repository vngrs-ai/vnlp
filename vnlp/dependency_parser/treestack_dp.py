from typing import List, Tuple

import pickle

import tensorflow as tf
import numpy as np

from ..stemmer_morph_analyzer import StemmerAnalyzer
from ..part_of_speech_tagger import PoSTagger
from ..tokenizer import TreebankWordTokenize
from ..utils import check_and_download, load_keras_tokenizer
from .utils import dp_pos_to_displacy_format, decode_arc_label_vector
from ._treestack_utils import (
    create_dependency_parser_model,
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

PROD_WEIGHTS_LOC = RESOURCES_PATH + "DP_TreeStack_prod.weights"
EVAL_WEIGHTS_LOC = RESOURCES_PATH + "DP_TreeStack_eval.weights"
WORD_EMBEDDING_MATRIX_LOC = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "resources/TBWTokenized_word_embedding.matrix",
    )
)

PROD_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/DP_TreeStack_prod.weights"
EVAL_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/DP_TreeStack_eval.weights"
WORD_EMBEDDING_MATRIX_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/TBWTokenized_word_embedding.matrix"

TOKENIZER_WORD_LOC = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "resources/TB_word_tokenizer.json"
    )
)
TOKENIZER_POS_LOC = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "part_of_speech_tagger/resources/PoS_label_tokenizer.json",
    )
)  # using the tokenizer of part_of_speech_tagger
TOKENIZER_TAG_LOC = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "stemmer_morph_analyzer/resources/Stemmer_morph_tag_tokenizer.json",
    )
)  # using the tokenizer of stemmer_morph_analyzer
TOKENIZER_LABEL_LOC = RESOURCES_PATH + "DP_label_tokenizer.json"

# Data Preprocessing Config
SENTENCE_MAX_LEN = 40
TAG_MAX_LEN = 15

WORD_OOV_TOKEN = "<OOV>"

# Loading Tokenizers
# Have to load tokenizers here because model config depends on them
tokenizer_word = load_keras_tokenizer(TOKENIZER_WORD_LOC)
tokenizer_pos = load_keras_tokenizer(TOKENIZER_POS_LOC)
tokenizer_tag = load_keras_tokenizer(TOKENIZER_TAG_LOC)
tokenizer_label = load_keras_tokenizer(TOKENIZER_LABEL_LOC)

LABEL_VOCAB_SIZE = len(tokenizer_label.word_index)
POS_VOCAB_SIZE = len(tokenizer_pos.word_index)

# Model Config
WORD_EMBEDDING_VECTOR_SIZE = 128  # Word2Vec_medium.model
WORD_EMBEDDING_VOCAB_SIZE = 63_992  # Word2Vec_medium.model
# WORD_EMBEDDING_MATRIX and TAG_EMBEDDING MATRIX are initialized as Zeros, will be overwritten when model is loaded.
WORD_EMBEDDING_MATRIX = np.zeros(
    (WORD_EMBEDDING_VOCAB_SIZE, WORD_EMBEDDING_VECTOR_SIZE)
)
TAG_EMBEDDING_MATRIX = np.zeros((127, 32))
POS_EMBEDDING_VECTOR_SIZE = 8

NUM_RNN_STACKS = 2
RNN_UNITS_MULTIPLIER = 3
TAG_NUM_RNN_UNITS = WORD_EMBEDDING_VECTOR_SIZE
LC_NUM_RNN_UNITS = TAG_NUM_RNN_UNITS * RNN_UNITS_MULTIPLIER
LC_ARC_LABEL_NUM_RNN_UNITS = TAG_NUM_RNN_UNITS * RNN_UNITS_MULTIPLIER
RC_NUM_RNN_UNITS = TAG_NUM_RNN_UNITS * RNN_UNITS_MULTIPLIER
ARC_LABEL_VECTOR_LEN = SENTENCE_MAX_LEN + 1 + LABEL_VOCAB_SIZE + 1
FC_UNITS_MULTIPLIERS = (8, 4)
WORD_FORM = "whole"
DROPOUT = 0.2


class TreeStackDP:
    """
    Tree-stack Dependency Parser class.

    - This dependency parser is *inspired* by `Tree-stack LSTM in Transition Based Dependency Parsing <https://aclanthology.org/K18-2012/>`_.
    - "Inspire" is emphasized because this implementation uses the approach of using Morphological Tags, Pre-trained word embeddings and POS tags as input for the model, rather than implementing the exact network proposed in the paper.
    - It achieves 0.6914 LAS (Labeled Attachment Score) and 0.8048 UAS (Unlabeled Attachment Score) on all of test sets of Universal Dependencies 2.9.
    - Input data is processed by NLTK.tokenize.TreebankWordTokenizer.
    - For more details about the training procedure, dataset and evaluation metrics, see `ReadMe <https://github.com/vngrs-ai/VNLP/blob/main/vnlp/dependency_parser/ReadMe.md>`_.

    """

    def __init__(self, evaluate):
        self.model = create_dependency_parser_model(
            WORD_EMBEDDING_VOCAB_SIZE,
            WORD_EMBEDDING_VECTOR_SIZE,
            WORD_EMBEDDING_MATRIX,
            POS_VOCAB_SIZE,
            POS_EMBEDDING_VECTOR_SIZE,
            SENTENCE_MAX_LEN,
            TAG_MAX_LEN,
            ARC_LABEL_VECTOR_LEN,
            NUM_RNN_STACKS,
            TAG_NUM_RNN_UNITS,
            LC_NUM_RNN_UNITS,
            LC_ARC_LABEL_NUM_RNN_UNITS,
            RC_NUM_RNN_UNITS,
            DROPOUT,
            TAG_EMBEDDING_MATRIX,
            FC_UNITS_MULTIPLIERS,
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
        # Insert word embedding weights to correct position (1 for TreeStack Dependency Parsing model)
        model_weights.insert(1, word_embedding_matrix)
        # Set model weights
        self.model.set_weights(model_weights)

        self.tokenizer_word = tokenizer_word
        self.tokenizer_tag = tokenizer_tag
        self.tokenizer_pos = tokenizer_pos
        self.tokenizer_label = tokenizer_label

        # I don't want StemmerAnalyzer and PosTagger to occupy any memory in GPU!
        with tf.device("/cpu:0"):
            stemmer_analyzer = StemmerAnalyzer()
            self.stemmer_analyzer = stemmer_analyzer
            # stemmer_analyzer is passed to PoSTagger to prevent chain stemmer_analyzer initializations
            pos_tagger = PoSTagger(
                "TreeStackPoS", evaluate, self.stemmer_analyzer
            )
            self.pos_tagger = pos_tagger

    def predict(
        self, sentence: str, displacy_format: bool = False, *args
    ) -> List[Tuple[int, str, int, str]]:
        """
        Args:
            sentence:
                Input sentence.
            displacy_format:
                When set True, returns the result in spacy.displacy format to allow visualization.

        Returns:
            List of (token_index, token, arc, label).

        Raises:
            ValueError: Sentence is too long. Try again by splitting it into smaller pieces.
        """
        sentence_word_punct_tokenized = TreebankWordTokenize(sentence)
        sentence_analysis_result = self.stemmer_analyzer.predict(sentence)
        sentence_analysis_result = [
            sentence_analysis.replace("^", "+")
            for sentence_analysis in sentence_analysis_result
        ]
        num_tokens_in_sentence = len(sentence_word_punct_tokenized)

        if num_tokens_in_sentence > SENTENCE_MAX_LEN:
            raise ValueError(
                "Sentence is too long. Try again by splitting it into smaller pieces."
            )

        # This is for debugging purposes in case a consistency occurs during tokenization.
        if not len(sentence_analysis_result) == num_tokens_in_sentence:
            raise Exception(
                sentence,
                "Length of sentence and sentence_analysis_result don't match",
            )

        # *args exist for API compatability. pos_result is overwritten here.
        pos_result = self.pos_tagger.predict(sentence)
        pos_tags = [pos for (word, pos) in pos_result]

        arcs = []
        labels = []
        for t in range(num_tokens_in_sentence):
            # t is the index of token/word
            X = process_single_word_input(
                t,
                sentence,
                sentence_analysis_result,
                SENTENCE_MAX_LEN,
                TAG_MAX_LEN,
                ARC_LABEL_VECTOR_LEN,
                self.tokenizer_word,
                self.tokenizer_tag,
                self.tokenizer_label,
                self.tokenizer_pos,
                arcs,
                labels,
                pos_tags,
                WORD_FORM,
            )

            # Predicting
            logits = self.model(X).numpy()[0]

            arc, label = decode_arc_label_vector(
                logits, SENTENCE_MAX_LEN, LABEL_VOCAB_SIZE
            )

            arcs.append(arc)
            labels.append(label)

        # 0 arc index is reserved for root, therefore arc = 1 means word is dependent on the first word
        dp_result = []
        for idx, word in enumerate(sentence_word_punct_tokenized):
            dp_result.append(
                (
                    idx + 1,
                    word,
                    arcs[idx],
                    self.tokenizer_label.sequences_to_texts([[labels[idx]]])[
                        0
                    ],
                )
            )

        if not displacy_format:
            return dp_result
        else:
            dp_result_displacy_format = dp_pos_to_displacy_format(
                dp_result, pos_result
            )
            return dp_result_displacy_format
