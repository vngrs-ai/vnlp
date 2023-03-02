from typing import List, Tuple

import pickle

import tensorflow as tf
import numpy as np

from ..stemmer_morph_analyzer import StemmerAnalyzer
from ..tokenizer import TreebankWordTokenize
from ..utils import check_and_download, load_keras_tokenizer
from ._treestack_utils import (create_pos_tagger_model, process_single_word_input)

# Resolving parent dependencies
from inspect import getsourcefile
import os
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")

PROD_WEIGHTS_LOC = RESOURCES_PATH + "PoS_TreeStack_prod.weights"
EVAL_WEIGHTS_LOC = RESOURCES_PATH + "PoS_TreeStack_eval.weights"
WORD_EMBEDDING_MATRIX_LOC = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'resources/TBWTokenized_word_embedding.matrix'))

PROD_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/PoS_TreeStack_prod.weights"
EVAL_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/PoS_TreeStack_eval.weights"
WORD_EMBEDDING_MATRIX_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/TBWTokenized_word_embedding.matrix"

TOKENIZER_WORD_LOC = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'resources/TB_word_tokenizer.json'))
TOKENIZER_MORPH_TAG_LOC_LOC = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'stemmer_morph_analyzer/resources/Stemmer_morph_tag_tokenizer.json')) # using the tokenizer of stemmer_morph_analyzer
TOKENIZER_POS_LABEL_LOC = RESOURCES_PATH + "PoS_label_tokenizer.json"

# Data Preprocessing Config
SENTENCE_MAX_LEN = 40
TAG_MAX_LEN = 15

WORD_OOV_TOKEN = '<OOV>'

# Loading Tokenizers
# Have to load tokenizers here because model config depends on them
tokenizer_word = load_keras_tokenizer(TOKENIZER_WORD_LOC)
# This is transferred from StemmerAnalyzer
tokenizer_morph_tag = load_keras_tokenizer(TOKENIZER_MORPH_TAG_LOC_LOC)
tokenizer_pos_label = load_keras_tokenizer(TOKENIZER_POS_LABEL_LOC)

POS_VOCAB_SIZE = len(tokenizer_pos_label.word_index)

# Model Config
WORD_EMBEDDING_VECTOR_SIZE = 128 # Word2Vec_medium.model
WORD_EMBEDDING_VOCAB_SIZE = 63_992 # Word2Vec_medium.model
# WORD_EMBEDDING_MATRIX and TAG_EMBEDDING MATRIX are initialized as Zeros, will be overwritten when model is loaded.
WORD_EMBEDDING_MATRIX = np.zeros((WORD_EMBEDDING_VOCAB_SIZE, WORD_EMBEDDING_VECTOR_SIZE))
TAG_EMBEDDING_MATRIX = np.zeros((127, 32))

NUM_RNN_STACKS = 2
RNN_UNITS_MULTIPLIER = 2
TAG_NUM_RNN_UNITS = WORD_EMBEDDING_VECTOR_SIZE
LC_NUM_RNN_UNITS = TAG_NUM_RNN_UNITS * RNN_UNITS_MULTIPLIER
RC_NUM_RNN_UNITS = TAG_NUM_RNN_UNITS * RNN_UNITS_MULTIPLIER
FC_UNITS_MULTIPLIERS = (2, 1)
WORD_FORM = 'whole'
DROPOUT = 0.2

class TreeStackPoS:
    """
    Tree-stack Part of Speech Tagger class.

    - This Part of Speech Tagger is *inspired* by `Tree-stack LSTM in Transition Based Dependency Parsing <https://aclanthology.org/K18-2012/>`_.
    - "Inspire" is emphasized because this implementation uses the approach of using Morphological Tags, Pre-trained word embeddings and POS tags as input for the model, rather than implementing the exact network proposed in the paper.
    - It achieves 0.89 Accuracy and 0.71 F1_macro_score on test sets of Universal Dependencies 2.9.
    - Input data is processed by NLTK.tokenize.TreebankWordTokenizer.
    - For more details about the training procedure, dataset and evaluation metrics, see `ReadMe <https://github.com/vngrs-ai/VNLP/blob/main/vnlp/part_of_speech_tagger/ReadMe.md>`_.
    """
    def __init__(self, evaluate, stemmer_analyzer = None):
        self.model = create_pos_tagger_model(WORD_EMBEDDING_VOCAB_SIZE, WORD_EMBEDDING_VECTOR_SIZE, WORD_EMBEDDING_MATRIX,
                                             POS_VOCAB_SIZE, SENTENCE_MAX_LEN, TAG_MAX_LEN, NUM_RNN_STACKS, 
                                             TAG_NUM_RNN_UNITS, LC_NUM_RNN_UNITS, RC_NUM_RNN_UNITS,
                                             DROPOUT, TAG_EMBEDDING_MATRIX, FC_UNITS_MULTIPLIERS)
         # Check and download word embedding matrix and model weights
        check_and_download(WORD_EMBEDDING_MATRIX_LOC, WORD_EMBEDDING_MATRIX_LINK)
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
        with open(MODEL_WEIGHTS_LOC, 'rb') as fp:
            model_weights = pickle.load(fp)
        # Insert word embedding weights to correct position (1 for Part of Speech Tagger model)
        model_weights.insert(1, word_embedding_matrix)
        # Set model weights
        self.model.set_weights(model_weights)

        self.tokenizer_word = tokenizer_word
        self.tokenizer_morph_tag = tokenizer_morph_tag
        self.tokenizer_pos_label = tokenizer_pos_label

        # I don't want StemmerAnalyzer to occupy any memory in GPU!
        if stemmer_analyzer is None:
            with tf.device('/cpu:0'):
                stemmer_analyzer = StemmerAnalyzer()
        self.stemmer_analyzer = stemmer_analyzer

    def predict(self, sentence: str) -> List[Tuple[str, str]]:
        """
        Args:
            sentence:
                Input text(sentence).

        Returns:
             List of (token, pos_label).
        """

        whole_tokens_in_sentence = TreebankWordTokenize(sentence)
        sentence_analysis_result = self.stemmer_analyzer.predict(sentence)
        sentence_analysis_result = [sentence_analysis.replace('^', '+') for sentence_analysis in sentence_analysis_result]
        num_tokens_in_sentence = len(whole_tokens_in_sentence)

        # This is for debugging purposes in case an inconsistency occurs during tokenization.
        if not len(sentence_analysis_result) == num_tokens_in_sentence:
            raise Exception(sentence, "Length of sentence and sentence_analysis_result don't match")

        pos_int_labels = []
        for t in range(num_tokens_in_sentence):
            # t is the index of token/word
            X = process_single_word_input(t, whole_tokens_in_sentence, sentence_analysis_result, 
                                          SENTENCE_MAX_LEN, TAG_MAX_LEN, 
                                          self.tokenizer_word, self.tokenizer_morph_tag, self.tokenizer_pos_label,
                                          pos_int_labels, WORD_FORM)
            
            # Predicting
            raw_pred = self.model(X).numpy()[0]
            
            pos_int_label = np.argmax(raw_pred, axis = -1)
            pos_int_labels.append(pos_int_label)

        # Converting integer labels to text form 
        pos_labels = [self.tokenizer_pos_label.sequences_to_texts([[pos_int_label]])[0] for pos_int_label in pos_int_labels]
        result = [(token, pos_label) for token, pos_label in zip(whole_tokens_in_sentence, pos_labels)]

        return result