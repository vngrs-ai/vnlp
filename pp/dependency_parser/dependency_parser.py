from typing import List, Tuple

import pickle

import tensorflow as tf
import numpy as np

from ._utils import (create_dependency_parser_model, process_single_word_input)

from ..stemmer_morph_analyzer import StemmerAnalyzer

# Resolving parent dependencies
from inspect import getsourcefile
import os
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from utils import WordPunctTokenize

RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")

MODEL_LOC = RESOURCES_PATH + "model_weights.hdf5"
TOKENIZER_WORD_LOC = RESOURCES_PATH + "tokenizer_word.pickle"
TOKENIZER_LABEL_LOC = RESOURCES_PATH + "tokenizer_label.pickle"
TOKENIZER_TAG_LOC = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'stemmer_morph_analyzer/resources/tokenizer_tag.pickle')) # using the tokenizer of stemmer_morph_analyzer

# Data Preprocessing Config
SENTENCE_MAX_LEN = 40
TAG_MAX_LEN = 15

WORD_OOV_TOKEN = '<OOV>'

# Loading Tokenizers
# Have to load tokenizers here because model config depends on them
with open(TOKENIZER_WORD_LOC, 'rb') as handle:
    tokenizer_word = pickle.load(handle)

with open(TOKENIZER_LABEL_LOC, 'rb') as handle:
    tokenizer_label = pickle.load(handle)
    
with open(TOKENIZER_TAG_LOC, 'rb') as handle: # This is transferred from StemmerAnalyzer
    tokenizer_tag = pickle.load(handle)

# Model Config
WORD_EMBEDDING_VECTOR_SIZE = 128 # Word2Vec_medium.model
WORD_EMBEDDING_VOCAB_SIZE = 63_998 # Word2Vec_medium.model
WORD_EMBEDDING_MATRIX = np.zeros(((WORD_EMBEDDING_VOCAB_SIZE + 1, WORD_EMBEDDING_VECTOR_SIZE))) # Initializing as zeros, will be overwritten by pretrained weights

NUM_RNN_STACKS = 2
RNN_UNITS_MULTIPLIER = 2
TAG_NUM_RNN_UNITS = WORD_EMBEDDING_VECTOR_SIZE
LC_NUM_RNN_UNITS = TAG_NUM_RNN_UNITS * RNN_UNITS_MULTIPLIER
LC_ARC_LABEL_NUM_RNN_UNITS = TAG_NUM_RNN_UNITS * RNN_UNITS_MULTIPLIER
RC_NUM_RNN_UNITS = TAG_NUM_RNN_UNITS * RNN_UNITS_MULTIPLIER
ARC_LABEL_VECTOR_LEN = SENTENCE_MAX_LEN + len(tokenizer_label.word_index) + 1
DROPOUT = 0.2

class DependencyParser:
    def __init__(self):
        self.model = create_dependency_parser_model(WORD_EMBEDDING_VOCAB_SIZE, WORD_EMBEDDING_VECTOR_SIZE, WORD_EMBEDDING_MATRIX,
                                                    SENTENCE_MAX_LEN, TAG_MAX_LEN, ARC_LABEL_VECTOR_LEN, NUM_RNN_STACKS, 
                                                    TAG_NUM_RNN_UNITS, LC_NUM_RNN_UNITS, LC_ARC_LABEL_NUM_RNN_UNITS, RC_NUM_RNN_UNITS,
                                                    DROPOUT)
        self.model.load_weights(MODEL_LOC)
        self.tokenizer_word = tokenizer_word
        self.tokenizer_label = tokenizer_label
        self.tokenizer_tag = tokenizer_tag

        # I don't want StemmerAnalyzer to occupy any memory in GPU!
        with tf.device('/cpu:0'):
            sa = StemmerAnalyzer()
        self.sa = sa

    def predict(self, sentence: str) -> List[Tuple[int, str, int, str]]:

        """
        High level user function

        Input:
        sentence (str): string of text(sentence)

        Output:
        result (List[Tuple[int, str, int, str]]): list of selected morphological analysis for each token

        Sample use:
        from pp.dependency_parser import DependencyParser
        dp = DependencyParser()
        dp.predict("Onun için yol arkadaşlarımızı titizlikle seçer, kendilerini iyice sınarız.")

        [(1, 'Onun', 6, 'obl'),
        (2, 'için', 1, 'case'),
        (3, 'yol', 4, 'nmod:poss'),
        (4, 'arkadaşlarımızı', 6, 'obj'),
        (5, 'titizlikle', 6, 'advmod'),
        (6, 'seçer', 0, 'root'),
        (7, ',', 10, 'punct'),
        (8, 'kendilerini', 10, 'obj'),
        (9, 'iyice', 10, 'advmod'),
        (10, 'sınarız', 6, 'conj'),
        (11, '.', 10, 'punct')]
        
        """

        sentence_word_punct_tokenized = WordPunctTokenize(sentence)
        sentence_analysis_result = self.sa.predict(sentence)
        sentence_analysis_result = [sentence_analysis.replace('^', '+') for sentence_analysis in sentence_analysis_result]
        num_tokens_in_sentence = len(sentence_analysis_result)

        if not len(sentence_analysis_result) == num_tokens_in_sentence:
            raise Exception(sentence, "Length of sentence and sentence_analysis_result don't match")

        arcs = []
        labels = []
        for t in range(num_tokens_in_sentence):
            # t is the index of token/word
            X = process_single_word_input(t, sentence_analysis_result, 
                                          SENTENCE_MAX_LEN, TAG_MAX_LEN, ARC_LABEL_VECTOR_LEN, 
                                          self.tokenizer_word, self.tokenizer_tag, self.tokenizer_label, arcs, labels)
            
            # Predicting
            raw_pred = self.model.predict(X)[0]
            
            arc = np.argmax(raw_pred[:SENTENCE_MAX_LEN])
            label = np.argmax(raw_pred[SENTENCE_MAX_LEN:])

            arcs.append(arc)
            labels.append(label)

        # 0 arc index is reserved for root, therefore arc = 1 means word is dependent on the first word
        result = []
        for idx, word in enumerate(sentence_word_punct_tokenized):
            result.append((idx + 1, word, arcs[idx], tokenizer_label.sequences_to_texts([[labels[idx]]])[0]))

        return result