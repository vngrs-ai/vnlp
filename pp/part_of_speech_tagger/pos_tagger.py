from typing import List, Tuple

import pickle

import tensorflow as tf
import numpy as np

from ._utils import (create_pos_tagger_model, process_single_word_input)

from ..stemmer_morph_analyzer import StemmerAnalyzer

# Resolving parent dependencies
from inspect import getsourcefile
import os
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from tokenizer import TreebankWordTokenize

RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")

MODEL_LOC = RESOURCES_PATH + "model_weights.hdf5"
TOKENIZER_WORD_LOC = RESOURCES_PATH + "tokenizer_word.pickle"
TOKENIZER_TAG_LOC = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'stemmer_morph_analyzer/resources/tokenizer_tag.pickle')) # using the tokenizer of stemmer_morph_analyzer
TOKENIZER_POS_LABEL_LOC = RESOURCES_PATH + "tokenizer_pos_label.pickle"

# Data Preprocessing Config
SENTENCE_MAX_LEN = 40
TAG_MAX_LEN = 15

WORD_OOV_TOKEN = '<OOV>'

# Loading Tokenizers
# Have to load tokenizers here because model config depends on them
with open(TOKENIZER_WORD_LOC, 'rb') as handle:
    tokenizer_word = pickle.load(handle)
    
with open(TOKENIZER_TAG_LOC, 'rb') as handle: # This is transferred from StemmerAnalyzer
    tokenizer_tag = pickle.load(handle)

with open(TOKENIZER_POS_LABEL_LOC, 'rb') as handle:
    tokenizer_pos_label = pickle.load(handle)

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

class PoSTagger:
    def __init__(self, stemmer_analyzer = None):
        self.model = create_pos_tagger_model(WORD_EMBEDDING_VOCAB_SIZE, WORD_EMBEDDING_VECTOR_SIZE, WORD_EMBEDDING_MATRIX,
                                             POS_VOCAB_SIZE, SENTENCE_MAX_LEN, TAG_MAX_LEN, NUM_RNN_STACKS, 
                                             TAG_NUM_RNN_UNITS, LC_NUM_RNN_UNITS, RC_NUM_RNN_UNITS,
                                             DROPOUT, TAG_EMBEDDING_MATRIX, FC_UNITS_MULTIPLIERS)

        self.model.load_weights(MODEL_LOC)
        self.tokenizer_word = tokenizer_word
        self.tokenizer_tag = tokenizer_tag
        self.tokenizer_pos_label = tokenizer_pos_label

        # I don't want StemmerAnalyzer to occupy any memory in GPU!
        if stemmer_analyzer is None:
            with tf.device('/cpu:0'):
                stemmer_analyzer = StemmerAnalyzer()
        self.stemmer_analyzer = stemmer_analyzer

    def predict(self, sentence: str) -> List[Tuple[str, str]]:

        """
        High level user function

        Input:
        sentence (str): string of text(sentence)

        Output:
        result (List[Tuple[str, str]]): list of (token, pos_tag)

        Sample use:
        from pp.part_of_speech_tagger import PoSTagger
        pos = PoSTagger()
        pos.predict("Vapurla Beşiktaş'a geçip yürüyerek Maçka Parkı'na ulaştım.")

        [('Vapurla', 'NOUN'),
        ("Beşiktaş'a", 'PROPN'),
        ('geçip', 'ADV'),
        ('yürüyerek', 'ADV'),
        ('Maçka', 'PROPN'),
        ("Parkı'na", 'NOUN'),
        ('ulaştım', 'VERB'),
        ('.', 'PUNCT')]
        
        """

        whole_tokens_in_sentence = TreebankWordTokenize(sentence)
        sentence_analysis_result = self.stemmer_analyzer.predict(sentence)
        sentence_analysis_result = [sentence_analysis.replace('^', '+') for sentence_analysis in sentence_analysis_result]
        num_tokens_in_sentence = len(whole_tokens_in_sentence)

        if not len(sentence_analysis_result) == num_tokens_in_sentence:
            raise Exception(sentence, "Length of sentence and sentence_analysis_result don't match")

        pos_int_labels = []
        for t in range(num_tokens_in_sentence):
            # t is the index of token/word
            X = process_single_word_input(t, whole_tokens_in_sentence, sentence_analysis_result, 
                                          SENTENCE_MAX_LEN, TAG_MAX_LEN, 
                                          self.tokenizer_word, self.tokenizer_tag, self.tokenizer_pos_label,
                                          pos_int_labels, WORD_FORM)
            
            # Predicting
            raw_pred = self.model.predict(X)[0]
            
            pos_int_label = np.argmax(raw_pred, axis = -1)
            pos_int_labels.append(pos_int_label)

        # Converting integer labels to text form 
        pos_labels = [self.tokenizer_pos_label.sequences_to_texts([[pos_int_label]])[0] for pos_int_label in pos_int_labels]
        result = [(token, pos_label) for token, pos_label in zip(whole_tokens_in_sentence, pos_labels)]

        return result