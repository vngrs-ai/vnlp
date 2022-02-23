from typing import List

import pickle

import numpy as np
import tensorflow as tf

from ._utils import create_sentiment_analysis_model, preprocess_text

# Resolving parent dependencies
from inspect import getsourcefile
import os
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")

MODEL_LOC = RESOURCES_PATH + "model_weights.hdf5"
TOKENIZER_WORD_LOC = RESOURCES_PATH + "tokenizer_word.pickle"

# Data Processing Config
TEXT_MAX_LEN = 128 # 0.995th quantile of text length in data (post tokenization) is 125, so 128 is inclusive enough
WORD_EMBEDDING_VECTOR_SIZE = 128 # Word2Vec_medium.model
WORD_EMBEDDING_VOCAB_SIZE = 63_992 # Word2Vec_medium.model
# WORD_EMBEDDING_MATRIX and TAG_EMBEDDING MATRIX are initialized as Zeros, will be overwritten when model is loaded.
WORD_EMBEDDING_MATRIX = np.zeros((WORD_EMBEDDING_VOCAB_SIZE, WORD_EMBEDDING_VECTOR_SIZE))
WORD__OOV = '<OOV>'

# Model Config
NUM_RNN_UNITS = 256
NUM_RNN_STACKS = 2
DROPOUT = 0.2


class SentimentAnalyzer:
    def __init__(self):
        self.model = create_sentiment_analysis_model(WORD_EMBEDDING_VOCAB_SIZE, WORD_EMBEDDING_VECTOR_SIZE, WORD_EMBEDDING_MATRIX,
                                                     NUM_RNN_UNITS, NUM_RNN_STACKS, DROPOUT)
        self.model.load_weights(MODEL_LOC)

        with open(TOKENIZER_WORD_LOC, 'rb') as handle:
            tokenizer_word = pickle.load(handle)
        self.tokenizer_word = tokenizer_word


    def predict(self, input_text: str) -> int:
        """
        High level user function to obtain sentiment label
        1: positive sentiment
        0: negative sentiment

        Input:
        input_text (str): string of text

        Output:
        result (int): Sentiment label of input_text

        Sample use:
        from pp.sentiment_analyzer import SentimentAnalyzer
        sent_analyzer = SentimentAnalyzer()
        sent_analyzer.predict("Sipariş geldiğinde biz karnımızı atıştırmalıklarla doyurmuştuk.")
        
        0

        """

        prob = self.predict_proba(input_text)

        return 1 if prob > 0.5 else 0


    def predict_proba(self, input_text: str) -> float:
        """
        High level user function to obtain probability score for sentiment

        Input:
        input_text (str): string of text

        Output:
        result (float): Probability that given input_text is positive sentiment

        Sample use:
        from pp.sentiment_analyzer import SentimentAnalyzer
        sent_analyzer = SentimentAnalyzer()
        sent_analyzer.predict_proba("Sipariş geldiğinde biz karnımızı atıştırmalıklarla doyurmuştuk.")
        
        0.15

        """
        preprocessed_text = preprocess_text(input_text)
        integer_tokenized_text = self.tokenizer_word.texts_to_sequences([preprocessed_text])
        
        num_preprocessed_text_tokens = len(preprocessed_text.split())
        num_int_tokens = len(integer_tokenized_text[0])

        # if text is longer than the length the model is trained on
        if num_int_tokens > TEXT_MAX_LEN:
            first_half_of_preprocessed_text = " ".join(preprocessed_text.split()[:num_preprocessed_text_tokens // 2])
            print('First half:', first_half_of_preprocessed_text)
            second_half_of_preprocessed_text = " ".join(preprocessed_text.split()[num_preprocessed_text_tokens // 2:])
            print('Second half:', second_half_of_preprocessed_text)
            prob = (self.predict_proba(first_half_of_preprocessed_text) + self.predict_proba(second_half_of_preprocessed_text)) / 2

        else:
            padded_text = tf.keras.preprocessing.sequence.pad_sequences(integer_tokenized_text, maxlen = TEXT_MAX_LEN, 
                                                padding = 'pre', truncating = 'pre')
            prob = self.model.predict(padded_text)[0][0]

        return prob