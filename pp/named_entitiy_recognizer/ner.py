from typing import List, Tuple

import pickle

import numpy as np
import tensorflow as tf

from ..utils import WordPunctTokenize
from ._utils import create_ner_model

import os
RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")

MODEL_LOC = RESOURCES_PATH + "model_weights.hdf5"
TOKENIZER_X_LOC = RESOURCES_PATH + "tokenizer_X.pickle"
TOKENIZER_Y_LOC = RESOURCES_PATH + "tokenizer_y.pickle"

VOCAB_SIZE = 140
SEQ_LEN = 256
OOV_TOKEN = '<OOV>'
PADDING_STRAT = 'post'

EMBED_SIZE = 32
RNN_DIM = 128
NUM_RNN_STACKS = 5
MLP_DIM = 32
NUM_CLASSES = 5 #Equals to len(tokenizer_y.index_word) + 1. +1 is reserved to 0, which corresponds to padded values.
DROPOUT = 0.3

class NamedEntityRecognizer:
    def __init__(self):
        self.model = create_ner_model(VOCAB_SIZE, EMBED_SIZE, SEQ_LEN, NUM_RNN_STACKS, RNN_DIM, MLP_DIM, NUM_CLASSES, DROPOUT)
        self.model.load_weights(MODEL_LOC)

        with open(TOKENIZER_X_LOC, 'rb') as handle:
            tokenizer_X = pickle.load(handle)
        
        with open(TOKENIZER_Y_LOC, 'rb') as handle:
            tokenizer_y = pickle.load(handle)

        self.tokenizer_X = tokenizer_X
        self.tokenizer_y = tokenizer_y


    def _predict_char_level(self, word_punct_tokenized: List[str]) -> List[int]:
        """
        Input:
        word_punct_tokenized List[str]: List of tokens (WordPunct level)
        i.e: ["İstanbul", "'", "da", "yaşıyorum", "."]

        Output:
        arg_max_pred (List[int]): List of integers, indicating classes
        that can be fed into sequences_to_text function.
        """
        white_space_joined_word_punct_tokens = " ".join(word_punct_tokenized)
        sequences = self.tokenizer_X.texts_to_sequences([white_space_joined_word_punct_tokens])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen = SEQ_LEN, padding = PADDING_STRAT)
        raw_pred = self.model.predict([padded])
        arg_max_pred = tf.math.argmax(raw_pred, axis = 2).numpy().reshape(-1)
        
        return arg_max_pred

    def _charner_decoder(self, word_punct_tokenized: List[str], arg_max_pred: List[int]) -> List[str]:
        """
        Input:
        word_punct_tokenized: List[str] : List of tokens (WordPunct level)
        i.e: ["İstanbul", "'", "da", "yaşıyorum", "."]
        
        arg_max_pred: List[int] : argmax(axis = -1) of model output
        
        Output:
        decoded_entities: List[str] : List of entities, one entity per token
        """
        
        lens = [0] + [len(token) + 1 for token in word_punct_tokenized]
        cumsum_of_lens = np.cumsum(lens)
        
        decoded_entities = []
        for idx in range(len(cumsum_of_lens) - 1):
            lower_bound = cumsum_of_lens[idx]
            upper_bound = cumsum_of_lens[idx + 1]

            island = arg_max_pred[lower_bound:upper_bound]
            # Extracting mode value
            vals, counts = np.unique(island, return_counts = True)
            mode_value = vals[np.argmax(counts)]
            
            detokenized_pred = self.tokenizer_y.sequences_to_texts([[mode_value]])[0]
            decoded_entities.append(detokenized_pred)
            
        return decoded_entities

    def predict(self, text: str) -> List[Tuple[str, str]]:
        """
        High level API for Named Entity Recognition.

        Input:
        text (str): String of text. Works for both untokenized raw text
        and white-space separated tokenized text.

        Output:
        token_entity_pairs (List[Tuple[str, str]]): 

        Sample use:
        ner = NER()
        print(ner.predict("Ben Melikşah, 28 yaşındayım, İstanbul'da ikamet ediyorum ve VNGRS AI Takımı'nda çalışıyorum."))
        [('Ben', 'O'),
        ('Melikşah', 'PER'),
        (',', 'O'),
        ('28', 'O'),
        ('yaşındayım', 'O'),
        (',', 'O'),
        ('İstanbul', 'LOC'),
        ("'", 'O'),
        ('da', 'O'),
        ('ikamet', 'O'),
        ('ediyorum', 'O'),
        ('ve', 'O'),
        ('VNGRS', 'ORG'),
        ('AI', 'ORG'),
        ('Takımı', 'ORG'),
        ("'", 'O'),
        ('nda', 'O'),
        ('çalışıyorum', 'O'),
        ('.'), 'O']
        """
        word_punct_tokenized = WordPunctTokenize(text)

        # if len chars (including whitespaces) > sequence length, split it recursively
        len_text = len(list(" ".join(word_punct_tokenized)))
        if len_text > SEQ_LEN:
            
            num_tokens = len(word_punct_tokenized)
            
            first_half_result = self.predict(" ".join(word_punct_tokenized[:num_tokens // 2]))
            first_half_tokens = [pair[0] for pair in first_half_result]
            first_half_entities = [pair[1] for pair in first_half_result]
            
            second_half_result = self.predict(" ".join(word_punct_tokenized[(num_tokens // 2):]))
            second_half_tokens =  [pair[0] for pair in second_half_result]
            second_half_entities = [pair[1] for pair in second_half_result]

            word_punct_tokenized = first_half_tokens + second_half_tokens
            decoded_entities = first_half_entities + second_half_entities

        else:
            charlevel_pred = self._predict_char_level(word_punct_tokenized)
            decoded_entities = self._charner_decoder(word_punct_tokenized, charlevel_pred)
            
        token_entity_pairs = [(t,e) for t,e in zip(word_punct_tokenized, decoded_entities)]
        
        return token_entity_pairs