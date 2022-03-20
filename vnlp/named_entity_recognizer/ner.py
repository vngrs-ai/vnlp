from typing import List, Tuple

import pickle
import re

import numpy as np
import tensorflow as tf

from ._utils import create_ner_model

# Resolving parent dependencies
from inspect import getsourcefile
import os
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from tokenizer import WordPunctTokenize

RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")

MODEL_LOC = RESOURCES_PATH + "model_weights.hdf5"
TOKENIZER_CHAR_LOC = RESOURCES_PATH + "tokenizer_char.pickle"
TOKENIZER_LABEL_LOC = RESOURCES_PATH + "tokenizer_label.pickle"

CHAR_VOCAB_SIZE = 150
SEQ_LEN_MAX = 256
OOV_TOKEN = '<OOV>'
PADDING_STRAT = 'post'

EMBED_SIZE = 32
RNN_DIM = 128
NUM_RNN_STACKS = 5
MLP_DIM = 32
NUM_CLASSES = 5 #Equals to len(tokenizer_label.index_word) + 1. +1 is reserved to 0, which corresponds to padded values.
DROPOUT = 0.3

class NamedEntityRecognizer:
    def __init__(self):
        self.model = create_ner_model(CHAR_VOCAB_SIZE, EMBED_SIZE, SEQ_LEN_MAX, NUM_RNN_STACKS, RNN_DIM, MLP_DIM, NUM_CLASSES, DROPOUT)
        self.model.load_weights(MODEL_LOC)

        with open(TOKENIZER_CHAR_LOC, 'rb') as handle:
            tokenizer_char = pickle.load(handle)
        
        with open(TOKENIZER_LABEL_LOC, 'rb') as handle:
            tokenizer_label = pickle.load(handle)

        self.tokenizer_char = tokenizer_char
        self.tokenizer_label = tokenizer_label


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
        white_space_joined_word_punct_tokens = [char for char in white_space_joined_word_punct_tokens]
        sequences = self.tokenizer_char.texts_to_sequences([white_space_joined_word_punct_tokens])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen = SEQ_LEN_MAX, padding = PADDING_STRAT)
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
            upper_bound = cumsum_of_lens[idx + 1] -1 # minus one prevents including the whitespace after the token

            island = arg_max_pred[lower_bound:upper_bound]
            # Extracting mode value
            vals, counts = np.unique(island, return_counts = True)
            mode_value = vals[np.argmax(counts)]
            
            detokenized_pred = self.tokenizer_label.sequences_to_texts([[mode_value]])[0]
            decoded_entities.append(detokenized_pred)
            
        return decoded_entities

    def predict(self, text: str, displacy_format: bool = False) -> List[Tuple[str, str]]:
        """
        High level API for Named Entity Recognition.

        Input:
        text (str): String of text. Works for both untokenized raw text
        and white-space separated tokenized text.

        Output:
        ner_result (List[Tuple[str, str]]): 

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
        if len_text > SEQ_LEN_MAX:
            
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
            
        ner_result = [(t,e) for t,e in zip(word_punct_tokenized, decoded_entities)]
        
        if not displacy_format:
            return ner_result
        else:
            # Obtain Token Start and End indices
            token_loc = {}
            for duo in ner_result:
                word = duo[0]

                # https://stackoverflow.com/a/13989661/4505301
                if word == '.':
                    continue

                if not word in token_loc:
                    token_loc[word] = []
                for match in re.finditer(word, text):
                    start, end = match.start(), match.end()
                    substring_of_prev_strings = False
                    # check if this match is part another previously matched string and skip it if so
                    for token in token_loc:
                        list_of_token_indices = token_loc[token]
                        for prev_token_indices in list_of_token_indices:
                            prev_token_start = prev_token_indices[0]
                            prev_token_end = prev_token_indices[1]
                            
                            if (start >= prev_token_start) and (end <= prev_token_end):
                                substring_of_prev_strings = True
                                break

                    if (not (start, end) in token_loc[word]) and (not substring_of_prev_strings):
                        token_loc[word].append((start, end))

            # Process for Spacy
            ner_result_displacy_format = {'text': text,
                'ents': [],
                'title': None}

            is_continuation = False
            ents = {}
            for idx, duo in enumerate(ner_result):
                word = duo[0]
                entity = duo[1]
                
                # https://stackoverflow.com/a/13989661/4505301
                if word == '.':
                    continue
                
                start, end = token_loc[word][0]
                del token_loc[word][0]
                
                if not (entity == 'O'):
                    if not is_continuation:
                        ents['start'] = start
                        ents['label'] = entity
                    
                    if (idx != (len(ner_result) -1)) and (ner_result[idx + 1][1] == entity):
                        is_continuation = True
                    else:
                        ents['end'] = end
                        ner_result_displacy_format['ents'].append(ents)
                        ents = {}
                        is_continuation = False

            return ner_result_displacy_format