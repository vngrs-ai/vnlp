from typing import List, Tuple

import pickle

import tensorflow as tf
import numpy as np

from ..stemmer_morph_analyzer import StemmerAnalyzer
from ..part_of_speech_tagger import PoSTagger
from ..tokenizer import TreebankWordTokenize
from ._utils import (create_dependency_parser_model, process_single_word_input)

# Resolving parent dependencies
from inspect import getsourcefile
import os
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)


RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")

MODEL_WEIGHTS_LOC = RESOURCES_PATH + "model_weights_except_word_embedding.pkl"
WORD_EMBEDDING_MATRIX_LOC = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'tokenizer/word_embedding.matrix'))
TOKENIZER_WORD_LOC = RESOURCES_PATH + "tokenizer_word.pickle"
TOKENIZER_POS_LOC = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'part_of_speech_tagger/resources/tokenizer_pos_label.pickle')) # using the tokenizer of part_of_speech_tagger
TOKENIZER_TAG_LOC = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'stemmer_morph_analyzer/resources/tokenizer_tag.pickle')) # using the tokenizer of stemmer_morph_analyzer
TOKENIZER_LABEL_LOC = RESOURCES_PATH + "tokenizer_label.pickle"

# Data Preprocessing Config
SENTENCE_MAX_LEN = 40
TAG_MAX_LEN = 15

WORD_OOV_TOKEN = '<OOV>'

# Loading Tokenizers
# Have to load tokenizers here because model config depends on them
with open(TOKENIZER_WORD_LOC, 'rb') as handle:
    tokenizer_word = pickle.load(handle)

with open(TOKENIZER_POS_LOC, 'rb') as handle:
    tokenizer_pos = pickle.load(handle)
    
with open(TOKENIZER_TAG_LOC, 'rb') as handle: # This is transferred from StemmerAnalyzer
    tokenizer_tag = pickle.load(handle)

with open(TOKENIZER_LABEL_LOC, 'rb') as handle:
    tokenizer_label = pickle.load(handle)

LABEL_VOCAB_SIZE = len(tokenizer_label.word_index)
POS_VOCAB_SIZE = len(tokenizer_pos.word_index)

# Model Config
WORD_EMBEDDING_VECTOR_SIZE = 128 # Word2Vec_medium.model
WORD_EMBEDDING_VOCAB_SIZE = 63_992 # Word2Vec_medium.model
# WORD_EMBEDDING_MATRIX and TAG_EMBEDDING MATRIX are initialized as Zeros, will be overwritten when model is loaded.
WORD_EMBEDDING_MATRIX = np.zeros((WORD_EMBEDDING_VOCAB_SIZE, WORD_EMBEDDING_VECTOR_SIZE))
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
WORD_FORM = 'whole'
DROPOUT = 0.2

class DependencyParser:
    """
    Dependency Parser class.

    - This dependency parser is *inspired* by `Tree-stack LSTM in Transition Based Dependency Parsing <https://aclanthology.org/K18-2012/>`_.
    - Inspire is emphasized because simply the approach of using Morphological Tags, Pre-trained word embeddings and POS tags as input to the model is followed, rather than implementing the network proposed there.
    - It achieves 0.6914 LAS (Labeled Attachment Score) and 0.8048 UAS (Unlabeled Attachment Score) on all of test sets of Universal Dependencies 2.9.
    - Input data is processed by NLTK.TreebankWordTokenize().
    - For more details about the training procedure, dataset and evaluation metrics, see `ReadMe <https://github.com/vngrs-ai/VNLP/blob/main/vnlp/dependency_parser/ReadMe.md>`_.

    """
    def __init__(self):
        self.model = create_dependency_parser_model(WORD_EMBEDDING_VOCAB_SIZE, WORD_EMBEDDING_VECTOR_SIZE, WORD_EMBEDDING_MATRIX,
                                                    POS_VOCAB_SIZE, POS_EMBEDDING_VECTOR_SIZE,
                                                    SENTENCE_MAX_LEN, TAG_MAX_LEN, ARC_LABEL_VECTOR_LEN, NUM_RNN_STACKS, 
                                                    TAG_NUM_RNN_UNITS, LC_NUM_RNN_UNITS, LC_ARC_LABEL_NUM_RNN_UNITS, RC_NUM_RNN_UNITS,
                                                    DROPOUT, TAG_EMBEDDING_MATRIX, FC_UNITS_MULTIPLIERS)
        # Load Word embedding matrix
        word_embedding_matrix = np.load(WORD_EMBEDDING_MATRIX_LOC)
        # Load Model weights
        with open(MODEL_WEIGHTS_LOC, 'rb') as fp:
            model_weights = pickle.load(fp)
        # Insert word embedding weights to correct position (1 for Dependency Parsing model)
        model_weights.insert(1, word_embedding_matrix)
        # Set model weights
        self.model.set_weights(model_weights)

        self.tokenizer_word = tokenizer_word
        self.tokenizer_tag = tokenizer_tag
        self.tokenizer_pos = tokenizer_pos
        self.tokenizer_label = tokenizer_label

        # I don't want StemmerAnalyzer and PosTagger to occupy any memory in GPU!
        with tf.device('/cpu:0'):
            stemmer_analyzer = StemmerAnalyzer()
            self.stemmer_analyzer = stemmer_analyzer
            # stemmer_analyzer is passed to PoSTagger to prevent chain stemmer_analyzer initializations
            pos_tagger = PoSTagger(self.stemmer_analyzer) 
            self.pos_tagger = pos_tagger
        

    def predict(self, sentence: str, displacy_format: bool = False) -> List[Tuple[int, str, int, str]]:

        """
        High level user API for Dependency Parsing.

        Args:
            sentence:
                Input text(sentence).
            displacy_format:
                When set True, returns the result in spacy.displacy format to allow visualization.
        
        Returns:
            List of (token_index, token, arc, label).
                
        Raises:
            ValueError: Sentence is too long. Try again by splitting it into smaller pieces.

        Example::

            from vnlp import DependencyParser
            dependency_parser = DependencyParser()
            dependency_parser.predict("Onun için yol arkadaşlarımızı titizlikle seçer, kendilerini iyice sınarız.")

            [(1, 'Onun', 5, 'obl'),
            (2, 'için', 1, 'case'),
            (3, 'yol', 1, 'nmod'),
            (4, 'arkadaşlarımızı', 5, 'obj'),
            (5, 'titizlikle', 6, 'obl'),
            (6, 'seçer', 7, 'acl'),
            (7, ',', 10, 'punct'),
            (8, 'kendilerini', 10, 'obj'),
            (9, 'iyice', 8, 'advmod'),
            (10, 'sınarız', 0, 'root'),
            (11, '.', 10, 'punct')]
            
            # Visualization with Spacy:
            import spacy
            from vnlp import DependencyParser
            dependency_parser = DependencyParser()
            result = dependency_parser.predict("Bu örnek bir cümledir.", displacy_format = True)
            spacy.displacy.render(result, style="dep", manual = True)
        """

        sentence_word_punct_tokenized = TreebankWordTokenize(sentence)
        sentence_analysis_result = self.stemmer_analyzer.predict(sentence)
        sentence_analysis_result = [sentence_analysis.replace('^', '+') for sentence_analysis in sentence_analysis_result]
        num_tokens_in_sentence = len(sentence_word_punct_tokenized)

        if num_tokens_in_sentence > SENTENCE_MAX_LEN:
            raise ValueError('Sentence is too long. Try again by splitting it into smaller pieces.')

        # This is for debugging purposes in case a consistency occurs during tokenization.
        if not len(sentence_analysis_result) == num_tokens_in_sentence:
            raise Exception(sentence, "Length of sentence and sentence_analysis_result don't match")

        pos_result = self.pos_tagger.predict(sentence)
        pos_tags = [pos for (word, pos) in pos_result]

        arcs = []
        labels = []
        for t in range(num_tokens_in_sentence):
            # t is the index of token/word
            X = process_single_word_input(t, sentence, sentence_analysis_result, 
                                          SENTENCE_MAX_LEN, TAG_MAX_LEN, ARC_LABEL_VECTOR_LEN, 
                                          self.tokenizer_word, self.tokenizer_tag, self.tokenizer_label, self.tokenizer_pos, 
                                          arcs, labels, pos_tags, WORD_FORM)

            # Predicting
            raw_pred = self.model.predict(X)[0]
            
            arc = np.argmax(raw_pred[:SENTENCE_MAX_LEN + 1]) # +1 is due to reserving of arc 0 for root
            label = np.argmax(raw_pred[SENTENCE_MAX_LEN + 1: SENTENCE_MAX_LEN + 1 + LABEL_VOCAB_SIZE + 1])

            arcs.append(arc)
            labels.append(label)

        # 0 arc index is reserved for root, therefore arc = 1 means word is dependent on the first word
        dp_result = []
        for idx, word in enumerate(sentence_word_punct_tokenized):
            dp_result.append((idx + 1, word, arcs[idx], tokenizer_label.sequences_to_texts([[labels[idx]]])[0]))

        if not displacy_format:
            return dp_result
        else:
            dp_result_displacy_format = {'words': [],
                   'arcs': []}
            for dp_res, pos_res in zip(dp_result, pos_result):
                word = dp_res[1]
                pos_tag = pos_res[1]
                
                arc_source = dp_res[0] - 1
                arc_dest = dp_res[2] - 1
                dp_label = dp_res[3]
                
                dp_result_displacy_format['words'].append({'text': word, 'tag': pos_tag})
                if arc_dest < 0:
                    continue
                else:
                    if arc_source <= arc_dest:
                        dp_result_displacy_format['arcs'].append({'start': arc_source, 'end': arc_dest, 'label': dp_label, 'dir': 'right'})
                    else:
                        dp_result_displacy_format['arcs'].append({'start': arc_dest, 'end': arc_source, 'label': dp_label, 'dir': 'left'})

            return dp_result_displacy_format