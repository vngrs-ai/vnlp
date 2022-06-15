from typing import List, Tuple

import pickle

import numpy as np

import sentencepiece as spm

from ..tokenizer import TreebankWordTokenize
from ..utils import check_and_download
from .utils import dp_pos_to_displacy_format, decode_arc_label_vector
from ._spu_context_utils import create_spucontext_dp_model, process_single_word_input

# Resolving parent dependencies
from inspect import getsourcefile
import os
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")

PROD_WEIGHTS_LOC = RESOURCES_PATH + "DP_SPUContext_prod.weights"
EVAL_WEIGHTS_LOC = RESOURCES_PATH + "DP_SPUContext_eval.weights"
WORD_EMBEDDING_MATRIX_LOC = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'resources/SPUTokenized_word_embedding_16k.matrix'))
PROD_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/DP_SPUContext_prod.weights"
EVAL_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/DP_SPUContext_eval.weights"
WORD_EMBEDDING_MATRIX_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/SPUTokenized_word_embedding_16k.matrix"

SPU_TOKENIZER_WORD_LOC = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'resources/SPU_word_tokenizer_16k.model'))
TOKENIZER_LABEL_LOC = RESOURCES_PATH + "DP_label_tokenizer.pickle"

# Data Preprocessing Config
TOKEN_PIECE_MAX_LEN = 8 # 0.9995 quantile is 8 for 16k_vocab, 7 for 32k_vocab
SENTENCE_MAX_LEN = 40 # 0.998 quantile is 42

# Loading Tokenizers
spu_tokenizer_word = spm.SentencePieceProcessor(SPU_TOKENIZER_WORD_LOC)

with open(TOKENIZER_LABEL_LOC, 'rb') as handle:
    tokenizer_label = pickle.load(handle)

sp_key_to_index = {spu_tokenizer_word.id_to_piece(id): id for id in range(spu_tokenizer_word.get_piece_size())}
sp_index_to_key = {id: spu_tokenizer_word.id_to_piece(id) for id in range(spu_tokenizer_word.get_piece_size())}

LABEL_VOCAB_SIZE = len(tokenizer_label.word_index)
WORD_EMBEDDING_VOCAB_SIZE = len(sp_key_to_index)
WORD_EMBEDDING_VECTOR_SIZE = 128
WORD_EMBEDDING_MATRIX = np.zeros((WORD_EMBEDDING_VOCAB_SIZE, WORD_EMBEDDING_VECTOR_SIZE))
ARC_LABEL_VECTOR_LEN = SENTENCE_MAX_LEN + 1 + len(tokenizer_label.word_index) + 1
NUM_RNN_STACKS = 2
RNN_UNITS_MULTIPLIER = 2
NUM_RNN_UNITS = WORD_EMBEDDING_VECTOR_SIZE * RNN_UNITS_MULTIPLIER
FC_UNITS_MULTIPLIER = (2, 1)
DROPOUT = 0.2

class SPUContextDP:
    """
    SentencePiece Unigram Context Dependency Parser class.

    - This is a context aware Deep GRU based Dependency Parser that uses `SentencePiece Unigram <https://arxiv.org/abs/1804.10959>`_ tokenizer and pre-trained Word2Vec embeddings.
    - It achieves 0.7117 LAS (Labeled Attachment Score) and 0.8370 UAS (Unlabeled Attachment Score) on all of test sets of Universal Dependencies 2.9.
    - For more details about the training procedure, dataset and evaluation metrics, see `ReadMe <https://github.com/vngrs-ai/VNLP/blob/main/vnlp/dependency_parser/ReadMe.md>`_.

    """
    def __init__(self, evaluate):
        self.model = create_spucontext_dp_model(TOKEN_PIECE_MAX_LEN, SENTENCE_MAX_LEN, WORD_EMBEDDING_VOCAB_SIZE, ARC_LABEL_VECTOR_LEN,
                                                WORD_EMBEDDING_VECTOR_SIZE, WORD_EMBEDDING_MATRIX,
                                                NUM_RNN_UNITS, NUM_RNN_STACKS, FC_UNITS_MULTIPLIER,
                                                DROPOUT)
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
        # Insert word embedding weights to correct position (0 for SPUContext Dependency Parsing model)
        model_weights.insert(0, word_embedding_matrix)
        # Set model weights
        self.model.set_weights(model_weights)

        self.spu_tokenizer_word = spu_tokenizer_word
        self.tokenizer_label = tokenizer_label

        

    def predict(self, sentence: str, displacy_format: bool = False, pos_result: List[Tuple[str, str]] = None) -> List[Tuple[int, str, int, str]]:
        """
        Args:
            sentence:
                Input sentence.
            displacy_format:
                When set True, returns the result in spacy.displacy format to allow visualization.
            pos_result:
                Part of Speech tags. To be used when displacy_format = True.
        
        Returns:
            List of (token_index, token, arc, label).
                
        Raises:
            ValueError: Sentence is too long. Try again by splitting it into smaller pieces.
        """
        tokenized_sentence = TreebankWordTokenize(sentence)
        num_tokens_in_sentence = len(tokenized_sentence)

        if num_tokens_in_sentence > SENTENCE_MAX_LEN:
            raise ValueError('Sentence is too long. Try again by splitting it into smaller pieces.')

        arcs = []
        labels = []
        for t in range(num_tokens_in_sentence):
            # t is the index of token/word
            X = process_single_word_input(t, tokenized_sentence, self.spu_tokenizer_word, self.tokenizer_label,
                                          ARC_LABEL_VECTOR_LEN, arcs, labels)

            # Predicting
            logits = self.model(X).numpy()[0]
            
            arc, label = decode_arc_label_vector(logits, SENTENCE_MAX_LEN, LABEL_VOCAB_SIZE)

            arcs.append(arc)
            labels.append(label)

        # 0 arc index is reserved for root, therefore arc = 1 means word is dependent on the first word
        dp_result = []
        for idx, word in enumerate(tokenized_sentence):
            dp_result.append((idx + 1, word, arcs[idx], self.tokenizer_label.sequences_to_texts([[labels[idx]]])[0]))

        if not displacy_format:
            return dp_result
        else:
            dp_result_displacy_format = dp_pos_to_displacy_format(dp_result, pos_result)
            return dp_result_displacy_format