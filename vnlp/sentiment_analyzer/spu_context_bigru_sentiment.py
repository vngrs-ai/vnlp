from typing import List, Tuple

import pickle

import numpy as np

import sentencepiece as spm

from ..utils import check_and_download
from ._spu_context_bigru_utils import create_spucbigru_sentiment_model, process_text_input

# Resolving parent dependencies
from inspect import getsourcefile
import os
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")

PROD_WEIGHTS_LOC = RESOURCES_PATH + "Sentiment_SPUCBiGRU_prod.weights"
EVAL_WEIGHTS_LOC = RESOURCES_PATH + "Sentiment_SPUCBiGRU_eval.weights"
WORD_EMBEDDING_MATRIX_LOC = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'resources/SPUTokenized_word_embedding_16k.matrix'))
PROD_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/Sentiment_SPUCBiGRU_prod.weights"
EVAL_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/Sentiment_SPUCBiGRU_eval.weights"
WORD_EMBEDDING_MATRIX_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/SPUTokenized_word_embedding_16k.matrix"

SPU_TOKENIZER_WORD_LOC = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'resources/SPU_word_tokenizer_16k.model'))

# Data Preprocessing Config
TEXT_MAX_LEN = 256

# Loading Tokenizer
spu_tokenizer_word = spm.SentencePieceProcessor(SPU_TOKENIZER_WORD_LOC)

sp_key_to_index = {spu_tokenizer_word.id_to_piece(id): id for id in range(spu_tokenizer_word.get_piece_size())}
sp_index_to_key = {id: spu_tokenizer_word.id_to_piece(id) for id in range(spu_tokenizer_word.get_piece_size())}

WORD_EMBEDDING_VOCAB_SIZE = len(sp_key_to_index)
WORD_EMBEDDING_VECTOR_SIZE = 128
WORD_EMBEDDING_MATRIX = np.zeros((WORD_EMBEDDING_VOCAB_SIZE, WORD_EMBEDDING_VECTOR_SIZE))
NUM_RNN_STACKS = 3
NUM_RNN_UNITS = 128
DROPOUT = 0.2

class SPUCBiGRUSentimentAnalyzer:
    """
    SentencePiece Unigram Context Bidirectional GRU Sentiment Analyzer class.

    - This is a Bidirectional `GRU <https://arxiv.org/abs/1412.3555>`_ based Sentiment Analyzer that uses `SentencePiece Unigram <https://arxiv.org/abs/1804.10959>`_ tokenizer and pre-trained Word2Vec embeddings.
    - It achieves 0.9469 Accuracy, 0.9380 F1 macro score and 0.9147 F1 score (treating class 0 as minority).
    - For more details about the training procedure, dataset and evaluation metrics, see `ReadMe <https://github.com/vngrs-ai/VNLP/blob/main/vnlp/sentiment_analyzer/ReadMe.md>`_.

    """
    def __init__(self, evaluate):
        self.model = create_spucbigru_sentiment_model(TEXT_MAX_LEN, WORD_EMBEDDING_VOCAB_SIZE, WORD_EMBEDDING_VECTOR_SIZE, WORD_EMBEDDING_MATRIX,
                                                    NUM_RNN_UNITS, NUM_RNN_STACKS, DROPOUT)
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
        # Insert word embedding weights to correct position (0 for SPUBiGRUSentimentAnalyzer model)
        model_weights.insert(0, word_embedding_matrix)
        # Set model weights
        self.model.set_weights(model_weights)

        self.spu_tokenizer_word = spu_tokenizer_word

    def predict(self, text: str) -> List[Tuple[str, str]]:
        """
        Args:
            text: 
                Input text.

        Returns:
             Sentiment label of input text.
        """

        prob = self.predict_proba(text)

        return 1 if prob > 0.5 else 0

    def predict_proba(self, text: str) -> float:
        """
        Args:
            text: 
                Input text.

        Returns:
            Probability that the input text has positive sentiment.
        """

        tokenized_text = process_text_input(text, self.spu_tokenizer_word, TEXT_MAX_LEN)
        num_int_tokens = len(tokenized_text[0])
        num_str_tokens = len(text.split())

        # if the text is longer than the length the model is trained on
        if num_int_tokens > TEXT_MAX_LEN:
            first_half_of_preprocessed_text = " ".join(text.split()[:(num_str_tokens // 2)])
            second_half_of_preprocessed_text = " ".join(text.split()[(num_str_tokens // 2):])
            prob = (self.predict_proba(first_half_of_preprocessed_text) + self.predict_proba(second_half_of_preprocessed_text)) / 2

        else:
            prob = self.model(tokenized_text).numpy()[0][0]
        
        return prob