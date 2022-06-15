import os
import logging

import requests
import tensorflow as tf
import numpy as np

def check_and_download(file_path, file_url):
    """
    This utility function checks whether the file exists in local directory,
    and downloads it otherwise.
    """
    if not os.path.isfile(file_path):
        logging.warning(f'Downloading model file: {file_url.split("/")[-1]}')
        response = requests.get(file_url)
        if response.ok:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            open(file_path, "wb").write(response.content)
            logging.warning(f'Download completed.')
        else:
            raise ValueError(f'ERROR: {response.status_code} {response.reason}. {file_url.split("/")[-1]} could not be downloaded. Try initializing the model again.')

# SentencePiece Unigram Models Utils
# ===================================
def create_rnn_stacks(NUM_RNN_STACKS, NUM_RNN_UNITS, DROPOUT, GO_BACKWARDS = False):
    rnn_stack = tf.keras.models.Sequential()
    for n in range(NUM_RNN_STACKS - 1):
        rnn_stack.add(tf.keras.layers.GRU(NUM_RNN_UNITS, dropout = DROPOUT, return_sequences = True, go_backwards = GO_BACKWARDS))
    rnn_stack.add(tf.keras.layers.GRU(NUM_RNN_UNITS, dropout = DROPOUT, return_sequences = False, go_backwards = GO_BACKWARDS))
    
    return rnn_stack

def tokenize_single_word(word, tokenizer_word, TOKEN_PIECE_MAX_LEN):
    tokenized = tokenizer_word.encode_as_ids(word)
    padded = tf.keras.preprocessing.sequence.pad_sequences([tokenized], maxlen = TOKEN_PIECE_MAX_LEN,
                                                           padding = 'pre', truncating = 'pre')
    return padded[0]

def process_word_context(w, sentence, tokenizer_word, SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN):
    """
    w: index of currenty processed word
    sentence: tokenized list of strings
    """

    current_word = sentence[w]
    left_context_words = sentence[:w]
    right_context_words = sentence[w+1:]

    # Tokenize and Pad for Current, Left and Right
    current_word_processed = tokenize_single_word(current_word, tokenizer_word, TOKEN_PIECE_MAX_LEN)
    left_context_words_processed = [tokenize_single_word(word, tokenizer_word, TOKEN_PIECE_MAX_LEN) for word in left_context_words]
    right_context_words_processed = [tokenize_single_word(word, tokenizer_word, TOKEN_PIECE_MAX_LEN) for word in right_context_words]

    # 2D Padding for Left
    left_context_ = []
    for _ in range(max(0, SENTENCE_MAX_LEN - len(left_context_words_processed))):
        left_context_.append(np.zeros(TOKEN_PIECE_MAX_LEN))
    left_context_words_processed = np.array(left_context_ + left_context_words_processed)
    left_context_words_processed = left_context_words_processed[-SENTENCE_MAX_LEN:] # pre truncate

    # 2D Padding for Right
    for _ in range(max(0, SENTENCE_MAX_LEN - len(right_context_words_processed))):
        right_context_words_processed.append(np.zeros(TOKEN_PIECE_MAX_LEN))
    right_context_words_processed = np.array(right_context_words_processed)
    right_context_words_processed = right_context_words_processed[:SENTENCE_MAX_LEN] # post truncate
    
    # SentencePiece Tokenizer does not support np.int so I cast to Python int
    current_word_processed = current_word_processed.astype(int)
    left_context_words_processed = left_context_words_processed.astype(int)
    right_context_words_processed = right_context_words_processed.astype(int)
    
    return current_word_processed, left_context_words_processed, right_context_words_processed
        