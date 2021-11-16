from typing import List
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from .. import _utils as utils

PATH = "../_resources/"
PATH = str(Path(__file__).parent / PATH)

class StopwordRemover:

    def __init__(self):
        self.static_stop_words = utils.load_words(PATH + '/turkish_stop_words.txt')
        self.dynamic_stop_words = []
        self.stop_words = self.static_stop_words.copy()
        self.rare_words = []

    def dynamically_detect_stop_words(self, list_of_tokens: List[str], rare_words_freq: int = 0):
        """
        Dynamically detects stop words and updates self.dynamic_stop_words list.
        Args:
            rare_words_freq (int): Maximum frequency of words when deciding rarity.
            Default value is 0 so it does not detect and drop any rare words.

        Input:
        list_of_tokens(List[str]): list of string tokens
        """
        unq, cnts = np.unique(list_of_tokens, return_counts = True)
        
        if len(unq) < 3:
            raise ValueError('Number of tokens must be at least 3 for Dynamic Stop Word Detection')
            
        df_words = pd.DataFrame({'word': unq, 'counts': cnts}).sort_values(by = 'counts', ascending = False).reset_index(drop = True)
        
        # Adds most frequent words to stop_words list
        argmax_second_der = df_words['counts'].pct_change().abs().pct_change().abs().dropna().idxmax()
        stop_words_extracted = df_words.loc[:argmax_second_der, 'word'].values.tolist()
        self.dynamic_stop_words += [x for x in stop_words_extracted if x not in self.dynamic_stop_words]
        
        # Determine rare_words according to given rare_words_freq value
        # Add them to dynamic_stop_words list
        rare_words = df_words.loc[df_words['counts'] <= rare_words_freq, 'word'].values.tolist()
        stop_words_extracted += rare_words
        self.rare_words += rare_words
        self.dynamic_stop_words += self.rare_words
        
        logging.info("Dynamically detected stopwords are: " + ", ".join(stop_words_extracted))

    def unify_stop_words(self):
        """
        Updates self.stop_words by merging static and dynamic ones.
        """
        self.stop_words = sorted(list(set(self.static_stop_words).union(set(self.dynamic_stop_words))))
        logging.info('List of stop words is unified and updated.')

    def drop_stop_words(self, list_of_tokens: List[str]) -> List[str]:
        """
        Given list of tokens, returns list of tokens without drop words.

        Input:
        list_of_tokens(List[str]): list of string tokens

        Output:
        tokens_without_stopwords(List[str]): list of string tokens, stripped of stopwords
        """
        
        tokens_without_stopwords = [token for token in list_of_tokens if token not in self.stop_words]
        return tokens_without_stopwords