from typing import List
import logging
from pathlib import Path

import numpy as np

# To suppress zero and nan division errors
np.seterr(divide='ignore', invalid='ignore')

PATH = "../resources/"
PATH = str(Path(__file__).parent / PATH)

class StopwordRemover:

    def __init__(self):

        # Loading static stop words from the lexicon
        f = open(PATH + '/turkish_stop_words.txt', encoding = 'utf-8')

        self.static_stop_words = [line.strip() for line in f]
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
        sorted_indices = cnts.argsort()[::-1] # I need them in descending order
        unq = unq[sorted_indices]
        cnts = cnts[sorted_indices]
        
        if len(unq) < 3:
            raise ValueError('Number of tokens must be at least 3 for Dynamic Stop Word Detection')
            
        # Below is equivalent to:
        # df_words['counts'].pct_change().abs().pct_change().abs().dropna().idxmax()
        
        # First deriv
        diffs_one = np.diff(cnts)
        pct_change_one = np.abs(diffs_one / cnts[:-1])
        # Second deriv
        diffs_two = np.diff(pct_change_one)
        pct_change_two = np.abs(diffs_two / pct_change_one[:-1])
        pct_change_two = pct_change_two[~np.isnan(pct_change_two)] # removing nan
        argmax_second_der = np.argmax(pct_change_two)
        
        # +2 is due to shifting twice due to np.diff()
        stop_words_extracted = unq[:argmax_second_der + 2].tolist()
        self.dynamic_stop_words += [x for x in stop_words_extracted if x not in self.dynamic_stop_words]
        
        # Determine rare_words according to given rare_words_freq value
        # Add them to dynamic_stop_words list
        rare_words = unq[cnts <= rare_words_freq].tolist()
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