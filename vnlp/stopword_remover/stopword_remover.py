from typing import List
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

        self.stop_words = dict.fromkeys([line.strip() for line in f])

    def dynamically_detect_stop_words(self, list_of_tokens: List[str], rare_words_freq: int = 0) -> List[str]:
        """
        Dynamically detects stop words and returns them as List of strings.
        Args:
            rare_words_freq (int): Maximum frequency of words when deciding rarity.
            Default value is 0 so it does not detect any rare words by default.

        Input:
        list_of_tokens(List[str]): list of string tokens

        Output:
        detected_stop_words(List[str]): list of string tokens
        """
        unq, cnts = np.unique(list_of_tokens, return_counts = True)
        sorted_indices = cnts.argsort()[::-1] # I need them in descending order
        unq = unq[sorted_indices]
        cnts = cnts[sorted_indices]
        
        if len(unq) < 3:
            raise ValueError('Number of unique tokens must be at least 3 for Dynamic Stop Word Detection')
            
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
        detected_stop_words = unq[:argmax_second_der + 2].tolist()
        
        # Determine rare_words according to given rare_words_freq value
        # Add them to dynamic_stop_words list
        rare_words = unq[cnts <= rare_words_freq].tolist()
        detected_stop_words += rare_words

        return detected_stop_words

    def add_to_stop_words(self, novel_stop_words: List[str]):
        """
        Updates self.stop_words by adding given novel_stop_words to existing dictionary.
        """
        self.stop_words.update(dict.fromkeys(novel_stop_words))

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