from typing import List
from pathlib import Path

import numpy as np

# To suppress zero and nan division errors
np.seterr(divide='ignore', invalid='ignore')

PATH = "../resources/"
PATH = str(Path(__file__).parent / PATH)

class StopwordRemover:
    """
    Stopword Remover class.

    Consists of Static and Dynamic stopword detection methods.
    Static stopwords list are taken from https://github.com/ahmetax/trstop and some minor improvements are done by removing numbers from it.

    - Dynamic stopword algorithm is implemented according to two papers below:
        - Saif, Fernandez, He, Alani. 
        “On Stopwords, Filtering and Data Sparsity for Sentiment Analysis of Twitter”.  
        Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC'14), pp. 810–817, 2014.

        - Automatic cut-point of stop-words is determined according to:
        Satopaa, Albrecht, Irwin, Raghavan.
        Detecting Knee Points in System Behavior”.  
        Distributed Computing Systems Workshops (ICDCSW), 2011 31st International Conference, 2011.

    Attributes:
        stop_words: static stopwords list.

    Methods:
        dynamically_detect_stop_words(tokens):
            Returns dynamically detected stop words.
        add_to_stop_words(tokens):
            Updates stop_words dictionary by adding given tokens.
        drop_stop_words(tokens):
            Removes stopwords from given list of tokens and returns the result.
        
    """

    def __init__(self):

        # Loading static stop words from the lexicon
        f = open(PATH + '/turkish_stop_words.txt', encoding = 'utf-8')

        self.stop_words = dict.fromkeys([line.strip() for line in f])

    def dynamically_detect_stop_words(self, list_of_tokens: List[str], rare_words_freq: int = 0) -> List[str]:
        """
        Dynamically detects stop words and returns them as List of strings.

        Args:
            list_of_tokens:
                List of input string tokens
            rare_words_freq:
                Maximum frequency of words when deciding rarity.
                Default value is 0 so it does not detect any rare words by default.

        Returns:
            List of dynamically detected stop words.
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

        Args:
            novel_stop_words:
                Tokens to be updated to existing stop_words dictionary.
        """
        self.stop_words.update(dict.fromkeys(novel_stop_words))

    def drop_stop_words(self, list_of_tokens: List[str]) -> List[str]:
        """
        Given list of tokens, drops stop words and returns list of remaining tokens.

        Args:
            list_of_tokens:
                List of input string tokens.

        Returns:
            List of tokens stripped of stopwords
        """
        tokens_without_stopwords = [token for token in list_of_tokens if token not in self.stop_words]
        return tokens_without_stopwords