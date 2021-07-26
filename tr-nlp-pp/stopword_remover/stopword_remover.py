from pathlib import Path

import pandas as pd
import numpy as np

from .. import _utils as utils

PATH = "../_resources/"
PATH = str(Path(__file__).parent / PATH)

# Static stopwords list are taken from https://github.com/ahmetax/trstop

# Dynamic stopword algorithm is implemented according to:
# Saif, Fernandez, He, Alani. 
# “On Stopwords, Filtering and Data Sparsity for Sentiment Analysis of Twitter”.  
# Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC'14), pp. 810–817, 2014.

class StopwordRemover:

    def __init__(self):
        self.static_stop_words = utils.load_words(PATH + '/turkce_stop_words.txt')
        self.dynamic_stop_words = []
        self.stop_words = self.static_stop_words.copy()

    def dynamically_detect_stop_words(self, list_of_tokens, drop_rare_words = False):
        """
        Dynamically detects stop words and updates self.dynamic_stop_words list.
        drop_rare_words: flag to drop words with frequency of 1.
        """
        unq, cnts = np.unique(list_of_tokens, return_counts = True)
        
        if len(unq) < 3:
            raise ValueError('Number of tokens must be at least 3 for Dynamic Stop Word Detection')
            
        df_words = pd.DataFrame({'word': unq, 'counts': cnts}).sort_values(by = 'counts', ascending = False).reset_index(drop = True)
        
        # Adds most frequent words to stop_words list
        argmax_second_der = df_words['counts'].pct_change().abs().pct_change().abs().dropna().idxmax()
        stop_words_extracted = df_words.loc[:argmax_second_der, 'word'].values.tolist()
        self.dynamic_stop_words += [x for x in stop_words_extracted if x not in self.dynamic_stop_words]
        
        # Adds frequency 1 words to stop_words list
        if drop_rare_words:
            self.stop_words += df_words.loc[df_words['counts'] == 1, 'word'].values.tolist()
        
        print("Dynamically detected stopwords are:", stop_words_extracted)

    def unify_stop_words(self):
        """
        Updates self.stop_words by merging static and dynamic ones.
        """
        self.stop_words = sorted(list(set(self.static_stop_words).union(set(self.dynamic_stop_words))))
        print('List of stop words is unified and updated.')

    def drop_stop_words(self, list_of_tokens):
        """
        Given list of tokens, returns list of tokens without drop words.
        """
        return [token for token in list_of_tokens if token not in self.stop_words]