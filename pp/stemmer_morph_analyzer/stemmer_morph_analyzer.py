from typing import List

import pickle

import numpy as np

from ._melik_utils import create_model, process_data
from ._yildiz_analyzer import TurkishStemSuffixCandidateGenerator, capitalize

# Resolving parent dependencies
from inspect import getsourcefile
import os
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from tokenizer import TreebankWordTokenize

RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")

MODEL_LOC = RESOURCES_PATH + "model_weights.hdf5"
TOKENIZER_CHAR_LOC = RESOURCES_PATH + "tokenizer_char.pickle"
TOKENIZER_TAG_LOC = RESOURCES_PATH + "tokenizer_tag.pickle"

# Data Processing Config
NUM_MAX_ANALYSIS = 10 # 0.99 quantile
STEM_MAX_LEN = 10 # 0.99 quantile
TOKENIZER_CHAR_OOV = '<OOV>'

TAG_MAX_LEN = 15 # 0.99 quantile
TOKENIZER_TAG_OOV = '<OOV>'

SENTENCE_MAX_LEN = 40 # 0.95 quantile is 42
SURFACE_TOKEN_MAX_LEN = 15 # 0.99 quantile

# Model Config
CHAR_EMBED_SIZE = 32
TAG_EMBED_SIZE = 32
STEM_RNN_DIM = 128
TAG_RNN_DIM = 128
NUM_RNN_STACKS = 1
DROPOUT = 0.2
EMBED_JOIN_TYPE = 'add'

CAPITALIZE_PNONS = False


class StemmerAnalyzer:
    def __init__(self):
        with open(TOKENIZER_CHAR_LOC, 'rb') as handle:
            tokenizer_char = pickle.load(handle)
        
        with open(TOKENIZER_TAG_LOC, 'rb') as handle:
            tokenizer_tag = pickle.load(handle)

        CHAR_VOCAB_SIZE = len(tokenizer_char.word_index) + 1
        TAG_VOCAB_SIZE = len(tokenizer_tag.word_index) + 1

        self.model = create_model(NUM_MAX_ANALYSIS, STEM_MAX_LEN, CHAR_VOCAB_SIZE, CHAR_EMBED_SIZE, STEM_RNN_DIM,
                                  TAG_MAX_LEN, TAG_VOCAB_SIZE, TAG_EMBED_SIZE, TAG_RNN_DIM,
                                  SENTENCE_MAX_LEN, SURFACE_TOKEN_MAX_LEN, EMBED_JOIN_TYPE, DROPOUT,
                                  NUM_RNN_STACKS)
        self.model.load_weights(MODEL_LOC)

        self.tokenizer_char = tokenizer_char
        self.tokenizer_tag = tokenizer_tag

        self.candidate_generator = TurkishStemSuffixCandidateGenerator(case_sensitive=True)


    def predict(self, input_sentence: str) -> List[str]:
        """
        High level user function

        Input:
        input_sentence (str): string of text(sentence)

        Output:
        result (List[str]): list of selected morphological analysis for each token

        Sample use:
        from pp.stemmer_morph_analyzer import StemmerAnalyzer
        stemmer = StemmerAnalyzer()
        sentence = ma.predict("Eser miktardaki geçici bir güvenlik için temel özgürlüklerinden vazgeçenler, ne özgürlüğü ne de güvenliği hak ederler. Benjamin Franklin")
        stemmer.predict(sentence)

        ['eser+Noun+A3sg+Pnon+Nom',
        'miktar+Noun+A3sg+Pnon+Loc^DB+Adj+Rel',
        'geçici+Adj',
        'bir+Det',
        'güvenlik+Noun+A3sg+Pnon+Nom',
        'için+Postp+PCNom',
        'temel+Noun+A3sg+Pnon+Nom',
        'özgür+Adj^DB+Noun+Ness+A3sg+P3pl+Abl',
        'vazgeç+Verb+Pos^DB+Adj+PresPart^DB+Noun+Zero+A3pl+Pnon+Nom',
        ',+Punc',
        'ne+Adj',
        'özgür+Adj^DB+Noun+Ness+A3sg+P3sg+Nom',
        'ne+Adj',
        'de+Conj',
        'güvenlik+Noun+A3sg+P3sg+Nom',
        'hak+Noun+A3sg+Pnon+Nom',
        'et+Verb+Pos+Aor+A3pl',
        '.+Punc',
        'benjamin+Noun+A3sg+Pnon+Nom',
        'franklin+Noun+A3sg+Pnon+Nom']
        
        """
        tokens = TreebankWordTokenize(input_sentence)

        # Obtaining Analyses
        sentence = [[], []]
        for token in tokens:
            sentence[0].append(token)
            candidate_analyzes = self.candidate_generator.get_analysis_candidates(token)
            
            root_tags = []
            for analysis in candidate_analyzes:
                root = analysis[0]
                tags = analysis[2]
                if isinstance(tags, str):
                    tags = [tags]
                joined_root_tags = "+".join([root] + tags)
                root_tags.append(joined_root_tags)
                
            sentence[1].append(root_tags)

        # Tokenizing Input
        X, _ = process_data([sentence], self.tokenizer_char, self.tokenizer_tag, STEM_MAX_LEN, TAG_MAX_LEN, SURFACE_TOKEN_MAX_LEN,    
                                  SENTENCE_MAX_LEN, NUM_MAX_ANALYSIS, exclude_unambigious = False, shuffle = False)


        ambig_levels = np.array([len(analyses) for analyses in sentence[1]])

        # Model Output (indices)
        probs_of_sentence = self.model.predict(X)
        predicted_indices = []
        for idx, probs_of_single_token in enumerate(probs_of_sentence):
            ambig_level_of_token = ambig_levels[idx]
            # Below code prevents model from giving output higher than the ambiguity level
            # e.g. in an edge case, when ambiguity is 3, model returned 7.
            # instead of argmax over all indices, now I return argmax of first ambig_level_of_token
            # from output layer
            probs_of_single_token = probs_of_single_token[:ambig_level_of_token]
            predicted_indice_for_token = np.argmax(probs_of_single_token)
            predicted_indices.append(predicted_indice_for_token)

        predicted_indices = np.array(predicted_indices)

        # Obtaining Result
        result = []
        for idx, analyses_of_token in enumerate(sentence[1]):
            result.append(analyses_of_token[predicted_indices[idx]])

        # Slight update to Result
        for idx, r in enumerate(result):
            splitted = r.split("+")
            root = splitted[0]
            tags = splitted[1:]

            if ("Prop" in tags) and CAPITALIZE_PNONS:
                root = capitalize(root)
                result[idx] = "+".join([root] + tags)
                
            result[idx] = result[idx].replace('+DB', '^DB')

        return result