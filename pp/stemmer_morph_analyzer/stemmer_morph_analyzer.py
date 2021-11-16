from typing import List

import pickle

import numpy as np
from nltk.tokenize import WordPunctTokenizer

from ._melik_utils import create_model, process_data
from ._yildiz_analyzer import TurkishStemSuffixCandidateGenerator, capitalize

import os
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
        tokens = WordPunctTokenizer().tokenize(input_sentence)

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

        # Model Output (indices)
        predicted_indices = np.argmax(self.model.predict(X), axis = -1)

        # If ambiguity level is 1, override model output with 0
        ambig_levels = np.array([len(analyses) for analyses in sentence[1]])
        predicted_indices[ambig_levels == 1] = 0

        # Obtaining Result
        result = []
        for idx, analyses_of_token in enumerate(sentence[1]):
            result.append(analyses_of_token[predicted_indices[idx]])

        # Slight update to Result
        for idx, r in enumerate(result):
            splitted = r.split("+")
            root = splitted[0]
            tags = splitted[1:]

            if "Prop" in tags:
                root = capitalize(root)
                result[idx] = "+".join([root] + tags)
                
            result[idx] = result[idx].replace('+DB', '^DB')

        return result