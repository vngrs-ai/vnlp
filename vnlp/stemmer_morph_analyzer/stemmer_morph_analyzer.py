from typing import List

import pickle

import tensorflow as tf
import numpy as np

from ..tokenizer import TreebankWordTokenize
from ..utils import check_and_download
from ._melik_utils import create_stemmer_model, process_input_text
from ._yildiz_analyzer import TurkishStemSuffixCandidateGenerator, capitalize

# Resolving parent dependencies
from inspect import getsourcefile
import os
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")

PROD_WEIGHTS_LOC = RESOURCES_PATH + "Stemmer_Shen_prod.weights"
EVAL_WEIGHTS_LOC = RESOURCES_PATH + "Stemmer_Shen_eval.weights"
PROD_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/Stemmer_Shen_prod.weights"
EVAL_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/Stemmer_Shen_eval.weights"

TOKENIZER_CHAR_LOC = RESOURCES_PATH + "Stemmer_char_tokenizer.pickle"
TOKENIZER_TAG_LOC = RESOURCES_PATH + "Stemmer_morph_tag_tokenizer.pickle"

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
    """
    StemmerAnalyzer Class.

    This is a Morphological Disambiguator.

    - This is an implementation of `The Role of Context in Neural Morphological Disambiguation <https://aclanthology.org/C16-1018.pdf>`_.
    - There are slight modifications to the original paper:
    - This version uses GRU instead of LSTM, which decreases the number of parameters by 25% with no actual performance penalty.
    - This version has an extra Dense layer before the output(p) layer.
    - During training, the positions of candidates and labels are shuffled in every batch.
    - It achieves 0.9467 accuracy on ambigious tokens and 0.9664 accuracy on all tokens on trmorph2006 dataset, compared to 0.910 and 0.964 in the original paper.
    - For more details about the implementation, training procedure and evaluation metrics, see `ReadMe <https://github.com/vngrs-ai/VNLP/blob/main/vnlp/stemmer_morph_analyzer/ReadMe.md>`_.
    """
    def __init__(self, evaluate = False):
        with open(TOKENIZER_CHAR_LOC, 'rb') as handle:
            tokenizer_char = pickle.load(handle)
        
        with open(TOKENIZER_TAG_LOC, 'rb') as handle:
            tokenizer_tag = pickle.load(handle)

        CHAR_VOCAB_SIZE = len(tokenizer_char.word_index) + 1
        TAG_VOCAB_SIZE = len(tokenizer_tag.word_index) + 1

        self.model = create_stemmer_model(NUM_MAX_ANALYSIS, STEM_MAX_LEN, CHAR_VOCAB_SIZE, CHAR_EMBED_SIZE, STEM_RNN_DIM,
                                  TAG_MAX_LEN, TAG_VOCAB_SIZE, TAG_EMBED_SIZE, TAG_RNN_DIM,
                                  SENTENCE_MAX_LEN, SURFACE_TOKEN_MAX_LEN, EMBED_JOIN_TYPE, DROPOUT,
                                  NUM_RNN_STACKS)
        # Check and download model weights
        if evaluate:
            MODEL_WEIGHTS_LOC = EVAL_WEIGHTS_LOC
            MODEL_WEIGHTS_LINK = EVAL_WEIGHTS_LINK
        else:
            MODEL_WEIGHTS_LOC = PROD_WEIGHTS_LOC
            MODEL_WEIGHTS_LINK = PROD_WEIGHTS_LINK

        check_and_download(MODEL_WEIGHTS_LOC, MODEL_WEIGHTS_LINK)
        # Load Model weights
        with open(MODEL_WEIGHTS_LOC, 'rb') as fp:
            model_weights = pickle.load(fp)
        # Set model weights
        self.model.set_weights(model_weights)

        self.tokenizer_char = tokenizer_char
        self.tokenizer_tag = tokenizer_tag

        self.candidate_generator = TurkishStemSuffixCandidateGenerator(case_sensitive=True)


    def predict(self, sentence: str, batch_size: int = 64) -> List[str]:
        """
        High level user API for Morphological Disambiguation.

        Args:
            sentence:
                Input text(sentence).
            batch_size:
                batch size (number of tokens to be predicted together.) In case of long sentences, you can increase this for better performance on GPU. If you come across OOM error, decrease until error disappears.

        Returns:
            List of selected stem and morphological tags for each token.

        Example::
        
            from vnlp import StemmerAnalyzer
            stemmer = StemmerAnalyzer()
            stemmer.predict("Üniversite sınavlarına canla başla çalışıyorlardı.")

            ['üniversite+Noun+A3sg+Pnon+Nom',
            'sınav+Noun+A3pl+P3sg+Dat',
            'can+Noun+A3sg+Pnon+Ins',
            'baş+Noun+A3sg+Pnon+Ins',
            'çalış+Verb+Pos+Prog1+A3pl+Past',
            '.+Punc']
        """
        tokens = TreebankWordTokenize(sentence)

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
        X, _ = process_input_text([sentence], self.tokenizer_char, self.tokenizer_tag, STEM_MAX_LEN, TAG_MAX_LEN, SURFACE_TOKEN_MAX_LEN,    
                                  SENTENCE_MAX_LEN, NUM_MAX_ANALYSIS, exclude_unambigious = False, shuffle = False)


        ambig_levels = np.array([len(analyses) for analyses in sentence[1]], dtype = np.int32)

        # Model Output (indices)
        if len(tokens) <= batch_size:
            probs_of_sentence = self.model(X)
        else:
            probs_of_sentence = tf.constant(0, shape = (0, NUM_MAX_ANALYSIS), dtype = tf.float32)
            for idx in range(0, len(tokens), batch_size):
                low = idx
                high = low + batch_size
                probs_ = self.model([X_[low:high] for X_ in X])
                probs_of_sentence = tf.concat((probs_of_sentence, probs_), axis = 0)


        # Below code prevents model from giving output higher than the ambiguity level
        # e.g. in an edge case, when ambiguity is 3, model returned 7.
        # instead of argmax over all indices, now I return argmax of first ambig_level_of_token
        # from output layer
        # -------------------------------------------------------------------------------
        # Vectorized Implementation note:
        # Instead of iterating over logits to exclude indices higher than ambiguity level
        # I create a 2D mask and then argmax, which result in the same outcome
        # This vectorized implementation is MUCH faster (680 ms vs 4 ms)
        ambig_levels_tiled = tf.tile(tf.expand_dims(ambig_levels, axis = 1), [1, NUM_MAX_ANALYSIS])
        tf_range = tf.range(1, NUM_MAX_ANALYSIS+1)
        mask = tf.cast(ambig_levels_tiled >= tf_range, dtype = tf.float32)
        probs_of_sentence  = probs_of_sentence * mask

        # tf.argmax() --> numpy() is faster than numpy() --> np.argmax().
        predicted_indices = tf.argmax(probs_of_sentence, axis = -1).numpy()

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