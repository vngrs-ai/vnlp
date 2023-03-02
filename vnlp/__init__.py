import os
# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
# Prevent tensorflow from allocating whole GPU memory.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from .dependency_parser import DependencyParser
from .named_entity_recognizer import NamedEntityRecognizer
from .normalizer import Normalizer
from .part_of_speech_tagger import PoSTagger
from .sentence_splitter import SentenceSplitter
from .sentiment_analyzer import SentimentAnalyzer
from .stemmer_morph_analyzer import StemmerAnalyzer
from .stopword_remover import StopwordRemover

__all__ = ['DependencyParser', 'NamedEntityRecognizer', 'Normalizer', 
           'PoSTagger', 'SentenceSplitter', 'SentimentAnalyzer', 
           'StemmerAnalyzer', 'StopwordRemover']
