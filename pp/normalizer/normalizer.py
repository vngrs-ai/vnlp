#!/usr/bin/env python
# coding: utf-8
from typing import List
from pathlib import Path
import string

import pandas as pd

from num2words import num2words
from Levenshtein import distance as levenshtein_distance

from ._deasciifier import Deasciifier
from ..stemmer_morph_analyzer import StemmerAnalyzer

PATH = "../_resources/"
PATH = str(Path(__file__).parent / PATH)

# Normalization according to Levenshtein distance is implemented using a method proposed by
# Göker and Buğlalılar, 
# “NEURAL TEXT NORMALIZATION FOR TURKISH SOCIAL MEDIA”. 
# Hacettepe University, Thesis for Degree of Master of Science in Computer Engineering
# However our method first measures Levenshtein distance for all characters first
# Then measures according to Göker's method.

# Deascification is ported from Emre Sevinç's implementation in https://github.com/emres/turkish-deasciifier

class Normalizer():
    def __init__(self):

        # Multiword Lexicon
        dict_mwe = dict.fromkeys(pd.read_csv(PATH +'/mwe_lexicon.txt', sep = "\n", on_bad_lines="skip", header = None).values.reshape(-1).tolist())

        # General Purpose Typo, Abbreviation, Social Media, etc. Normalization Lexicon
        dict_typo = dict(pd.read_csv(PATH +'/typo_correction_lexicon.txt').values)

        # Word Lexicon merged from TDK-Zemberek, Zargan, Bilkent Creative Writing, Turkish Broadcast News
        dict_words_lexicon = dict.fromkeys(pd.read_csv(PATH + '/turkish_known_words_lexicon.txt', na_filter = False).values.reshape(-1).tolist())

        self._words_lexicon = dict_words_lexicon
        self._mwe_lexicon = dict_mwe
        self._typo_lexicon = dict_typo
        
        self._vowels = set("aeiou")
        self._consonants = set(string.ascii_lowercase) - self._vowels
        self._non_turkish_accent_marks = {'â':'a', 'ô':'o', 'î':'ı', 'ê':'e', 'û':'u'}

        self._stemmer_analyzer = StemmerAnalyzer()
    

    def remove_punctuations(self, text: str)-> str:
        return ''.join([t for t in text if t not in string.punctuation])
    
    def convert_number_to_word(self, tokens: List[str])-> List[str]:
        converted_tokens = []
        for token in tokens:
            if token.isnumeric():
                converted_tokens.append(num2words(float(token), lang = 'tr'))
            else:
                converted_tokens.append(token)

        return converted_tokens
    
    def remove_accent_marks(self, text: str)-> str:
        return ''.join(self._non_turkish_accent_marks.get(char, char) for char in text)
    
    def correct_typos(self, tokens: List[str], use_levenshtein: bool = False) -> List[str]:
        corrected_tokens = []
        for token in tokens:
            if self._is_token_valid_turkish(token):
                corrected_tokens.append(token)
            elif token in self._typo_lexicon:
                corrected_tokens.append(self._typo_lexicon[token])
            elif use_levenshtein:
                corrected_tokens.append(self._return_most_similar_word(token))
            else:
                corrected_tokens.append(token)
        
        return corrected_tokens

    def deasciify(self, tokens: List[str]) -> List[str]:
        deasciified_tokens = []
        for token in tokens:
            deasciifier = Deasciifier(token)
            deasciified_tokens.append(deasciifier.convert_to_turkish())
        return deasciified_tokens
    

    def _return_most_similar_word(self, s1):
        s1_consonant = "".join([l for l in s1 if l in self._consonants])
        distance_list = []
        consonant_distance_list = []
        s2_list = []    # to keep track of known words when measuring similarity because dict is unordered

        for s2 in self._words_lexicon:
            s2_list.append(s2)

            dist = levenshtein_distance(s1, s2)
            distance_list.append(dist)

            s2_consonant = "".join([l for l in s2 if l in self._consonants])
            consonant_dist = levenshtein_distance(s1_consonant, s2_consonant)
            consonant_distance_list.append(consonant_dist)

        # This can be modified to idxmin of distance only to speedup a bit
        # But this sort part takes 1% time of this function
        df_sorted = pd.DataFrame({'Distance': distance_list, 'Consonant_Distance': consonant_distance_list}, index = s2_list).sort_values(by = ['Consonant_Distance', 'Distance'])
        most_similar_word = df_sorted.index[0]

        return most_similar_word    

    def _is_token_valid_turkish(self, token):
        valid_according_to_stemmer_analyzer = not (self._stemmer_analyzer.candidate_generator.get_analysis_candidates(token)[0][-1] == 'Unknown')
        valid_according_to_lexicon = token in self._words_lexicon
        return valid_according_to_stemmer_analyzer or valid_according_to_lexicon
        
        """
        Given list of tokens, returns list of normalized tokens

        Args:
        remove_punctuations (bool): Whether to remove punctuations 
        "merhaba." -> "merhaba"

        convert_to_lowercase (bool): Whether to convert letters to lowercase
        "Merhaba" -> "merhaba"

        convert_number_to_word (bool): Whether to convert numbers from numeric form to text form
        "3" -> "üç"

        remove_accent_marks (bool): Whether to remove accent marks
        "merhâbâ" -> "merhaba"

        normalize_via_lexicon (bool): Whether to normalize spelling mistakes/typos
        according looking into up pre-defined typo lexicon.
        
        normalize_via_levenshtein (bool): Whether to use levenshtein distance to normalize unknown words
        according to its similarity to known words. Calculates 2 distances:
        First calculates levenshtein distance for all characters.
        Then calculates levenshtein distance for consonants only.
        This part is implemented based on “NEURAL TEXT NORMALIZATION FOR TURKISH SOCIAL MEDIA”.

        deascify (bool): Whether to de-ascify characters for Turkish words.
        "dusunuyorum" -> "düşünüyorum"
        Deascifier is ported from Emre Sevinç's implementation in https://github.com/emres/turkish-deasciifier
        """
