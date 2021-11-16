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
        """
        Removes punctuations from the given string.

        Input:
        text(str): string of text

        Output:
        text(str): string of text stripped from punctuations

        Sample use:
        normalizer = Normalizer()
        print(normalizer.remove_punctuations("merhaba..."))

        merhaba
        """
        return ''.join([t for t in text if t not in string.punctuation])
    
    def convert_number_to_word(self, tokens: List[str])-> List[str]:
        """
        Converts numbers to word forms in a given list of tokens.

        Input:
        tokens(List[str]): List of strings of tokens

        Output:
        converted_tokens(List[str]): List of strings of converted tokens

        Sample use:
        normalizer = Normalizer()
        print(normalizer.convert_number_to_word("bugün 3 yumurta yedim".split()))

        ["bugün", "üç", "yumurta", "yedim"]
        """
        converted_tokens = []
        for token in tokens:
            if token.isnumeric():
                converted_tokens.append(num2words(float(token), lang = 'tr'))
            else:
                converted_tokens.append(token)

        return converted_tokens
    
    def remove_accent_marks(self, text: str)-> str:
        """
        Removes accent marks from the given string.

        Input:
        text(str): string of text

        Output:
        text(str): string of text stripped from accent marks

        Sample use:
        normalizer = Normalizer()
        print(normalizer.remove_accent_marks("merhâbâ"))

        merhaba
        """
        return ''.join(self._non_turkish_accent_marks.get(char, char) for char in text)
    
    def correct_typos(self, tokens: List[str], use_levenshtein: bool = False) -> List[str]:
        """
        Corrects spelling mistakes and typos.
        Args:
        use_levenshtein(bool): Whether to use levenshtein distance measure to find the correct word

        Input:
        tokens(List[str]): list of tokens

        Output:
        corrected_tokens(List[str]): list of corrected tokens

        Sample use:
        normalizer = Normalizer()
        print(normalizer.correct_typos("Kasıtlı yazişm hatasıı ekliyoruum".split()))

        ["Kasıtlı", "yazım", "hatası", "ekliyorum"]
        """
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
        """
        Deasciification for Turkish.

        Input:
        tokens(List[str]): list of tokens

        Output:
        deasciified_tokens(List[str]): list of deasciified tokens

        Sample use:
        normalizer = Normalizer()
        print(normalizer.deasciify("dusunuyorum da boyle sey gormedim duymadim".split()))

        ["düşünüyorum", "da", "böyle", "şey", "görmedim", "duymadım"]
        """
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