#!/usr/bin/env python
# coding: utf-8
from typing import List
from pathlib import Path
import string

import pandas as pd

from num2words import num2words

from ._deasciifier import Deasciifier

PATH = "../_resources/"
PATH = str(Path(__file__).parent / PATH)


# https://stackoverflow.com/questions/2460177/edit-distance-in-python
def _levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

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
        dict_words_lexicon = dict.fromkeys(pd.read_csv(PATH + '/turkish_known_words_lexicon.csv', na_filter = False).values.reshape(-1).tolist())

        self._words_lexicon = dict_words_lexicon
        self._mwe_lexicon = dict_mwe
        self._typo_lexicon = dict_typo
        
        self._vowels = set("aeiou")
        self._consonants = set(string.ascii_lowercase) - self._vowels
        self._non_turkish_accent_marks = {'â':'a', 'ô':'o', 'î':'ı', 'ê':'e', 'û':'u'}
    
    def _remove_punctuations(self, token):
        return ''.join([t for t in token if t not in string.punctuation])
    
    def _convert_to_lower_case(self, token):
        return token.lower()
    
    def _convert_number_to_word(self, number):
        return num2words(number, lang = 'tr')
    
    def _remove_accent_marks(self, token):
        return ''.join(self._non_turkish_accent_marks.get(char, char) for char in token)
    """
    def multiword_replace(self, token):
        if token in self.mwe_lexicon['corrected_expression'].values:
            return token
        elif token in self.mwe_lexicon['original_expression'].values:
            return self.mwe_lexicon['corrected_expression'][self.mwe_lexicon['original_expression'] == token].values[0].split(' ')
        else:
            return token
    """
    
    def _general_purpose_normalize_by_lexicon(self, token):
        if token in self._words_lexicon:
            return token
        elif token is self._typo_lexicon:
            return self._typo_lexicon[token]
        else:
            return token
    
    def _return_most_similar_word(self, s1):
        s1_consonant = "".join([l for l in s1 if l in self._consonants])
        distance_list = []
        consonant_distance_list = []
        s2_list = []    # to keep track of known words when measuring similarity because dict is unordered

        for s2 in self._words_lexicon:
            s2_list.append(s2)

            dist = _levenshtein_distance(s1, s2)
            distance_list.append(dist)

            s2_consonant = "".join([l for l in s2 if l in self._consonants])
            consonant_dist = _levenshtein_distance(s1_consonant, s2_consonant)
            consonant_distance_list.append(consonant_dist)

        # This can be modified to idxmin of distance only to speedup a bit
        # But this sort part takes 1% time of this function
        df_sorted = pd.DataFrame({'Distance': distance_list, 'Consonant_Distance': consonant_distance_list}, index = s2_list).sort_values(by = ['Consonant_Distance', 'Distance'])
        most_similar_word = df_sorted.index[0]

        return most_similar_word
        
    # High Level Function
    def normalize(self, list_of_tokens: List[str], remove_punctuations: bool = True, 
                  convert_to_lowercase: bool = True, convert_number_to_word: bool = True, 
                  remove_accent_marks: bool = True, normalize_via_lexicon: bool = True,
                  normalize_via_levenshtein: bool = False, deascify: bool = False) -> List[str]:
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
            
            use_levenshtein (bool): Whether to use levenshtein distance to normalize unknown words
            according to its similarity to known words. Calculates 2 distances:
            First calculates levenshtein distance for all characters.
            Then calculates levenshtein distance for consonants only.
            This part is implemented based on “NEURAL TEXT NORMALIZATION FOR TURKISH SOCIAL MEDIA”.

            deascify (bool): Whether to de-ascify characters for Turkish words.
            "dusunuyorum" -> "düşünüyorum"
            Deascifier is ported from Emre Sevinç's implementation in https://github.com/emres/turkish-deasciifier
        """
        normalized_list_of_tokens = []
        
        for token in list_of_tokens:
            if remove_punctuations:
                token = self._remove_punctuations(token)
            if convert_to_lowercase:
                token = self._convert_to_lower_case(token)
            if token in self._mwe_lexicon:
                return token
            
            if token.isnumeric() & convert_number_to_word:
                token = self._convert_number_to_word(float(token))
                normalized_list_of_tokens.append(token)
                continue
            
            if remove_accent_marks:
                token = self._remove_accent_marks(token)

            if deascify:
                deascifier = Deasciifier(token)
                token = deascifier.convert_to_turkish()
                
            if normalize_via_lexicon:
                token = self._general_purpose_normalize_by_lexicon(token)
        
            if (token not in self._words_lexicon) & (normalize_via_levenshtein):
                token = self._return_most_similar_word(token)
            if token != '':
                normalized_list_of_tokens.append(token)
        
        return normalized_list_of_tokens