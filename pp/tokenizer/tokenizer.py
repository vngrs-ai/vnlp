__version__ = '0.0.0.1'

from typing import Tuple, Union

import regex as re

class WordTokenizer:
    __slots__ = 'pre_compiled_regexes'

    def __init__(self):
        suffixes: str
        numbers: str
        any_word: str
        punctuations: str

        suffixes = r"[a-zğçşöüı]{3,}' ?[a-zğçşöüı]+"
        numbers = r"%\d{2,}[.,:/\d-]+"
        any_word = r"[a-zğçşöüı_+%\.()@&`’/\\\d-]+"
        punctuations = r"[a-zğçşöüı]*[,!?;:]"
        self.pre_compiled_regexes = re.compile(
            "|".join(
                [suffixes,
                 numbers,
                 any_word,
                 punctuations
                 ]
            ), re.I
        )

    def tokenize(self, sentence: str) -> Tuple:
        words: Union[list, tuple]
        dots: str

        try:
            words = self.pre_compiled_regexes.findall(sentence)
        except (re.error, TypeError):
            return ()
        else:
            # If last word ends with dot, it should be another word
            words = tuple(words)
            if words:
                end_dots = re.search(r'\b(\.+)$', words[-1])
                if end_dots:
                    dots = end_dots.group(1)
                    words = words[:-1] + (words[-1][:-len(dots)],) + (dots,)
            return words


# Tokenizer can be either updated extensively or simply removed
# since tokenization needs vary a lot depending on the task
"""
from nltk.tokenize import WordPunctTokenizer, MWETokenizer
f = open("mwe_lexicon.txt", 'r', encoding = 'utf-8')
mwe = [line.rstrip().strip('\ufeff') for line in f]
sentence= "Gelirken istanbul'a türk hava yolları'nı kullanarak"
tokens = WordPunctTokenizer().tokenize(sentence)
mwe_tokenizer = MWETokenizer()

tuples_of_mwe = [tuple(m.split()) for m in mwe]
for _tuple in tuples_of_mwe:
    mwe_tokenizer.add_mwe(_tuple)
    
mwe_tokenizer.tokenize(tokens)

"""