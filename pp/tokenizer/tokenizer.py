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
