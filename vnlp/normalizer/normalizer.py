from typing import List
from pathlib import Path
import string

from hunspell import Hunspell

from ._deasciifier import Deasciifier
from ..stemmer_morph_analyzer import StemmerAnalyzer

RESOURCES_PATH = "../resources/"
RESOURCES_PATH = str(Path(__file__).parent / RESOURCES_PATH)

class Normalizer():
    def __init__(self):

        # Word Lexicon merged from TDK-Zemberek, Zargan, Bilkent Creative Writing, Turkish Broadcast News
        with open(RESOURCES_PATH +'/turkish_known_words_lexicon.txt', 'r', encoding = 'utf-8') as f:
            words_lexicon = [line.strip() for line in f]
        dict_words_lexicon = dict.fromkeys(words_lexicon)

        self._words_lexicon = dict_words_lexicon

        self._stemmer_analyzer = StemmerAnalyzer()
        self._hunspell = Hunspell('tr_TR', hunspell_data_dir= RESOURCES_PATH + '/tdd-hunspell-tr-1.1.0')

    @staticmethod
    def lower_case(text: str) -> str:
        """
        Converts a string of text to lowercase for Turkish language.
        This is needed because Python does not properly handle all Turkish characters. e.g "İ" -> "i"

        Input:
        text(str): string of text

        Output:
        text(str): string of text in lowercase form

        Sample use:
        normalizer = Normalizer()
        print(normalizer.lower_case("Test karakterleri: İIĞÜÖŞÇ"))
        
        'test karakterleri: iığüöşç'

        """
        turkish_lowercase_dict = {"İ": "i", "I": "ı", "Ğ": "ğ", "Ü": "ü", "Ö": "ö", "Ş": "ş", "Ç": "ç"}
        for k, v in turkish_lowercase_dict.items():
            text = text.replace(k, v)

        return text.lower()
    
    @staticmethod
    def remove_punctuations(text: str)-> str:
        """
        Removes punctuations from the given string.

        Input:
        text(str): string of text

        Output:
        text(str): string of text stripped from punctuations

        Sample use:
        normalizer = Normalizer()
        print(normalizer.remove_punctuations("merhaba..."))

        'merhaba'
        """
        return ''.join([t for t in text if (t.isalnum() or t == " ")])

    @staticmethod
    def remove_accent_marks(text: str)-> str:
        """
        Removes accent marks from the given string.

        Input:
        text(str): string of text

        Output:
        text(str): string of text stripped from accent marks

        Sample use:
        normalizer = Normalizer()
        print(normalizer.remove_accent_marks("merhâbâ"))

        'merhaba'
        """
        _non_turkish_accent_marks = {'â':'a', 'ô':'o', 'î':'ı', 'ê':'e', 'û':'u',
                                     'Â':'A', 'Ô':'o', 'Î':'ı', 'Ê':'e', 'Û': 'u'}
        return ''.join(_non_turkish_accent_marks.get(char, char) for char in text)

    @staticmethod
    def deasciify(tokens: List[str]) -> List[str]:
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

def correct_typos(self, tokens: List[str]) -> List[str]:
        """
        Corrects spelling mistakes and typos.

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
            if (self._is_token_valid_turkish(token)) or (self._hunspell.spell(token)):
                corrected_tokens.append(token)
            else:
                hunspell_suggestions = self._hunspell.suggest(token)
                if len(hunspell_suggestions) > 0:
                    corrected_token = hunspell_suggestions[0]
                    corrected_tokens.append(corrected_token)
                else:
                    # there is no suggestion so return the original token
                    corrected_tokens.append(token)
        
        return corrected_tokens
    
    def convert_numbers_to_words(self, tokens: List[str], num_dec_digits: int = 6)-> List[str]:
        """
        Converts numbers to word forms in a given list of tokens.

        Input:
        tokens(List[str]): List of strings of tokens

        Args:
        num_dec_digits: number of precision (decimal points) for floats

        Output:
        converted_tokens(List[str]): List of strings of converted tokens

        Sample use:
        normalizer = Normalizer()
        print(normalizer.convert_numbers_to_words("sabah 3 yumurta yedim ve tartıldığımda 1.15 kilogram aldığımı gördüm".split()))

        ['sabah',
        'üç',
        'yumurta',
        'yedim',
        've',
        'tartıldığımda',
        'bir',
        'virgül',
        'on',
        'beş',
        'kilogram',
        'aldığımı',
        'gördüm']
        """
        converted_tokens = []
        for token in tokens:
            # Try to convert token to number
            try:
                num = float(token)
                converted_tokens+= self._num_to_words(num, num_dec_digits).split()
            # If fails, then return it as string
            except:
                converted_tokens.append(token)
                
        return converted_tokens

    def _is_token_valid_turkish(self, token):
        valid_according_to_stemmer_analyzer = not (self._stemmer_analyzer.candidate_generator.get_analysis_candidates(token)[0][-1] == 'Unknown')
        valid_according_to_lexicon = token in self._words_lexicon
        return valid_according_to_stemmer_analyzer or valid_according_to_lexicon

    def _int_to_words(self, main_num, put_commas = False):
        """
        This function is adapted from:
        https://github.com/Omerktn/Turkish-Lexical-Representation-of-Numbers/blob/master/src.py
        """
        
        # yüz=10^2 ve vigintilyon=10^63, ith element is 10^3 times greater then (i-1)th.
        tp = [" yüz", " bin", "", "", " milyon", " milyar", " trilyon", " katrilyon", " kentilyon",
            " seksilyon", " septilyon", " oktilyon", " nonilyon", " desilyon", " undesilyon",
            " dodesilyon", " tredesilyon", " katordesilyon", " seksdesilyon", " septendesilyon",
            " oktodesilyon", " novemdesilyon", " vigintilyon"]

        # dec[]: every decimal digit,  ten[]: every tenth number
        dec = ["", " bir", " iki", " üç", " dört", " beş", " altı", " yedi", " sekiz", " dokuz"]
        ten = ["", " on", " yirmi", " otuz", " kırk", " elli", " altmış", " yetmiş", " seksen", " doksan"]

        text = ""

        # get length of main_num
        num = main_num
        leng = 0
        while num != 0:
            num = num // 10
            leng += 1

        if main_num == 0:
            text = " sıfır"

        # split main_num to (three digit) pieces and read them by mod 3.
        for i in range(leng, 0, -1):
            digit = int((main_num // (10 ** (i - 1))) % 10)

            if i % 3 == 0:
                if digit == 1:
                    text += tp[0]
                else:
                    text += dec[digit] + tp[0]
            elif i % 3 == 1:
                if i > 3:
                    text += dec[digit] + tp[i - 3]
                else:
                    text += dec[digit]
                if i>3 and put_commas: 
                    text += ","
            elif i % 3 == 2:
                text += ten[digit]
        
        return text[1:]

    def _num_to_words(self, num, num_dec_digits):
        integer_part = int(num)
        decimal_part = round(num % 1, num_dec_digits)

        # if number is int (considering significant decimal digits)
        if decimal_part < 10**-num_dec_digits:
            return self._int_to_words(integer_part)
        # if number is float
        else:
            str_decimal = '{:f}'.format(round(num % 1, num_dec_digits))[2:]
            
            zeros_after_decimal = 0
            for char in str_decimal:
                if char =="0":
                    zeros_after_decimal+=1
                else:
                    break
            str_decimal_stripped_from_zeros = str_decimal.strip("0") # strip gets rid of heading and trailing 0s in string form
            if str_decimal_stripped_from_zeros == "":
                decimal_part = 0
            else:
                decimal_part = int(str_decimal_stripped_from_zeros)

            return self._int_to_words(integer_part) + " virgül " + "sıfır " * zeros_after_decimal + self._int_to_words(decimal_part)
