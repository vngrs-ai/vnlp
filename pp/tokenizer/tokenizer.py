from pathlib import Path
import re
from ._tokenization_rules import *
from .. import _utils as utils

PATH = "../_resources/"
PATH = str(Path(__file__).parent / PATH)

class RuleBasedTokenizer:

    def __init__(self):
        self._mwe_lexicon = utils.load_words(PATH + '/mwe_lexicon.txt')
        self._abbrevations = utils.load_words(PATH + '/abbrevations.txt')

    def tokenize(self, input_sentence, rules=rules, split_characters=split, split_token='<*>'):
        """
        Given a string of tokens, returns list of string tokens.
        """

        sentence = input_sentence

        # Check regular expressions for matches and add split:
        for rule in rules:
            sentence = re.sub(rule, " \g<0> ", sentence)   #The backreference \g<0> substitutes in the entire substring matched by the RE.

        # Split from all splitted characters
        working_sentence = re.sub(split_characters, split_token, sentence)
        list_of_token_strings = [x.strip() for x in working_sentence.split(split_token) if x.strip() !=""]

        original_list_of_token_strings = list(list_of_token_strings)

        # Normalization:
        index = 0
        inserted_dots = 0
        for token in original_list_of_token_strings:
            index += 1
            if token[-1] == '.':
                abbrevation = False
                # Check if abbrevation:
                if token in self._abbrevations:
                    abbrevation = True
                if not abbrevation:
                    new_token = token[:-1]
                    list_of_token_strings.insert(index + inserted_dots, '.')
                    list_of_token_strings[index + inserted_dots-1] = new_token
                    inserted_dots += 1

        # Multi Word Expressions
        # Known bug:
        # If MWE appears at the end of the sentence,
        # Bug appears.

        original_length = len(original_list_of_token_strings)
        original_list_of_token_strings = list(list_of_token_strings)
        index = 0

        while index < original_length:

            token = original_list_of_token_strings[index]

            for expression in self._mwe_lexicon:
                expression_length = expression.count(' ') + 1
                check_index = index
                is_multiword = True

                for i in range(expression_length):
                    if index+i >= original_length:
                        continue
                    else:
                        if original_list_of_token_strings[index+i] not in expression:
                            is_multiword = False

                if is_multiword:
                    # Pass if already multiword:
                    if token.count(' ') == 0:
                        list_of_token_strings.insert(index, expression)

                        for deleter in range(expression_length):
                            if index+1 < original_length:
                                list_of_token_strings.pop(index + 1)

                        index += expression_length

            index += 1


        return list_of_token_strings
