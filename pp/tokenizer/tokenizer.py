import re
from typing import List

def WordPunctTokenize(text: str) -> List[str]:
    """
    This is a simplified version of NLTK's WordPunctTokenizer that can be found in
    https://github.com/nltk/nltk/blob/90fa546ea600194f2799ee51eaf1b729c128711e/nltk/tokenize/regexp.py
    """

    pattern = r"\w+|[^\w\s]+"
    pattern = getattr(pattern, "pattern", pattern)
    flags = re.UNICODE | re.MULTILINE | re.DOTALL
    regexp = re.compile(pattern, flags)

    return regexp.findall(text)

def TreebankWordTokenize(text: str) -> List[str]:
    """
    This is a simplified version of NLTK's TreebankWordTokenizer that can be found in
    https://github.com/nltk/nltk/blob/90fa546ea600194f2799ee51eaf1b729c128711e/nltk/tokenize/treebank.py
    """
    # starting quotes
    STARTING_QUOTES = [
        (re.compile(r"^\""), r"``"),
        (re.compile(r"(``)"), r" \1 "),
        (re.compile(r"([ \(\[{<])(\"|\'{2})"), r"\1 `` "),
    ]

    # punctuation
    PUNCTUATION = [
        (re.compile(r"([:,])([^\d])"), r" \1 \2"),
        (re.compile(r"([:,])$"), r" \1 "),
        (re.compile(r"\.\.\."), r" ... "),
        (re.compile(r"[;@#$%&]"), r" \g<0> "),
        (
            re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'),
            r"\1 \2\3 ",
        ),  # Handles the final period.
        (re.compile(r"[?!]"), r" \g<0> "),
        (re.compile(r"([^'])' "), r"\1 ' "),
    ]

    # Pads parentheses
    PARENS_BRACKETS = (re.compile(r"[\]\[\(\)\{\}\<\>]"), r" \g<0> ")

    DOUBLE_DASHES = (re.compile(r"--"), r" -- ")

    # ending quotes
    ENDING_QUOTES = [
        (re.compile(r"''"), " '' "),
        (re.compile(r'"'), " '' "),
        (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
        (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
    ]

    for regexp, substitution in STARTING_QUOTES:
        text = regexp.sub(substitution, text)

    for regexp, substitution in PUNCTUATION:
        text = regexp.sub(substitution, text)

    # Handles parentheses.
    regexp, substitution = PARENS_BRACKETS
    text = regexp.sub(substitution, text)

    # Handles double dash.
    regexp, substitution = DOUBLE_DASHES
    text = regexp.sub(substitution, text)

    # add extra space to make things easier
    text = " " + text + " "

    for regexp, substitution in ENDING_QUOTES:
        text = regexp.sub(substitution, text)

    return text.replace("``", '"').replace("''", '"').split()