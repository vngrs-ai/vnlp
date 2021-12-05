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