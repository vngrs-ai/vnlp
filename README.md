### Turkish-NLP-preprocessing-module
NLP Preprocessing module for Turkish language

Consists of:
- Sentence Splitter
- Normalizer:
	- Punctuation Remover
	- Convert numbers to word form
	- Remove accent marks
	- Spelling/typo correction using:
		- Pre-defined typos lexicon
		- Levenshtein distance
		- Morphological Analyzer
- Stopword Remover:
	- Static
	- Dynamic
		- Frequent words
		- Rare words
- Stemmer: Morphological Analyzer & Disambiguator
- NER: Named Entity Recognizer
- Turkish Embeddings
	- FastText
	- Word2Vec
	
# ---------------------------------------------------

#### Usage:

```
from pp.stemmer_morph_analyzer import StemmerAnalyzer
ma = StemmerAnalyzer()

ma.predict("üniversite sınavlarına canla başla çalışıyorlardı")
['üniversite+Noun+A3sg+Pnon+Nom',
 'sınav+Noun+A3pl+P3sg+Dat',
 'can+Noun+A3sg+Pnon+Ins',
 'baş+Noun+A3sg+Pnon+Ins',
 'çalış+Verb+Pos+Prog1+A3pl+Past']
 ```