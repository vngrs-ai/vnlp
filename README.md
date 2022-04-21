<img src="https://github.com/vngrs-ai/VNLP/blob/main/vnlp.png" width="256">

## VNLP: Turkish NLP Tools
State of the art, lightweight NLP tools for Turkish language.
Developed by VNGRS.
https://vngrs.com/

#### Functionality:
- Sentence Splitter
- Normalizer
	- Spelling/Typo correction
	- Converts numbers to word form
	- Deasciification
- Stopword Remover:
	- Static
	- Dynamic
- Stemmer: Morphological Analyzer & Disambiguator
- Named Entity Recognizer (NER) 
- Dependency Parser
- Part of Speech (POS) Tagger
- Sentiment Analyzer
- Turkish Word Embeddings
	- FastText
	- Word2Vec
- Text Summarization: In development progress...

### Compatability:

Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10
:------------ | :-------------| :-------------| :-------------| :-------------
:heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :negative_squared_cross_mark:

- cyhunspell does not currently support 3.10

### Documentation:
- Detailed documentation about usage, classes, functions, datasets and evaluation metrics are available at [Documentation](https://vnlp.readthedocs.io).

### Installation
#### pip
```
pip install vnlp
```

#### Build from source
open shell.
write
```
git clone https://github.com/vngrs-ai/VNLP.git
cd vnlp
python setup.py install
```

For Linux/MacOS, you might need to use
```
python3 setup.py install
```
instead.

To install extra dependencies to read word embeddings and visualize dependency parsing tree
```
pip install -e '.[extras]'
```
or you can simply install gensim and spacy manually.

#### Example:
**Dependency Parser**
```
from vnlp import DependencyParser
dep_parser = DependencyParser()

dep_parser.predict("Onun için yol arkadaşlarımızı titizlikle seçer, kendilerini iyice sınarız.")
[(1, 'Onun', 5, 'obl'),
(2, 'için', 1, 'case'),
(3, 'yol', 1, 'nmod'),
(4, 'arkadaşlarımızı', 5, 'obj'),
(5, 'titizlikle', 6, 'obl'),
(6, 'seçer', 7, 'acl'),
(7, ',', 10, 'punct'),
(8, 'kendilerini', 10, 'obj'),
(9, 'iyice', 8, 'advmod'),
(10, 'sınarız', 0, 'root'),
(11, '.', 10, 'punct')]

# Spacy's submodule Displacy can be used to visualize DependencyParser result.
import spacy
from vnlp import DependencyParser
dependency_parser = DependencyParser()
result = dependency_parser.predict("Bu örnek bir cümledir.", displacy_format = True)
spacy.displacy.render(result, style="dep", manual = True)

```
