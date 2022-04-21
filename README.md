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
:heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :white_check_mark:

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
