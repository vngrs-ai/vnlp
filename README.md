<img src="https://github.com/vngrs-ai/vnlp/blob/main/img/logo.png?raw=true" width="256">

## VNLP: Turkish NLP Tools
State-of-the-art, lightweight NLP tools for Turkish language.

Developed by VNGRS.

https://vngrs.com/


[![PyPI version](https://badge.fury.io/py/vngrs-nlp.svg)](https://badge.fury.io/py/vngrs-nlp)
[![PyPi downloads](https://static.pepy.tech/personalized-badge/vngrs-nlp?period=total&units=international_system&left_color=grey&right_color=orange&left_text=pip%20downloads)](https://pypi.org/project/vngrs-nlp/)
[![Docs](<https://readthedocs.org/projects/vnlp/badge/?version=latest&style=plastic>)](https://vnlp.readthedocs.io/)
[![License](<https://img.shields.io/badge/license-AGPL%203.0-green.svg>)](https://github.com/vngrs-ai/vnlp/blob/main/LICENSE)
[![Python check](https://github.com/vngrs-ai/vnlp/actions/workflows/test.yml/badge.svg)](https://github.com/vngrs-ai/vnlp/actions/workflows/test.yml)

### Functionality:
- Sentence Splitter
- Normalizer
	- Spelling/Typo correction
	- Convert numbers to word form
	- Deasciification
- Stopword Remover:
	- Static
	- Dynamic
- Stemmer: Morphological Analyzer & Disambiguator
- Named Entity Recognizer (NER) 
- Dependency Parser
- Part of Speech (PoS) Tagger
- Sentiment Analyzer
- Turkish Word Embeddings
	- FastText
	- Word2Vec
	- SentencePiece Unigram Tokenizer
- News Summarization
- News Paraphrasing

- Summarization and Paraphrasing models are available in the demo. Contact us at vnlp@vngrs.com for API.

### Demo:
- Try the [Demo](https://demo.vnlp.io).

### Installation
```
pip install vngrs-nlp
```

### Documentation:
- See the [Documentation](https://vnlp.readthedocs.io) for the details about usage, classes, functions, datasets and evaluation metrics.

### Metrics:
<img src="https://github.com/vngrs-ai/vnlp/blob/main/img/metrics.png?raw=true" width="600">

<img src="https://github.com/vngrs-ai/vnlp/blob/main/img/sum_metrics.png?raw=true" width="124">

### Usage Example:
**Dependency Parser**
```
from vnlp import DependencyParser
dep_parser = DependencyParser()

dep_parser.predict("Oğuz'un kırmızı bir Astra'sı vardı.")
[("Oğuz'un", 'PROPN'),
 ('kırmızı', 'ADJ'),
 ('bir', 'DET'),
 ("Astra'sı", 'PROPN'),
 ('vardı', 'VERB'),
 ('.', 'PUNCT')]

# Spacy's submodule Displacy can be used to visualize DependencyParser result.
import spacy
from vnlp import DependencyParser
dependency_parser = DependencyParser()
result = dependency_parser.predict("Oğuz'un kırmızı bir Astra'sı vardı.", displacy_format = True)
spacy.displacy.render(result, style="dep", manual = True)

<img src="https://raw.githubusercontent.com/vngrs-ai/vnlp/main/img/dp_vis_sample.png" width="512">

## Citation

```bibtex
@article{turker2024vnlp,
  title={VNLP: Turkish NLP Package},
  author={Turker, Meliksah and Ari, Erdi and Han, Aydin},
  journal={arXiv preprint arXiv:2403.01309},
  year={2024}
}
```
