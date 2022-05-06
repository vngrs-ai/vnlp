![](https://raw.githubusercontent.com/vngrs-ai/VNLP/main/img/logo.png)

## VNLP: Turkish NLP Tools
State of the art, lightweight NLP tools for Turkish language.
Developed by VNGRS.
https://vngrs.com/

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
- Part of Speech (POS) Tagger
- Sentiment Analyzer
- Turkish Word Embeddings
	- FastText
	- Word2Vec
- Text Summarization: In development progress...

### Installation
```
pip install vngrs-nlp
```
```
conda install vngrs-nlp
```
### Documentation:
- Detailed documentation about usage, classes, functions, datasets and evaluation metrics are available at [Documentation](https://vnlp.readthedocs.io).

### Usage Example:
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
result = dependency_parser.predict("Oğuz'un kırmızı bir Astra'sı vardı.", displacy_format = True)
spacy.displacy.render(result, style="dep", manual = True)
```
![](https://raw.githubusercontent.com/vngrs-ai/VNLP/main/img/dp_vis_sample.png | width=512)
