#### Stemmer: Morphological Analyzer & Disambiguator

- This is an implementation of "The Role of Context in Neural Morphological Disambiguation", which can be found here: https://aclanthology.org/C16-1018.pdf
- There are slight modifications to original paper:
	- My network uses GRU instead of LSTM, which decreases the number of parameters by 25% with no actual performance penalty.
	- My network has an extra Dense layer before output(p) layer.
	- My network optionally uses Concatenation instead of Addition while merging representations of stems & tags, and left & right surface form contexts.
	- My network optionally uses Deep RNNs.

- It achieves 0.943 accuracy on ambigious tokens and 0.964 accuracy on all tokens of trmorph2006 test set, compared to 0.910 and 0.964 in the original paper.
- After development phase, final model in the repository is trained on all of train, test and handtagged sets of trmorph2006, trmorph2016 and trmorph2018 sets for 10 epochs. Therefore you cannot evaluate on these handtagged at this point.
- As analyzer, it uses Yildiz's analyzer, which can be found here: https://github.com/erayyildiz/LookupAnalyzerDisambiguator