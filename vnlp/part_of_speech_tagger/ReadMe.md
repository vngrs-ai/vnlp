#### Part of Speech (POS) Tagger

- This POS Tagger is inspired by "Tree-stack LSTM in Transition Based Dependency Parsing",
which can be found here: https://aclanthology.org/C16-1087/
- I indicate "inspire" because I simply used the approach of using Morphological Tags and Pre-trained word embeddings as input for the model.
- The model uses pre-trained Word2Vec_medium embeddings which is also a part of this project. Embedding weights make %77 of model weights, hence the model size as well.
- The model also uses pre-trained Morphological Tag embeddings, extracted from StemmerAnalyzer's neural network model.

- It achieves 0.89 Accuracy and 0.71 F1_macro_score on test sets of Universal Dependencies 2.9.
- Data is found at: https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4611
- Starting with 0.001 learning rate, lr decay of 0.95 is used after the 5th epoch, with class weights given below.
- After development phase, final model in the repository is trained with all of train, dev and test data for 20 epochs.
- UD 2.9 consists of below datasets with evaluation metrics on each one's test set:
	- UD_Turkish-Atis: Accuracy: 0.9695 - F1_macro_score: 0.8858
	- UD_Turkish-BOUN: Accuracy: 0.8543 - F1_macro_score: 0.7607
	- UD_Turkish-FrameNet: Accuracy: 0.9447 - F1_macro_score: 0.8146
	- UD_Turkish-GB: Accuracy: 0.8558 - F1_macro_score: 0.6274
	- UD_Turkish-IMST: Accuracy: 0.9041 - F1_macro_score: 0.7987
	- UD_Turkish-Kenet: Accuracy: 0.9039 - F1_macro_score: 0.8287
	- UD_Turkish-Penn: Accuracy: 0.9320 - F1_macro_score: 0.7967
	- UD_Turkish-PUD: Accuracy: 0.8303 - F1_macro_score: 0.6272
	- UD_Turkish-Tourism: Accuracy: 0.9799 - F1_macro_score: 0.9025
	- UD_Turkish_German-SAGT: This is skipped since it contains lots of non-Turkish tokens.

- Class weights are applied in training to offset the imbalance in class distribution. Class distribution and weights are as follows:
	Class	 Samples	Weights
	NOUN     244057		1
	PUNCT    115946		2
	VERB      98611		2
	ADJ       78511		2
	PROPN     46531		3
	ADV       41706		3
	DET       28449		3
	CCONJ     22150		3
	PRON      17323		3
	NUM       17104		3
	ADP       16363		3
	AUX        7198		3
	X           875		2
	SCONJ       729		2
	INTJ        645		2
	SYM           6		2
	PART          1		2
- Input data is processed by NLTK.TreebankWordTokenize().