#### Part of Speech (POS) Tagger

- This POS Tagger is inspired by "Tree-stack LSTM in Transition Based Dependency Parsing",
which can be found here: https://aclanthology.org/C16-1087/
- I indicate "inspire" because I simply used the approach of using Morphological Tags and Pre-trained word embeddings as input for the model.
- The model uses pre-trained Word2Vec_medium embeddings which is also a part of this project. Embedding weights make %77 of model weights, hence the model size as well.
- The model also uses pre-trained Morphological Tag embeddings, extracted from StemmerAnalyzer's neural network model.

- It achieves 0.90 Accuracy and 0.75 F1_macro_score on test sets of Universal Dependencies 2.9.
- Starting with 0.001 learning rate, lr decay of 0.95 is used after the 5th epoch, with class weights given below.
- After development phase, final model in the repository is trained with all of train, dev and test data for 20 epochs. Therefore you should not test it on the this test set. However you can train from scratch for evaluation using train data only which is available on https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4611
- UD 2.9 consists of below datasets with evaluation metrics on each one's test set:
	- UD_Turkish-Atis: Accuracy: 0.9774 - F1_macro_score: 0.9832
	- UD_Turkish-BOUN: Accuracy: 0.8683 - F1_macro_score: 0.7701
	- UD_Turkish-FrameNet: Accuracy: 0.9454 - F1_macro_score: 0.9218
	- UD_Turkish-GB: Accuracy: 0.8580 - F1_macro_score: 0.6606
	- UD_Turkish-IMST: Accuracy: 0.9120 - F1_macro_score: 0.7841
	- UD_Turkish-Kenet: Accuracy: 0.9172 - F1_macro_score: 0.8961
	- UD_Turkish-Penn: Accuracy: 0.9287 - F1_macro_score: 0.8979
	- UD_Turkish-PUD: Accuracy: 0.8365 - F1_macro_score: 0.6386
	- UD_Turkish-Tourism: Accuracy: 0.9843 - F1_macro_score: 0.9304
	- UD_Turkish_German-SAGT: This is skipped since it contains lots of non-Turkish tokens.

- Class weights are applied in training to offset the imbalance in class distribution. Class distribution and weights are as follows:
	Class	 Samples	Weights
	NOUN     244057		1
	PUNCT    115946		2
	VERB      98611		2
	ADJ       78511		2
	PROPN     46531		3
	ADV       41706		4
	DET       28449		4
	CCONJ     22150		4
	PRON      17323		5
	NUM       17104		5
	ADP       16363		5
	AUX        7198		5
	X           875		2
	SCONJ       729		2
	INTJ        645		2
	SYM           6		2
	PART          1		2
- Input data is processed by NLTK.TreebankWordTokenize().