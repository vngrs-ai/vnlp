### Part of Speech (PoS) Tagger
- Part of Speech tagging implementations of VNLP.
- Details of each model are provided below.

- Input data is processed by NLTK.tokenize.TreebankWordTokenizer.
- Training data can be accessed at: https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4611 .
- In order to evaluate, initialize the class with "evaluate = True" argument. This will load the model weights that are not trained on test sets.

#### SPUContext Part of Speech Tagger
- This is a context aware Part of Speech tagger that uses SentencePiece Unigram tokenizer and pre-trained Word2Vec embeddings.

- It achieves 0.9010 Accuracy and 0.7623 F1_macro_score on test sets of Universal Dependencies 2.9.
- UD 2.9 consists of below datasets with evaluation metrics on each one's test set:
	- UD_Turkish-Atis: Accuracy: 0.9874 - F1_macro_score: 0.9880
	- UD_Turkish-BOUN: Accuracy: 0.8708 - F1_macro_score: 0.7884
	- UD_Turkish-FrameNet: Accuracy: 0.9509 - F1_macro_score: 0.9039
	- UD_Turkish-GB: Accuracy: 0.8559 - F1_macro_score: 0.6620
	- UD_Turkish-IMST: Accuracy: 0.9069 - F1_macro_score: 0.7845
	- UD_Turkish-Kenet: Accuracy: 0.9194 - F1_macro_score: 0.8766
	- UD_Turkish-Penn: Accuracy: 0.9452 - F1_macro_score: 0.9329
	- UD_Turkish-PUD: Accuracy: 0.8387 - F1_macro_score: 0.6559
	- UD_Turkish-Tourism: Accuracy: 0.9845 - F1_macro_score: 0.9325
	- UD_Turkish_German-SAGT: This is skipped since it contains lots of non-Turkish tokens.

- After development phase, final model in the repository is trained with all of train, dev and test data for 50 epochs.
- Starting with 0.001 learning rate, lr decay of 0.95 is used after the 3rd epoch.

#### TreeStack Part of Speech Tagger
- This Part of Speech tagger is inspired by "Tree-stack LSTM in Transition Based Dependency Parsing",
which can be found here: https://aclanthology.org/K18-2012/ .
- "Inspire" is emphasized because this implementation uses the approach of using Morphological Tags, Pre-trained word embeddings and POS tags as input for the model, rather than implementing the exact network proposed in the paper.
- The model uses pre-trained Word2Vec_medium embeddings which is also a part of this project.
- The model also uses pre-trained Morphological Tag embeddings, extracted from StemmerAnalyzer's neural network model.

- It achieves 0.89 Accuracy and 0.71 F1_macro_score on test sets of Universal Dependencies 2.9.
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

- After development phase, final model in the repository is trained with all of train, dev and test data for 20 epochs.
- Starting with 0.001 learning rate, lr decay of 0.95 is used after the 5th epoch.