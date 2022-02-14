#### Dependency Parser

- This dependency parser is inspired by "Tree-stack LSTM in Transition Based Dependency Parsing",
which can be found here: https://aclanthology.org/C16-1087/
- I indicate "inspire" because I simply used the approach of using Morphological Tags, Pre-trained word embeddings and POS tags as input for the model, rather than implementing the network proposed there.
- The model uses pre-trained Word2Vec_medium embeddings which is also a part of this project. Embedding weights make %56 of model weights, hence the model size as well.
- The model also uses pre-trained Morphological Tag embeddings, extracted from StemmerAnalyzer's neural network model.

- It achieves 0.6841 LAS (Labeled Attachment Score) and 0.8003 UAS (Unlabeled Attachment Score) on all of test sets of Universal Dependencies 2.9.
- Starting with 0.001 learning rate, lr decay of 0.95 is used after the 3rd epoch.
- After development phase, final model in the repository is trained with all of train, dev and test data for 20 epochs. Therefore you should not test it on the this test set. However you can train from scratch for evaluation using train data only which is available on https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4611
- UD 2.9 consists of below datasets with evaluation metrics on each one's test set:
	- UD_Turkish-Atis: LAS: 0.8385 - UAS: 0.8763
	- UD_Turkish-BOUN: LAS: 0.6284 - UAS: 0.7252
	- UD_Turkish-FrameNet: LAS: 0.7962 - UAS: 0.9052
	- UD_Turkish-GB: LAS: 0.7467 - UAS: 0.8874
	- UD_Turkish-IMST: LAS: 0.5930 - UAS: 0.7122
	- UD_Turkish-Kenet: LAS: 0.6460 - UAS: 0.7880
	- UD_Turkish-Penn: LAS: 0.6727 - UAS: 0.8027
	- UD_Turkish-PUD: LAS: 0.5698 - UAS: 0.6950
	- UD_Turkish-Tourism: LAS: 0.9067 - UAS: 0.9659
	- UD_Turkish_German-SAGT: This is skipped since it contains lots of non-Turkish tokens.

- Input data is processed by NLTK.TreebankWordTokenize().