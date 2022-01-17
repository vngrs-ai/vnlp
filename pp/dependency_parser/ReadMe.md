#### Dependency Parser

- This transitioned-based dependency parser is inspired by, Tree-stack LSTM in Transition Based Dependency Parsing,
which can be found here: https://aclanthology.org/C16-1087/
- I indicate "inspire" because I simply used the approach of using Morphological Tags and Pre-trained word embeddings for the model,
rather than implementing the network proposed there.
- The paper uses POS tags, but I did not use that either, since I did not have a POS tagger implemented during the development of this module.
	
- It achieves 0.68 LAS and 0.80 on Conll on test sets of Universal Dependencies 2.9.
- Starting with 0.001 learning rate, lr decay of 0.95 is used after the first epoch.
- After development phase, final model in the repository is trained with all of train, dev and test data after development phase so you should not test it on the this test set. However you can train from scratch for evaluation using train data only which is available on https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4611.
- UD 2.9 consists of:
	- UD_Turkish_German-SAGT
	- UD_Turkish-Atis
	- UD_Turkish-BOUN
	- UD_Turkish-FrameNet
	- UD_Turkish-GB
	- UD_Turkish-IMST
	- UD_Turkish-Kenet
	- UD_Turkish-Penn
	- UD_Turkish-PUD
	- UD_Turkish-Tourism

- Input data is processed by NLTK.WordPunctTokenizer() so that each punctuation becomes a new token, with arc values shifted accordingly.