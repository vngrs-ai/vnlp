### Named Entity Recognizer (NER)
- Named Entity Recognition implementations of VNLP.
- Details of each model are provided below.

- In order to evaluate, initialize the class with "evaluate = True" argument. This will load the model weights that are not trained on test sets.

#### SPUContext Named Entity Recognizer
- This is a context aware Named Entity Recognizer that uses SentencePiece Unigram tokenizer and pre-trained Word2Vec embeddings.
- It achieves 0.9928 Accuracy and 0.9833 F1 macro score.
- Test set consists of below datasets with evaluation metrics on each one's test set:
	- wikiann-tdd.ai-xtreme-tr-test: Accuracy: 0.9880 - F1_macro_score: 0.9814
	- gungor.ner.test.14.only_consistent-test: Accuracy: 0.9970 - F1_macro_score: 0.9859
	- teghub-TurkishNER-BERT-test: Accuracy: 0.9974 - F1_macro_score: 0.9891
- F1 scores for each entity (treating entity of interest as positive, and all other entities as negative):
	- ORG: F1 score: 0.9766
	- PER: F1 score: 0.9852
	- LOC: F1 score: 0.9742

- After development phase, final model in the repository is trained with all of train, dev and test data for 50 epochs.
- Starting with 0.001 learning rate, lr decay of 0.95 is used after the 3rd epoch.

- Input data is processed by NLTK.tokenize.TreebankWordTokenizer.

#### CharNER
- This is an implementation of "CharNER: Character-Level Named Entity Recognition", which can be found here: https://aclanthology.org/C16-1087/
- There are slight modifications to original paper:
	- I did not train for all languages, but only Turkish.
	- I did not use Viterbi Decoder, mine is simple Mode operation among the outputs of each token.

- It achieves 0.9589 Accuracy and 0.9200 F1_macro_score.
- Tokens are processed by NLTK.tokenize.WordPunctTokenizer so that each punctuation becomes a new token.

#### Dataset
- I gathered, parsed and denoised a very extensive dataset.
- Train, dev and test sets are created by processing and merging multiple datasets in the literature.

- Gungor Joint Ner and Md Tagger dataset
https://github.com/onurgu/joint-ner-and-md-tagger/tree/master/dataset

- Turkish News NER dataset
https://data.tdd.ai/#/0a027105-498c-46f7-9867-2ceeac5e64b7

- XTREME Turkish NER dataset
https://github.com/google-research/xtreme
https://data.tdd.ai/#/204e1373-7a9e-4f76-aa75-7708593cf2dd

- TegHUB Turkish NER data 3 labels dataset
https://github.com/teghub/TurkishNER-BERT/tree/master/TurkishNERdata3Labels

- NER Turkish tweets dataset
https://github.com/dkucuk/NER-Turkish-Tweets/blob/master/2014_JRC_Twitter_TR_NER-dataset.zip

- Wikiann Turkish NER dataset
https://github.com/savasy/Turkish-Bert-Based-NERModel

- B/I prefixes are removed, other entities are converted to 'O' and all entities are standardized to: ['O', 'PER', 'LOC', 'ORG']
- Punctuation labels are strictly 'O'.
- TWNERTC_TC_Coarse Grained NER_DomainDependent_NoiseReduction.DUMP_train in train data is further noise reduced version of original form, consisting of 20 % of original dataset. Despite denoising, including this dataset hurts model performance because it contains lots of errors. Therefore it is not used in neither development, nor production phases of training.