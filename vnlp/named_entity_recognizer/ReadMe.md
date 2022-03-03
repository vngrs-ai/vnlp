#### Named Entity Recognizer (NER)

- This is an implementation of "CharNER: Character-Level Named Entity Recognition", which can be found here: https://aclanthology.org/C16-1087/
- There are slight modifications to original paper:
	- I did not train for all languages, but only Turkish.
	- I did not use Viterbi Decoder, mine is simple Mode operation among the outputs of each token.

- It achieves over 0.9589 Accuracy and 0.9200 F1_macro_score.
- After development phase, final model in the repository is trained with all of train, dev and test data for 40 epochs with learning rate decay of 0.95 after epoch 10, therefore you cannot evaluate on test sets using the model weights found in this repository. However, model_weights trained on train and dev sets only are available at: https://meliksahturker.s3.us-east-2.amazonaws.com/VNLP/model_weights/ner_trained_on_train_dev.hdf5 . Hence you can place this under resources and evaluate on test sets.
- I gathered, parsed and denoised a very extensive dataset.
Details are as below:

Train, dev and test sets are created by processing and merging multiple datasets in the literature.

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


- They are processed by NLTK.WordPunctTokenized() so that each punctuation becomes a new token.
- Punctuation labels are strictly 'O'.
- B/I prefixes are removed, other entities are converted to 'O' and all entities are standardized to: ['O', 'PER', 'LOC', 'ORG']
- TWNERTC_TC_Coarse Grained NER_DomainDependent_NoiseReduction.DUMP_train in train data is further noise reduced version of original form, consisting of 20 % of original dataset. Despite denoising, including this dataset hurts model performance because it contains lots of errors. Therefore it is not used in neither development, nor production phases of training.