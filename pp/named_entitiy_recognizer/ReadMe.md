#### CharNER

- This is an implementation of CharNER: Character-Level Named Entity Recognition, which can be found here: https://aclanthology.org/C16-1087/
- There are slight modifications to original paper:
	- I did not train for all languages, but only Turkish.
	- I did not use Viterbi Decoder, mine is simple Mode operation among each character output.
	
- It achieves over 0.96 F1 micro score.
- I gathered, parsed and denoised a very extensive dataset, details of which are as below:

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
- TWNERTC_TC_Coarse Grained NER_DomainDependent_NoiseReduction.DUMP_train in train data is further noise reduced version of original form, consisting of 20 % of original dataset.

You can track which row comes from which dataset from below indices:
File: gungor.ner.train.14.only_consistent-train.txt start index: 0 end_index 25512
File: tdd.ai-ner-news-dataset-train.txt start index: 25513 end_index 44800
File: tdd.ai-xtreme-tr-train.txt start index: 44801 end_index 64799
File: teghub-TurkishNER-BERT-train.txt start index: 64800 end_index 94144
File: twitter_train.txt start index: 94145 end_index 94231
File: wikiann-train.txt start index: 94232 end_index 114230
File: TWNERTC_TC_Coarse Grained NER_DomainDependent_NoiseReduction.DUMP_train.txt start index: 114231 end_index 224093

File: gungor.ner.dev.14.only_consistent-dev.txt start index: 0 end_index 2952
File: tdd.ai-ner-news-dataset-dev.txt start index: 2953 end_index 7084
File: tdd.ai-xtreme-tr-dev.txt start index: 7085 end_index 17083
File: teghub-TurkishNER-BERT-dev.txt start index: 17084 end_index 19908
File: wikiann-dev.txt start index: 19909 end_index 29907

File: gungor.ner.test.14.only_consistent-test.txt start index: 0 end_index 2913
File: tdd.ai-xtreme-tr-test.txt start index: 2914 end_index 12912
File: teghub-TurkishNER-BERT-test.txt start index: 12913 end_index 16240
File: wikiann-test.txt start index: 16241 end_index 26239