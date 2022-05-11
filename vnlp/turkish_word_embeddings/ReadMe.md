#### Word2Vec and FastText embeddings, SentencePiece Unigram Tokenizer

- Models are trained on a corpus of 32 GBs, made of 288 million sentences and 4.07 billion words consisting of:
	- OSCAR
	- OPUS
	- AI-KU corpus
	- Bilkent Turkish Writings
	- Milliyet news
	- My Dear Watson
	- TED talks
	- tr2en machine translation turkish part
	- Old newspapers corpus
	- TR Wiki dump (17.09.21)
	- Universal Dependencies (tr_boun-ud, tr_imst-ud, tr_pud-ud)
- Word2Vec and FastText models are trained for 10 epochs with a window value of 5.


- `Word2Vec <https://arxiv.org/pdf/1301.3781.pdf>`_ , `FastText <https://arxiv.org/pdf/1607.04606.pdf>`_ and `SentencePiece Unigram Tokenizer <https://arxiv.org/pdf/1808.06226.pdf>`_ and its associated Word2Vec Turkish word embeddings are trained on a corpora of 32 GBs.

- Regular (nltk.TreeBankWordTokenizer) tokenized tokens are converted to lowercase. Punctuations and numbers are NOT removed. Every numeric character is converted to 0 so should you when using these embeddings. E.g. "10-05-2010" becomes "00-00-0000".
- Regular (nltk.TreeBankWordTokenizer) tokenized Word2Vec and FastText embeddings come in 3 sizes:

	- **Large**: vocabulary size: 128_000, vector size: 256
	- **Medium**: vocabulary size: 64_000, vector size: 128
	- **Small**: vocabulary size: 32_000, vector size: 64

- They can be downloaded from the links below:

- **Large**:
    - `Word2Vec <https://vnlp-word-embeddings.s3.eu-west-1.amazonaws.com/Word2Vec_large.zip>`_
    - `FastText <https://vnlp-word-embeddings.s3.eu-west-1.amazonaws.com/FastText_large.zip>`_

- **Medium**:
    - `Word2Vec <https://vnlp-word-embeddings.s3.eu-west-1.amazonaws.com/Word2Vec_medium.zip>`_
    - `FastText <https://vnlp-word-embeddings.s3.eu-west-1.amazonaws.com/FastText_medium.zip>`_
	
- **Small**:
    - `Word2Vec: <https://vnlp-word-embeddings.s3.eu-west-1.amazonaws.com/Word2Vec_small.zip>`_
    - `FastText <https://vnlp-word-embeddings.s3.eu-west-1.amazonaws.com/FastText_small.zip>`_

- Sentence Piece Unigram Tokenizer and its associated Word2Vec embeddings come in 2 sizes for each config.

    - **Medium Tokenizer**: vocabulary size: 32_000
        - Large embeddings: vector size: 256
        - Medium embeddings: vector size: 128

    - **Small Tokenizer**: vocabulary size: 16_000
        - Large embeddings: vector size: 256
        - Medium embeddings: vector size: 128

- They can be downloaded from the links below:

- `Medium Tokenizer <https://vnlp-word-embeddings.s3.eu-west-1.amazonaws.com/SentencePiece_32k_Tokenizer.zip>`_ :
    - `Large Word2Vec embeddings <https://vnlp-word-embeddings.s3.eu-west-1.amazonaws.com/SentencePiece_32k_Word2Vec_large.zip>`_
    - `Medium Word2Vec embeddings <https://vnlp-word-embeddings.s3.eu-west-1.amazonaws.com/SentencePiece_32k_Word2Vec_medium.zip>`_

- `Small Tokenizer <https://vnlp-word-embeddings.s3.eu-west-1.amazonaws.com/SentencePiece_16k_Tokenizer.zip>`_ :
    - `Large Word2Vec embeddings <https://vnlp-word-embeddings.s3.eu-west-1.amazonaws.com/SentencePiece_16k_Word2Vec_large.zip>`_
    - `Medium Word2Vec embeddings <https://vnlp-word-embeddings.s3.eu-west-1.amazonaws.com/SentencePiece_16k_Word2Vec_medium.zip>`_

- Word2Vec and FastText embeddings are trained with `gensim <https://github.com/RaRe-Technologies/gensim>`_ algorithm.
- Sentence Piece Unigram tokenizer is trained with `SentencePiece <https://github.com/google/sentencepiece>`_  algorithm.

- Usage:
    >>> # Word2Vec
    >>> from gensim.models import Word2Vec
    >>> 
    >>> model = Word2Vec.load('Word2Vec_large.model')
    >>> model.wv.most_similar('gandalf', topn = 10)
    [('saruman', 0.7291593551635742),
    ('thorin', 0.6473978161811829),
    ('aragorn', 0.6401687264442444),
    ('isengard', 0.6123237013816833),
    ('orklar', 0.59786057472229),
    ('gollum', 0.5905635952949524),
    ('baggins', 0.5837421417236328),
    ('frodo', 0.5819021463394165),
    ('belgarath', 0.5811135172843933),
    ('sauron', 0.5763844847679138)]
    
    >>> # FastText
    >>> from gensim.models import FastText
    >>> 
    >>> model = FastText.load('FastText_large.model')
    >>> model.wv.most_similar('yamaçlardan', topn = 10)
    [('kayalardan', 0.8601457476615906),
    ('kayalıklardan', 0.8567330837249756),
    ('tepelerden', 0.8423191905021667),
    ('ormanlardan', 0.8362939357757568),
    ('dağlardan', 0.8140010833740234),
    ('amaçlardan', 0.810560405254364),
    ('bloklardan', 0.803180992603302),
    ('otlardan', 0.8026642203330994),
    ('kısımlardan', 0.7993910312652588),
    ('ağaçlardan', 0.7961613535881042)]

    >>> # SentencePiece Unigram Tokenizer
    >>> import sentencepiece as spm
    >>> sp = spm.SentencePieceProcessor('SentencePiece_16k_Tokenizer.model')
    >>> tokenizer.encode_as_pieces('bilemezlerken')
    ['▁bile', 'mez', 'lerken']
    >>> tokenizer.encode_as_ids('bilemezlerken')
    [180, 1200, 8167]