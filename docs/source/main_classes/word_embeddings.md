Word Embeddings
===================================

- `Word2Vec <https://arxiv.org/pdf/1301.3781.pdf>`_ and `FastText <https://arxiv.org/pdf/1607.04606.pdf>`_ Turkish word embeddings are trained on a corpora of 32 GBs.

- They come in 3 sizes:
	- **Large**: vocabulary size: 128_000, vector size: 256
	- **Medium**: vocabulary size: 64_000, vector size: 128
	- **Small**: vocabulary size: 32_000, vector size: 64

- Can be downloaded from below links:

- **Large**:
    - `Word2Vec <https://meliksahturker.s3.us-east-2.amazonaws.com/VNLP/turkish_word_embeddings/Word2Vec_large.zip>`_
    - `FastText <https://meliksahturker.s3.us-east-2.amazonaws.com/VNLP/turkish_word_embeddings/FastText_large.zip>`_

- **Medium**:
    - `Word2Vec <https://meliksahturker.s3.us-east-2.amazonaws.com/VNLP/turkish_word_embeddings/Word2Vec_medium.zip>`_
    - `FastText <https://meliksahturker.s3.us-east-2.amazonaws.com/VNLP/turkish_word_embeddings/FastText_medium.zip>`_
	
- **Small**:
    - `Word2Vec: <https://meliksahturker.s3.us-east-2.amazonaws.com/VNLP/turkish_word_embeddings/Word2Vec_small.zip>`_
    - `FastText <https://meliksahturker.s3.us-east-2.amazonaws.com/VNLP/turkish_word_embeddings/FastText_small.zip>`_


- They are trained with `gensim <https://github.com/RaRe-Technologies/gensim>`_, so it is needed to open them.

>>> from gensim.models import Word2Vec, FastText
>>> 
>>> # Word2Vec
>>> model = Word2Vec.load('vnlp/turkish_embeddings/Word2Vec_large.model')
>>> model.wv.most_similar('gandalf', topn = 20)
[('saruman', 0.7291593551635742),
 ('thorin', 0.6473978161811829),
 ('aragorn', 0.6401687264442444),
 ('isengard', 0.6123237013816833),
 ('orklar', 0.59786057472229),
 ('gollum', 0.5905635952949524),
 ('baggins', 0.5837421417236328),
 ('frodo', 0.5819021463394165),
 ('belgarath', 0.5811135172843933),
 ('sauron', 0.5763844847679138),
 ('elfler', 0.5758092999458313),
 ('bilbo', 0.5729959607124329),
 ('tyrion', 0.5728499889373779),
 ('rohan', 0.556411862373352),
 ('lancelot', 0.5517111420631409),
 ('mordor', 0.550175130367279),
 ('bran', 0.5482109189033508),
 ('goblin', 0.5393625497817993),
 ('thor', 0.5280926823616028),
 ('vader', 0.5258742570877075)]
 
>>> # FastText
>>> model = FastText.load('vnlp/turkish_embeddings/FastText_large.model')
>>> model.wv.most_similar('yamaçlardan', topn = 20)
[('kayalardan', 0.8601457476615906),
 ('kayalıklardan', 0.8567330837249756),
 ('tepelerden', 0.8423191905021667),
 ('ormanlardan', 0.8362939357757568),
 ('dağlardan', 0.8140010833740234),
 ('amaçlardan', 0.810560405254364),
 ('bloklardan', 0.803180992603302),
 ('otlardan', 0.8026642203330994),
 ('kısımlardan', 0.7993910312652588),
 ('ağaçlardan', 0.7961613535881042),
 ('dallardan', 0.7949419617652893),
 ('sahalardan', 0.7865065932273865),
 ('adalardan', 0.7819225788116455),
 ('sulardan', 0.7781057953834534),
 ('taşlardan', 0.7746424078941345),
 ('kuyulardan', 0.7689613103866577),
 ('köşelerden', 0.7678262591362),
 ('tünellerden', 0.7674043774604797),
 ('atlardan', 0.7657977342605591),
 ('kanatlardan', 0.7640945911407471)]

 - For more details about corpora, preprocessing and training, see `ReadMe <https://github.com/vngrs-ai/VNLP/blob/main/vnlp/turkish_word_embeddings/ReadMe.md>`_. 