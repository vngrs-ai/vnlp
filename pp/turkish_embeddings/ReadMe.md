#### Word2Vec and FastText embeddings

- They come in 3 sizes:
	- Large: vocabulary_size: 128_000, embedding_size: 256
	- Medium: vocabulary_size: 64_000, embedding_size: 128
	- Small: vocabulary_size: 32_000, embedding_size: 64

Can be downloaded from:
- Large:
	- Word2Vec: https://meliksahturker.s3.us-east-2.amazonaws.com/turkish-embeddings/trained_models/Word2Vec_large.zip
	- FastText: https://meliksahturker.s3.us-east-2.amazonaws.com/turkish-embeddings/trained_models/FastText_large.zip
- Medium:
	- Word2Vec: https://meliksahturker.s3.us-east-2.amazonaws.com/turkish-embeddings/trained_models/Word2Vec_medium.zip
	- FastText: https://meliksahturker.s3.us-east-2.amazonaws.com/turkish-embeddings/trained_models/FastText_medium.zip
	
- Small:
	- Word2Vec: https://meliksahturker.s3.us-east-2.amazonaws.com/turkish-embeddings/trained_models/Word2Vec_small.zip
	- FastText: https://meliksahturker.s3.us-east-2.amazonaws.com/turkish-embeddings/trained_models/FastText_small.zip

- Trained on a corpus of 32 GBs, made of 288 million sentences and 4.07 billion words consisting of:
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
- Trained for 10 epochs with window value of 5.