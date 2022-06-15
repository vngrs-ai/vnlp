### Sentiment Analyzer

- This is a Deep Bidirectional GRU based Sentiment Analysis classifier implementation.
- It uses SentencePiece Unigram tokenizer and pre-trained Word2Vec embeddings.
- It achieves 0.9469 Accuracy, 0.9381 on F1_macro_score and 0.9147 F1 score (treating class 0 as minority) on test set. Final model is trained on all of train, dev and test data.
- The data I gathered, compiled and standardized are found in the links below:
	- https://www.kaggle.com/baharyilmaz/turkish-sentiment-analysis/data
	- https://www.kaggle.com/egebasturk1/yemeksepeti-sentiment-analysis/data
	- https://www.kaggle.com/burhanbilenn/duygu-analizi-icin-urun-yorumlari
	- https://www.kaggle.com/burhanbilenn/turkish-customer-reviews-for-binary-classification
	- https://www.kaggle.com/anil1055/turkish-tweet-dataset
	- https://www.kaggle.com/seymasa/turkish-sales-comments
	- https://www.kaggle.com/busra88/turkish-reviews-dataset
	- https://www.kaggle.com/kbulutozler/5k-turkish-tweets-with-incivil-content
	- https://www.kaggle.com/berkaycokuner/yemek-sepeti-comments
	- https://github.com/boun-tabi/BounTi-Turkish-Sentiment-Analysis
	- https://github.com/ezgisubasi/turkish-tweets-sentiment-analysis
	- https://github.com/mertkahyaoglu/twitter-sentiment-analysis

- Merged training and test data can be accessed at: https://vnlp-datasets.s3.eu-west-1.amazonaws.com/sentiment_analysis/df_merged_sentiment_analysis_data.prq
- To evaluate, read this as df and split via "train_test_split(df, test_size = 0.10, random_state = 0, shuffle = True, stratify = df.loc[:, 'label'])"