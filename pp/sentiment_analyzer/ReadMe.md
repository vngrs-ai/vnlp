#### Sentiment Analyzer

- This is a Deep GRU based Sentiment Analysis classifier implementation.
- It uses pre-trained Word2Vec_medium embeddings, another part of this project as word embeddings.
- It achieves 0.9348 Accuracy, 0.9220 on F1_macro_score and 0.8914 F1 score (treating class 0 as minority) on test set. Final model is trained on all of train, dev and test data.
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