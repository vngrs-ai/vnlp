from .spu_context_bigru_sentiment import SPUCBiGRUSentimentAnalyzer

class SentimentAnalyzer:
    """
    Main API class for Sentiment Analyzer implementations.

    Available models: ['SPUCBiGRUSentimentAnalyzer']

    In order to evaluate, initialize the class with "evaluate = True" argument.
    This will load the model weights that are not trained on test sets.
    """

    def __init__(self, model = 'SPUCBiGRUSentimentAnalyzer', evaluate = False):
        self.models = ['SPUCBiGRUSentimentAnalyzer']
        self.evaluate = evaluate

        if model == 'SPUCBiGRUSentimentAnalyzer':
            self.instance = SPUCBiGRUSentimentAnalyzer(evaluate)
        
        else:
            raise ValueError(f'{model} is not a valid model. Try one of {self.models}')

    def predict(self, text: str) -> int:
        """
        High level user API for discrete Sentiment Analysis prediction.
        
        1: Positive sentiment.
        
        0: Negative sentiment.

        Args:
            text: 
                Input text.

        Returns:
             Sentiment label of input text.

        Example::

            from vnlp import SentimentAnalyzer
            sentiment_analyzer = SentimentAnalyzer()
            sentiment_analyzer.predict("Sipariş geldiğinde biz karnımızı çoktan atıştırmalıklarla doyurmuştuk.")
            
            0
        """

        return self.instance.predict(text)

    def predict_proba(self, text: str) -> float:
        """
        High level user API for probability estimation of Sentiment Analysis.

        Args:
            text: 
                Input text.

        Returns:
            Probability that the input text has positive sentiment.

        Example::
        
            from vnlp import SentimentAnalyzer
            sentiment_analyzer = SentimentAnalyzer()
            sentiment_analyzer.predict_proba("Sipariş geldiğinde biz karnımızı çoktan atıştırmalıklarla doyurmuştuk.")
            
            0.08
        """

        return self.instance.predict_proba(text)

    # this is called when an attribute is not found:
    def __getattr__(self, name):
        return self.instance.__getattribute__(name)