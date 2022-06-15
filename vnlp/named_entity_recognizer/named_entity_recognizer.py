from typing import List, Tuple

from .charner import CharNER
from .spu_context_ner import SPUContextNER

class NamedEntityRecognizer:
    """
    Main API class for Named Entity Recognizer implementations.

    Available models: ['SPUContextNER', 'CharNER']

    In order to evaluate, initialize the class with "evaluate = True" argument.
    This will load the model weights that are not trained on test sets.
    """

    def __init__(self, model = 'SPUContextNER', evaluate = False):
        self.models = ['SPUContextNER', 'CharNER']
        self.evaluate = evaluate

        if model == 'SPUContextNER':
            self.instance = SPUContextNER(evaluate)

        elif model == 'CharNER':
            self.instance = CharNER(evaluate)
        
        else:
            raise ValueError(f'{model} is not a valid model. Try one of {self.models}')

    def predict(self, sentence: str, displacy_format: bool = False) -> List[Tuple[str, str]]:
        """
        High level user API for Named Entity Recognition.

        Args:
            sentence:
                Input sentence/text.
            displacy_format:
                When set True, returns the result in spacy.displacy format to allow visualization.

        Returns:
            NER result as pairs of (token, entity).

        Example::
        
            from vnlp import NamedEntityRecognizer
            ner = NamedEntityRecognizer()
            ner.predict("Benim adım Melikşah, 29 yaşındayım, İstanbul'da ikamet ediyorum ve VNGRS AI Takımı'nda çalışıyorum.")

            [('Benim', 'O'),
            ('adım', 'O'),
            ('Melikşah', 'PER'),
            (',', 'O'),
            ('29', 'O'),
            ('yaşındayım', 'O'),
            (',', 'O'),
            ("İstanbul'da", 'LOC'),
            ('ikamet', 'O'),
            ('ediyorum', 'O'),
            ('ve', 'O'),
            ('VNGRS', 'ORG'),
            ('AI', 'ORG'),
            ("Takımı'nda", 'ORG'),
            ('çalışıyorum', 'O'),
            ('.', 'O')]

            # Visualization with Spacy:
            import spacy
            from vnlp import NamedEntityRecognizer
            ner = NamedEntityRecognizer()
            result = ner.predict("İstanbul'dan Foça'ya giderken Zeynep ile Bursa'ya uğradık.", displacy_format = True)
            spacy.displacy.render(result, style="ent", manual = True)
        """

        return self.instance.predict(sentence, displacy_format)