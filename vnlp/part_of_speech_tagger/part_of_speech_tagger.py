from typing import List, Tuple

from .spu_context_pos import SPUContextPoS
from .treestack_pos import TreeStackPoS


class PoSTagger:
    """
    Main API class for Part of Speech Tagger implementations.

    Available models: ['SPUContextPoS', 'TreeStackPoS']

    In order to evaluate, initialize the class with "evaluate = True" argument.
    This will load the model weights that are not trained on test sets.
    """

    def __init__(self, model="SPUContextPoS", evaluate=False, *args):
        self.models = ["SPUContextPoS", "TreeStackPoS"]
        self.evaluate = evaluate

        if model == "SPUContextPoS":
            self.instance = SPUContextPoS(evaluate)

        elif model == "TreeStackPoS":
            if args:
                stemmer_analyzer = args[0]
            else:
                stemmer_analyzer = None
            self.instance = TreeStackPoS(evaluate, stemmer_analyzer)

        else:
            raise ValueError(
                f"{model} is not a valid model. Try one of {self.models}"
            )

    def predict(self, sentence: str) -> List[Tuple[str, str]]:
        """
        High level user API for Part of Speech Tagging.

        Args:
            sentence:
                Input text(sentence).

        Returns:
             List of (token, pos_label).

        Example::

            from vnlp import PoSTagger
            pos = PoSTagger()
            pos.predict("Vapurla Beşiktaş'a geçip yürüyerek Maçka Parkı'na ulaştım.")

            [("Oğuz'un", 'PROPN'),
             ('kırmızı', 'ADJ'),
             ('bir', 'DET'),
             ("Astra'sı", 'PROPN'),
             ('vardı', 'VERB'),
             ('.', 'PUNCT')]

        """

        return self.instance.predict(sentence)

    # this is called when an attribute is not found:
    def __getattr__(self, name):
        return self.instance.__getattribute__(name)
