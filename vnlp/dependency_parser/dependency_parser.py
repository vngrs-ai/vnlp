from typing import List, Tuple

from .spu_context_dp import SPUContextDP
from .treestack_dp import TreeStackDP


class DependencyParser:
    """
    Main API class for Dependency Parser implementations.

    Available models: ['SPUContextDP', 'TreeStackDP']

    In order to evaluate, initialize the class with "evaluate = True" argument. 
    This will load the model weights that are not trained on test sets.
    """

    def __init__(self, model = 'SPUContextDP', evaluate = False):
        self.models = ['SPUContextDP', 'TreeStackDP']
        self.evaluate = evaluate

        if model == 'SPUContextDP':
            self.instance = SPUContextDP(evaluate)

        elif model == 'TreeStackDP':
            self.instance = TreeStackDP(evaluate)
        
        else:
            raise ValueError(f'{model} is not a valid model. Try one of {self.models}')

    def predict(self, sentence: str, displacy_format: bool = False, pos_result: List[Tuple[str, str]] = None) -> List[Tuple[int, str, int, str]]:
        """
        High level user API for Dependency Parsing.

        Args:
            sentence:
                Input sentence.
            displacy_format:
                When set True, returns the result in spacy.displacy format to allow visualization.
            pos_result:
                Part of Speech tags. To be used when displacy_format = True.
        
        Returns:
            List of (token_index, token, arc, label).
                
        Raises:
            ValueError: Sentence is too long. Try again by splitting it into smaller pieces.

        Example::

            from vnlp import DependencyParser
            dependency_parser = DependencyParser()
            dependency_parser.predict("Onun için yol arkadaşlarımızı titizlikle seçer, kendilerini iyice sınarız.")

            [(1, 'Onun', 6, 'obl'),
             (2, 'için', 1, 'case'),
             (3, 'yol', 4, 'nmod'),
             (4, 'arkadaşlarımızı', 6, 'obj'),
             (5, 'titizlikle', 6, 'obl'),
             (6, 'seçer', 10, 'parataxis'),
             (7, ',', 6, 'punct'),
             (8, 'kendilerini', 10, 'obj'),
             (9, 'iyice', 10, 'advmod'),
             (10, 'sınarız', 0, 'root'),
             (11, '.', 10, 'punct')]
            
            # Visualization with Spacy:
            import spacy
            from vnlp import DependencyParser
            dependency_parser = DependencyParser()
            result = dependency_parser.predict(Oğuz'un kırmızı bir Astra'sı vardı.", displacy_format = True)
            spacy.displacy.render(result, style="dep", manual = True)
        """

        return self.instance.predict(sentence, displacy_format, pos_result)

    # this is called when an attribute is not found:
    def __getattr__(self, name):
        return self.instance.__getattribute__(name)