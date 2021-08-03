from typing import List

from .model import AnalysisScorerModel

# Method by Eray Yıldız
# https://github.com/erayyildiz/LookupAnalyzerDisambiguator
# Using according to MIT License
class MorphAnalyzer():

    def __init__(self):
        self.model = AnalysisScorerModel.create_from_existed_model("lookup_disambiguator_wo_suffix")
    
    def predict(self, sentence: str) -> List[str]:
        """
        Given a sentence as string without punctuations,
        returns morphological analysis as list of strings.
        
        "Yarın ahmet'in doğum gününü kutlamayı unutma" ->
        ['yarın+Noun+A3sg+Pnon+Nom',
        'ahmet+Noun+Prop+A3sg+Pnon+Gen',
        'doğum+Noun+A3sg+Pnon+Nom',
        'gün+Noun+A3sg+P3sg+Acc',
        'kutla+Verb+Pos^DB+Noun+Inf2+A3sg+Pnon+Acc',
        'unut+Verb+Pos^DB+Noun+Inf2+A3sg+Pnon+Nom']
        """
        return self.model.predict(sentence.split(" "))