from .model import AnalysisScorerModel

# Method by Eray Yıldız
# https://github.com/erayyildiz/LookupAnalyzerDisambiguator
# Using according to MIT License
class MorphAnalyzer():

    def __init__(self):
        self.model = AnalysisScorerModel.create_from_existed_model("lookup_disambiguator_wo_suffix")

    def predict(self, sentence):
        """
        Given a sentence as string without punctuations,
        returns morphological analysis as list of strings.
        """
        return self.model.predict(sentence.split(" "))