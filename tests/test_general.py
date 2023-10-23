import unittest

from vnlp import (
    DependencyParser,
    PoSTagger,
    NamedEntityRecognizer,
    StemmerAnalyzer,
    SentimentAnalyzer,
    Normalizer,
    StopwordRemover,
)

import sys


class StemmerTest(unittest.TestCase):
    def setUp(self):
        self.analyzer = StemmerAnalyzer()

    def test_predict_1(self):
        res = self.analyzer.predict(
            "Üniversite sınavlarına canla başla çalışıyorlardı."
        )
        expected = [
            "üniversite+Noun+A3sg+Pnon+Nom",
            "sınav+Noun+A3pl+P3sg+Dat",
            "can+Noun+A3sg+Pnon+Ins",
            "baş+Noun+A3sg+Pnon+Ins",
            "çalış+Verb+Pos+Prog1+A3pl+Past",
            ".+Punc",
        ]
        self.assertEqual(res, expected)

    def test_predict_2(self):
        res = self.analyzer.predict("Şimdi baştan başla.")
        expected = [
            "şimdi+Adverb",
            "baş+Noun+A3sg+Pnon+Abl",
            "başla+Verb+Pos+Imp+A2sg",
            ".+Punc",
        ]
        self.assertEqual(res, expected)


class NerTest(unittest.TestCase):
    def setUp(self):
        self.ner = NamedEntityRecognizer()

    def test_predict_1(self):
        res = self.ner.predict(
            "Benim adım Melikşah, 29 yaşındayım, İstanbul'da ikamet ediyorum ve VNGRS AI Takımı'nda çalışıyorum."
        )
        expected = [
            ("Benim", "O"),
            ("adım", "O"),
            ("Melikşah", "PER"),
            (",", "O"),
            ("29", "O"),
            ("yaşındayım", "O"),
            (",", "O"),
            ("İstanbul'da", "LOC"),
            ("ikamet", "O"),
            ("ediyorum", "O"),
            ("ve", "O"),
            ("VNGRS", "ORG"),
            ("AI", "ORG"),
            ("Takımı'nda", "ORG"),
            ("çalışıyorum", "O"),
            (".", "O"),
        ]
        self.assertEqual(res, expected)


class PoSTest(unittest.TestCase):
    def setUp(self):
        self.pos_tagger = PoSTagger()

    def test_predict_1(self):
        res = self.pos_tagger.predict("Oğuz'un kırmızı bir Astra'sı vardı.")
        expected = [
            ("Oğuz'un", "PROPN"),
            ("kırmızı", "ADJ"),
            ("bir", "DET"),
            ("Astra'sı", "PROPN"),
            ("vardı", "VERB"),
            (".", "PUNCT"),
        ]
        self.assertEqual(res, expected)


class DependencyParserTest(unittest.TestCase):
    def setUp(self):
        self.dep_parser = DependencyParser()

    def test_predict_1(self):
        res = self.dep_parser.predict(
            "Onun için yol arkadaşlarımızı titizlikle seçer, kendilerini iyice sınarız."
        )
        expected = [
            (1, "Onun", 6, "obl"),
            (2, "için", 1, "case"),
            (3, "yol", 4, "nmod"),
            (4, "arkadaşlarımızı", 6, "obj"),
            (5, "titizlikle", 6, "obl"),
            (6, "seçer", 10, "parataxis"),
            (7, ",", 6, "punct"),
            (8, "kendilerini", 10, "obj"),
            (9, "iyice", 10, "advmod"),
            (10, "sınarız", 0, "root"),
            (11, ".", 10, "punct"),
        ]
        self.assertEqual(res, expected)


class SentimentAnalyzerTester(unittest.TestCase):
    def setUp(self):
        self.sentiment_analyzer = SentimentAnalyzer()

    def test_predicts(self):
        _places = 1  # 1 decimal places is enough for this test
        _msg = "Sentiment analyzer results could change if the model is updated. Please check the results manually and replace the values if necessary."
        self.assertAlmostEqual(
            self.sentiment_analyzer.predict_proba(
                "Zeynep'in okul taksitini Bürokratistan'daki Parabank'tan yatırdım."
            ),
            0.98,
            places=_places,
            msg=_msg,
        )
        self.assertAlmostEqual(
            self.sentiment_analyzer.predict_proba(
                "Sipariş geldiğinde biz karnımızı çoktan atıştırmalıklarla doyurmuştuk."
            ),
            0.08,
            places=_places,
            msg=_msg,
        )
        self.assertAlmostEqual(
            self.sentiment_analyzer.predict_proba(
                "Servis daha iyi olabilirdi ama lezzet ve hız geçer not aldı."
            ),
            1.00,
            places=_places,
            msg=_msg,
        )
        self.assertAlmostEqual(
            self.sentiment_analyzer.predict_proba("Mutlu değilim diyemem."),
            0.85,
            places=_places,
            msg=_msg,
        )
        self.assertAlmostEqual(
            self.sentiment_analyzer.predict_proba(
                "Geçmesin günümüz sevgilim yasla, o güzel başını göğsüme yasla."
            ),
            0.99,
            places=_places,
            msg=_msg,
        )


class NormalizerTester(unittest.TestCase):
    def setUp(self):
        self.normalizer = Normalizer()

    def test_correct_typos(self):
        self.assertEqual(
            self.normalizer.correct_typos("kassıtlı yazım hatası ekliyorumm"),
            "kasıtlı yazım hatası ekliyorum",
        )

    def test_convert_number_to_words(self):
        self.assertEqual(
            self.normalizer.convert_numbers_to_words(
                "sabah 2 yumurta yedim ve tartıldığımda 1,15 kilogram aldığımı gördüm".split()
            ),
            [
                "sabah",
                "iki",
                "yumurta",
                "yedim",
                "ve",
                "tartıldığımda",
                "bir",
                "virgül",
                "on",
                "beş",
                "kilogram",
                "aldığımı",
                "gördüm",
            ],
        )

    def test_deasciify(self):
        self.assertEqual(
            self.normalizer.deasciify("boyle sey gormedim duymadim".split()),
            ["böyle", "şey", "görmedim", "duymadım"],
        )
        self.assertEqual(
            self.normalizer.deasciify("yatirdim".split()), ["yatırdım"]
        )

    def test_misc(self):
        self.assertEqual(
            self.normalizer.lower_case("Test karakterleri: İIĞÜÖŞÇ"),
            "test karakterleri: iığüöşç",
        )
        self.assertEqual(
            self.normalizer.remove_punctuations(
                "noktalamalı test cümlesidir..."
            ),
            "noktalamalı test cümlesidir",
        )
        self.assertEqual(
            self.normalizer.remove_accent_marks("merhâbâ gûzel yîlkî atî"),
            "merhaba guzel yılkı atı",
        )


class StopwordRemoverTest(unittest.TestCase):
    def setUp(self):
        self.stopword_remover = StopwordRemover()

    def test_remove_stopwords(self):
        self.assertEqual(
            self.stopword_remover.drop_stop_words(
                "acaba bugün kahvaltıda kahve yerine çay mı içsem ya da neyse süt içeyim".split()
            ),
            ["bugün", "kahvaltıda", "kahve", "çay", "içsem", "süt", "içeyim"],
        )

    def test_dynamic_stopwords(self):
        py_version = int(sys.version.split('.')[1])
        dsw = self.stopword_remover.dynamically_detect_stop_words(
            "ben bugün gidip aşı olacağım sonra da eve gelip telefon açacağım aşı nasıl etkiledi eve gelip anlatırım aşı olmak bu dönemde çok ama ama ama ama çok önemli".split()
        )
        expected = ["ama", "aşı", "çok", "eve"]
        if py_version <= 8: #Sorting algorithm returns different results from python 3.8+ on
            expected = ["ama", "aşı", "gelip", "eve"]
        self.assertEqual(dsw, expected)
        self.stopword_remover.add_to_stop_words(dsw)
        self.assertEqual(
            self.stopword_remover.drop_stop_words(
                "aşı olmak önemli demiş miydim".split()
            ),
            ["önemli", "demiş", "miydim"],
        )

if __name__ == "__main__":
    unittest.main()
