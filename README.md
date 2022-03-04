<img src="https://github.com/vngrs-ai/VNLP/blob/main/vnlp.png" width="256">

## VNLP: Turkish NLP Tools
State of the art, lightweight NLP tools for Turkish language.
Developed by VNGRS.
https://vngrs.com/

Consists of:
- Sentence Splitter
- Normalizer:
	- Spelling/Typo correction
	- Converts numbers to word form
	- Deascification
	- Lowercasing
	- Punctuation removal
	- Accent mark removal
- Stopword Remover:
	- Static
	- Dynamic
		- Frequent words
		- Rare words
- Stemmer: Morphological Analyzer & Disambiguator
- Named Entity Recognizer (NER) 
- Dependency Parser
- Part of Speech (POS) Tagger
- Sentiment Analyzer
- Turkish Embeddings
	- FastText
	- Word2Vec
	
### Compatability: works on
- Python 3.6
- Python 3.7
- Python 3.8
- Python 3.9
### Installation
open shell.
write
```
git clone https://github.com/vngrs-ai/VNLP.git
cd VNLP
python setup.py install
```

For Linux/MacOS, you might need to use
```
python3 setup.py install
```
instead.


### Usage:
#### CLI
```
$ vnlp --task sentiment_analysis --text "Sipariş geldiğinde biz karnımızı atıştırmalıklarla doyurmuştuk."
0

# To list available tasks/functionality:
$ vnlp --list_tasks
```

#### Python
**Sentence Splitter**
```
from vnlp import SentenceSplitter
sent_splitter = SentenceSplitter()

sent_splitter.split_sentences('Av. Meryem Beşer, 3.5 yıldır süren dava ile ilgili dedi ki, "Duruşma bitti, dava lehimize sonuçlandı." Bu harika bir haberdi!')
['Av. Meryem Beşer, 3.5 yıldır süren dava ile ilgili dedi ki, "Duruşma bitti, dava lehimize sonuçlandı."',
 'Bu harika bir haberdi!']
 
sent_splitter.split_sentences('4. Murat, diğer yazım şekli ile IV. Murat, alkollü içecekleri halka yasaklamıştı.')
['4. Murat, diğer yazım şekli ile IV. Murat, alkollü içecekleri halka yasaklamıştı.']
```

**Normalizer**
```
from vnlp import Normalizer
normalizer = Normalizer()

# Correct Spelling Mistakes and Typos
normalizer.correct_typos("kassıtlı yaezım hatasssı ekliyorumm".split())
['kasıtlı', 'yazım', 'hatası', 'ekliyorum']

# Convert Numbers to Word Form
normalizer.convert_numbers_to_words("sabah 3 yumurta yedim ve tartıldığımda 1,15 kilogram aldığımı gördüm".split())
['sabah',
'üç',
'yumurta',
'yedim',
've',
'tartıldığımda',
'bir',
'virgül',
'on',
'beş',
'kilogram',
'aldığımı',
'gördüm']

# Below are static methods that require no initialization
# Deasciification
Normalizer.deasciify("boyle sey gormedim duymadim".split())
['böyle', 'şey', 'görmedim', 'duymadım']

# Converts a string of text to lowercase for Turkish language.
Normalizer.lower_case("Test karakterleri: İIĞÜÖŞÇ")
'test karakterleri: iığüöşç'

# Punctuation Removal
Normalizer.remove_punctuations("noktalamalı test cümlesidir...")
'noktalamalı test cümlesidir'

# Remove accent marks
Normalizer.remove_accent_marks("merhâbâ gûzel yîlkî atî")
'merhaba guzel yılkı atı'
```

**Stopword Remover**
```
from vnlp import StopwordRemover
stopword_remover = StopwordRemover()

stopword_remover.drop_stop_words("acaba bugün kahvaltıda kahve yerine çay mı içsem ya da neyse süt içeyim".split())
['bugün', 'kahvaltıda', 'kahve', 'çay', 'içsem', 'süt', 'içeyim']
 
stopword_remover.dynamically_detect_stop_words("ben bugün gidip aşı olacağım sonra da eve gelip telefon açacağım aşı nasıl etkiledi eve gelip anlatırım aşı olmak bu dönemde çok ama ama ama ama çok önemli".split())
print(sr.dynamic_stop_words)
['ama', 'aşı', 'gelip', 'çok', 'eve', 'bu']

# adding dynamically detected stop words to stop words lexicon
stopword_remover.unify_stop_words()

# "aşı" has become a stopword now
stopword_remover.drop_stop_words("aşı olmak önemli demiş miydim".split())
['önemli', 'demiş', 'miydim']
```

**Stemmer: Morphological Analyzer & Disambiguator**
```
from vnlp import StemmerAnalyzer
stemmer_analyzer = StemmerAnalyzer()

stemmer_analyzer.predict("Eser miktardaki geçici bir güvenlik için temel özgürlüklerinden vazgeçenler, ne özgürlüğü ne de güvenliği hak ederler. Benjamin Franklin")
['eser+Noun+A3sg+Pnon+Nom',
 'miktar+Noun+A3sg+Pnon+Loc^DB+Adj+Rel',
 'geçici+Adj',
 'bir+Det',
 'güvenlik+Noun+A3sg+Pnon+Nom',
 'için+Postp+PCNom',
 'temel+Noun+A3sg+Pnon+Nom',
 'özgür+Adj^DB+Noun+Ness+A3sg+P3pl+Abl',
 'vazgeç+Verb+Pos^DB+Adj+PresPart^DB+Noun+Zero+A3pl+Pnon+Nom',
 ',+Punc',
 'ne+Adj',
 'özgür+Adj^DB+Noun+Ness+A3sg+P3sg+Nom',
 'ne+Adj',
 'de+Conj',
 'güvenlik+Noun+A3sg+P3sg+Nom',
 'hak+Noun+A3sg+Pnon+Nom',
 'et+Verb+Pos+Aor+A3pl',
 '.+Punc',
 'benjamin+Noun+A3sg+Pnon+Nom',
 'franklin+Noun+A3sg+Pnon+Nom']
```
 
**Named Entity Recognizer (NER)**
```
from vnlp import NamedEntityRecognizer
ner = NamedEntityRecognizer()

ner.predict("Benim adım Melikşah, 29 yaşındayım, İstanbul'da ikamet ediyorum ve VNGRS AI Takımı'nda Aydın ile birlikte çalışıyorum.")
[('Benim', 'O'),
 ('adım', 'O'),
 ('Melikşah', 'PER'),
 (',', 'O'),
 ('29', 'O'),
 ('yaşındayım', 'O'),
 (',', 'O'),
 ('İstanbul', 'LOC'),
 ("'", 'O'),
 ('da', 'O'),
 ('ikamet', 'O'),
 ('ediyorum', 'O'),
 ('ve', 'O'),
 ('VNGRS', 'ORG'),
 ('AI', 'ORG'),
 ('Takımı', 'ORG'),
 ("'", 'O'),
 ('nda', 'O'),
 ('Aydın', 'PER'),
 ('ile', 'O'),
 ('birlikte', 'O'),
 ('çalışıyorum', 'O'),
 ('.', 'O')]
```

**Dependency Parser**
```
from vnlp import DependencyParser
dep_parser = DependencyParser()

dep_parser.predict("Onun için yol arkadaşlarımızı titizlikle seçer, kendilerini iyice sınarız.")
[(1, 'Onun', 5, 'obl'),
(2, 'için', 1, 'case'),
(3, 'yol', 1, 'nmod'),
(4, 'arkadaşlarımızı', 5, 'obj'),
(5, 'titizlikle', 6, 'obl'),
(6, 'seçer', 7, 'acl'),
(7, ',', 10, 'punct'),
(8, 'kendilerini', 10, 'obj'),
(9, 'iyice', 8, 'advmod'),
(10, 'sınarız', 0, 'root'),
(11, '.', 10, 'punct')]
```

**Part of Speech (POS) Tagger**
```
from vnlp import PoSTagger
pos_tagger = PoSTagger()

pos_tagger.predict("Vapurla Beşiktaş'a geçip yürüyerek Maçka Parkı'na ulaştım.")

[('Vapurla', 'NOUN'),
 ("Beşiktaş'a", 'PROPN'),
 ('geçip', 'ADV'),
 ('yürüyerek', 'VERB'),
 ('Maçka', 'PROPN'),
 ("Parkı'na", 'NOUN'),
 ('ulaştım', 'VERB'),
 ('.', 'PUNCT')]
```

**Sentiment Analyzer**
```
from vnlp import SentimentAnalyzer
sentiment_analyzer = SentimentAnalyzer()

sentiment_analyzer.predict_proba("Sipariş geldiğinde biz karnımızı atıştırmalıklarla doyurmuştuk.")

0.007

sentiment_analyzer.predict("Servis daha iyi olabilirdi ama lezzet ve hız geçer not aldı.")

1

sentiment_analyzer.predict_proba("Yemekleriniz o kadar şahaneydi ki artık uzun bir süre meksika yemeği yemeyi düşünmüyorum.")

0.448
```

**Turkish Embeddings: Word2Vec & FastText:**
- They come in 3 sizes:
	- Large: vocabulary_size: 128_000, embedding_size: 256
	- Medium: vocabulary_size: 64_000, embedding_size: 128
	- Small: vocabulary_size: 32_000, embedding_size: 64

- Can be downloaded from below links:
	- Large:
		- Word2Vec: https://meliksahturker.s3.us-east-2.amazonaws.com/VNLP/turkish_word_embeddings/Word2Vec_large.zip
		- FastText: https://meliksahturker.s3.us-east-2.amazonaws.com/VNLP/turkish_word_embeddings/FastText_large.zip
	- Medium:
		- Word2Vec: https://meliksahturker.s3.us-east-2.amazonaws.com/VNLP/turkish_word_embeddings/Word2Vec_medium.zip
		- FastText: https://meliksahturker.s3.us-east-2.amazonaws.com/VNLP/turkish_word_embeddings/FastText_medium.zip
		
	- Small:
		- Word2Vec: https://meliksahturker.s3.us-east-2.amazonaws.com/VNLP/turkish_word_embeddings/Word2Vec_small.zip
		- FastText: https://meliksahturker.s3.us-east-2.amazonaws.com/VNLP/turkish_word_embeddings/FastText_small.zip


You need gensim to execute the sample code below.
```
from gensim.models import Word2Vec, FastText
# Word2Vec
model = Word2Vec.load('vnlp/turkish_embeddings/Word2Vec_large.model')
model.wv.most_similar('gandalf', topn = 20)
[('saruman', 0.7291593551635742),
 ('thorin', 0.6473978161811829),
 ('aragorn', 0.6401687264442444),
 ('isengard', 0.6123237013816833),
 ('orklar', 0.59786057472229),
 ('gollum', 0.5905635952949524),
 ('baggins', 0.5837421417236328),
 ('frodo', 0.5819021463394165),
 ('belgarath', 0.5811135172843933),
 ('sauron', 0.5763844847679138),
 ('elfler', 0.5758092999458313),
 ('bilbo', 0.5729959607124329),
 ('tyrion', 0.5728499889373779),
 ('rohan', 0.556411862373352),
 ('lancelot', 0.5517111420631409),
 ('mordor', 0.550175130367279),
 ('bran', 0.5482109189033508),
 ('goblin', 0.5393625497817993),
 ('thor', 0.5280926823616028),
 ('vader', 0.5258742570877075)]
 
# FastText
model = Word2Vec.load('vnlp/turkish_embeddings/FastText_large.model')
model.wv.most_similar('yamaçlardan', topn = 20)
[('kayalardan', 0.8601457476615906),
 ('kayalıklardan', 0.8567330837249756),
 ('tepelerden', 0.8423191905021667),
 ('ormanlardan', 0.8362939357757568),
 ('dağlardan', 0.8140010833740234),
 ('amaçlardan', 0.810560405254364),
 ('bloklardan', 0.803180992603302),
 ('otlardan', 0.8026642203330994),
 ('kısımlardan', 0.7993910312652588),
 ('ağaçlardan', 0.7961613535881042),
 ('dallardan', 0.7949419617652893),
 ('sahalardan', 0.7865065932273865),
 ('adalardan', 0.7819225788116455),
 ('sulardan', 0.7781057953834534),
 ('taşlardan', 0.7746424078941345),
 ('kuyulardan', 0.7689613103866577),
 ('köşelerden', 0.7678262591362),
 ('tünellerden', 0.7674043774604797),
 ('atlardan', 0.7657977342605591),
 ('kanatlardan', 0.7640945911407471)]
```
