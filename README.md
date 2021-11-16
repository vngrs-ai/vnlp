### Turkish-NLP-preprocessing-module
NLP Preprocessing module for Turkish language

Consists of:
- Sentence Splitter
- Normalizer:
	- Punctuation Remover
	- Convert numbers to word form
	- Remove accent marks
	- Spelling Mistake & Typo correction using:
		- Pre-defined typos lexicon
		- Levenshtein distance
		- Morphological Analyzer
- Stopword Remover:
	- Static
	- Dynamic
		- Frequent words
		- Rare words
- Stemmer: Morphological Analyzer & Disambiguator
- NER: Named Entity Recognizer
- Turkish Embeddings
	- FastText
	- Word2Vec
	

#### Usage:
**Sentence Splitter:**
```
from pp.sentence_splitter import SentenceSplitter
ss = SentenceSplitter()

ss.split_sentences('Av. Meryem Beşer, 3.5 yıldır süren dava ile ilgili dedi ki, "Duruşma bitti, dava lehimize sonuçlandı." Bu harika bir haber!')
['Av. Meryem Beşer, 3.5 yıldır süren dava ile ilgili dedi ki, "Duruşma bitti, dava lehimize sonuçlandı."',
 'Bu harika bir haber!']
 
ss.split_sentences('4. Murat, diğer yazım şekli ile IV. Murat, alkollü içecekleri halka yasaklamıştı.')
['4. Murat, diğer yazım şekli ile IV. Murat, alkollü içecekleri halka yasaklamıştı.']
```

**Normalizer:**
```
from pp.normalizer import Normalizer
n = Normalizer()

# Correct Spelling Mistakes and Typos
n.correct_typos("kasitli yazişm hatasıı ekliyoruum".split(), use_levenshtein = True)
['kasıtlı', 'yazım', 'hatası', 'ekliyorum']

# Punctuation Removal
n.remove_punctuations("noktalamalı test cümlesidir...")
'noktalamalı test cümlesidir'
 
# Deasciification
n.deasciify("boyle sey gormedim duymadim".split())
['böyle', 'şey', 'görmedim', 'duymadım']

# Convert Numbers to Word Form
n.convert_number_to_word("sabah 3 yumurta yedim".split())
['sabah', 'üç', 'yumurta', 'yedim']

# Remove accent marks
n.remove_accent_marks("merhâbâ gûzel yîlkî atî")
'merhaba guzel yılkı atı'
```

**Stopword Remover:**
```
from pp.stopword_remover import StopwordRemover
sr = StopwordRemover()

sr.drop_stop_words("acaba bugün kahvaltıda kahve yerine çay mı içsem ya da neyse süt içeyim".split())
['bugün', 'kahvaltıda', 'kahve', 'çay', 'içsem', 'süt', 'içeyim']
 
sr.dynamically_detect_stop_words("ben bugün gidip aşı olacağım sonra da eve gelip telefon açacağım aşı nasıl etkiledi eve gelip anlatırım aşı olmak bu dönemde çok ama ama ama ama çok önemli".split())
print(sr.dynamic_stop_words)
['ama', 'aşı', 'gelip', 'çok', 'eve', 'bu']

# adding dynamically detected stop words to stop words lexicon
sr.unify_stop_words()

# "aşı" has become a stopword now
sr.drop_stop_words("aşı olmak önemli demiş miydim".split())
['önemli', 'demiş', 'miydim']
```

**Stemmer: Morphological Analyzer & Disambiguator:**
```
from pp.stemmer_morph_analyzer import StemmerAnalyzer
sa = StemmerAnalyzer()

sa.predict("Eser miktardaki geçici bir güvenlik için temel özgürlüklerinden vazgeçenler, ne özgürlüğü ne de güvenliği hak ederler. Benjamin Franklin")
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
 
**NER: Named Entity Recognizer:**
```
from pp.named_entitiy_recognizer import NamedEntityRecognizer
ner = NamedEntityRecognizer()

ner.predict("Benim adım Melikşah, 28 yaşındayım, İstanbul'da ikamet ediyorum ve VNGRS AI Takımı'nda Aydın ile birlikte çalışıyorum.")
[('Benim', 'O'),
 ('adım', 'O'),
 ('Melikşah', 'PER'),
 (',', 'O'),
 ('28', 'O'),
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

**Turkish Embeddings: Word2Vec & FastText:**
Download from below links first and place under directory pp/turkish_embeddings:
- Word2Vec: https://drive.google.com/drive/folders/172rLVXgMwTl3MwXdgXGn9qHnlSYt7AdO?usp=sharing
- FastText: https://drive.google.com/drive/folders/1FnmS1bVtOKK49D-PHzTp740No7MdJlEE?usp=sharing
```
from gensim.models import Word2Vec, FastText
# Word2Vec
model = Word2Vec.load('pp/turkish_embeddings/Word2Vec.model')
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
model = Word2Vec.load('pp/turkish_embeddings/FastText.model')
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