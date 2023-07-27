#### Normalizer

Contains following functionality:
- Spelling uses [jamspell](https://github.com/bakwc/JamSpell/) algorithm. Model is trained on a mixed Turkish corpus.
- Deasciification
- Convert numbers to word form
- Lower case
- Punctuation Remover
- Remove accent marks

- Deascification is ported from Emre Sevinç's implementation in https://github.com/emres/turkish-deasciifier
- Converting numbers to words is an improved version of https://github.com/Omerktn/Turkish-Lexical-Representation-of-Numbers/blob/master/src.py