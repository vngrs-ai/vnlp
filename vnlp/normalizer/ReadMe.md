#### Normalizer

Contains functionality:
- Spelling/typo correction uses Stemmer and Hunspell(tdd-hunspell-tr-1.1.0 dict) algorithm. For details see at pp/_resources/tdd-hunspell-tr-1.1.0/README.MD
- Deasciification
- Convert numbers to word form
- Lower case
- Punctuation Remover
- Remove accent marks

- Deascification is ported from Emre Sevinç's implementation in https://github.com/emres/turkish-deasciifier
- Converting numbers to words is an improved version of https://github.com/Omerktn/Turkish-Lexical-Representation-of-Numbers/blob/master/src.py