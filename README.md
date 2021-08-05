# Turkish-NLP-preprocessing-module
NLP Preprocessing module for Turkish language

Consists of:
- Morphological Analyzer/Disambiguator
- Normalizer
- Sentence Splitter
- Stopword Remover
- Tokenizer

#### ------------------------------------------------------------------------------------------------


**Morphological Analysis**
- Lemmatization

**Normalization:**
- Removes punctuations
- Lowercase
- Converts numbers to word form
- Removes accent marks
- Spelling/typo correction
    - pre-defined typos lexicon
    - Levenshtein distance
    
**Sentence Splitting**

**Stopwords:**
- Static
- Dynamic
    - frequent words
    - rare words

**Tokenization**

#### ------------------------------------------------------------------------------------------------

**Recently DONE**:
- Tokenizer removes extra whitespaces
- detect_rare_words flag of Normalizer is updated and is now a more flexible, integer argument.
- Documentation is improved:
    - Doc-strings are updated with Typing
    - Readme.md file is expanded with details, DONE, TODO, Potential improvements and resources

**TODO:**
- Replace print statements with logging
- Expand lexicons
- Find a better Tokenizer

**Potential improvements:**
- Named Entity Recognition:
    - Person, Location, Number, Date
    - Remove HTML: convert it to <HTML_LINK> token
    - Remove URLs: convert it to <URL> token
    - Remove Emoji and Emoticon: convert to <EMO> token
        - good emoticon replacer with positive and negative sentiments: https://www.kaggle.com/egebasturk1/yemeksepeti-sentiment-analysis
- Expand abbreviations (Vowelizer): e.g "mrb" -> "merhaba"
    - it is mentioned in https://aclanthology.org/E14-2001.pdf
- Remove duplicate characters: e.g "yeemeeeeek haaarikaydııı" -> "yemek harikaydı"
- Syllabication: e.g "unutmadım" -> ["u", "nut", "ma", "dım"]
    - Can be useful for syllable-level models
- Remove strings with len less than 2/3 (can be added optionally to Normalizer)
- Noise removal (optional and depends on context e.g html tags, twitter hashtags, etc)
    
    
**Some nice resources that can be used to improve our tool:**
- https://github.com/topics/turkish-nlp
- https://pypi.org/project/spark-nlp/
- https://github.com/MeteHanC/turkishnlp