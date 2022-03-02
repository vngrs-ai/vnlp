import argparse

from vnlp import (StemmerAnalyzer, NamedEntityRecognizer, DependencyParser, PoSTagger,
                  SentimentAnalyzer, SentenceSplitter, Normalizer)


def main():
    help_text = 'usage: vnlp [--task] [--text] \
    \
    # List of available tasks:\
    stemming_morph_analysis\
    named_entity_recognition\
    dependency_parsing\
    part_of_speech_tagging\
    sentiment_analysis\
    split_sentences\
    correct_typos\
    convert_numbers_to_words\
    deasciify\
    lower_case\
    remove_punctuations\
    remove_accent_marks\
    drop_stop_words\
    \
    # Example usage:\
    $ vnlp --task stemming_morph_analysis --text "Üniversite sınavlarına canla başla çalışıyorlardı."\
    '

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help=help_text)
    args = parser.parse_args()

    task = args.task
    input_text = args.text

    if task == 'stemming_morph_analysis':
        stemmer_analyzer = StemmerAnalyzer()
        result = stemmer_analyzer.predict(input_text)
    
    elif task == 'named_entity_recognition':
        ner = NamedEntityRecognizer()
        result = ner.predict(input_text)

    elif task == 'dependency_parsing':
        dependency_parser = DependencyParser()
        result = dependency_parser.predict(input_text)

    elif task == 'part_of_speech_tagging':
        pos_tagger = PoSTagger()
        result = pos_tagger.predict(input_text)

    elif task == 'sentiment_analysis':
        sentiment_analyzer = SentimentAnalyzer()
        result = sentiment_analyzer.predict(input_text)

    elif task == 'split_sentences':
        sentence_splitter = SentenceSplitter()
        result = sentence_splitter.split_sentences(input_text)

    elif task == 'correct_typos':
        normalizer = Normalizer()
        result = normalizer.correct_typos(input_text.split())

    elif task == 'convert_numbers_to_words':
        normalizer = Normalizer()
        result = normalizer.convert_numbers_to_words(input_text.split())

    elif task == 'deasciify':
        result = Normalizer.deasciify(input_text.split())

    elif task == 'lower_case':
        result = Normalizer.lower_case(input_text)

    elif task == 'remove_punctuations':
        result = Normalizer.remove_punctuations(input_text)

    elif task == 'remove_accent_marks':
        result = Normalizer.remove_accent_marks(input_text)

    elif task == 'drop_stop_words':
        stopword_remover = StopwordRemover()
        result = stopword_remover.drop_stop_words(input_text.split())

    else:
        print(help_text)

    print(result)
    return result


if __name__ == "__main__":
    main()