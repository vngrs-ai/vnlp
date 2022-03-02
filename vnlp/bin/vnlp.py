import argparse
import importlib


def main():
    """
    List of tasks:

    # Deep NN based functions
    stemming_morph_analysis
    named_entity_recognition
    dependency_parsing
    part_of_speech_tagging
    sentiment_analysis

    # Rule, Regex and Lexicon based functions
    split_sentences
    correct_typos
    convert_numbers_to_words
    deasciify
    lower_case
    remove_punctuations
    remove_accent_marks
    drop_stop_words
    
    Example:

    $ vnlp --task stemming_morph_analysis --text "Üniversite sınavlarına canla başla çalışıyorlardı."

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    args = parser.parse_args()

    task = args.task
    input_text = args.text

    if task == 'stemming_morph_analysis':
        importlib.import_module('StemmerAnalyzer', 'vnlp')

        stemmer_analyzer = StemmerAnalyzer()
        output_text = stemmer_analyzer.predict(input_text)
    
    elif task == 'named_entity_recognition':
        importlib.import_module('NamedEntityRecognizer', 'vnlp')

        ner = NamedEntityRecognizer()
        output_text = ner.predict(input_text)

    elif task == 'dependency_parsing':
        importlib.import_module('DependencyParser', 'vnlp')

        dependency_parser = DependencyParser()
        output_text = dependency_parser.predict(input_text)

    elif task == 'part_of_speech_tagging':
        importlib.import_module('PoSTagger', 'vnlp')

        pos_tagger = PoSTagger()
        output_text = pos_tagger.predict(input_text)

    elif task == 'sentiment_analysis':
        importlib.import_module('SentimentAnalyzer', 'vnlp')

        sentiment_analyzer = SentimentAnalyzer()
        output_text = sentiment_analyzer.predict(input_text)

    elif task == 'split_sentences':
        importlib.import_module('SentenceSplitter', 'vnlp')

        sentence_splitter = SentenceSplitter()
        output_text = sentence_splitter.split_sentences(input_text)

    elif task == 'correct_typos':
        importlib.import_module('Normalizer', 'vnlp')

        normalizer = Normalizer()
        output_text = normalizer.correct_typos(input_text.split())

    elif task == 'convert_numbers_to_words':
        importlib.import_module('Normalizer', 'vnlp')

        normalizer = Normalizer()
        output_text = normalizer.convert_numbers_to_words(input_text.split())

    elif task == 'deasciify':
        importlib.import_module('Normalizer', 'vnlp')

        normalizer = Normalizer()
        output_text = normalizer.deasciify(input_text.split())

    elif task == 'lower_case':
        importlib.import_module('Normalizer', 'vnlp')

        normalizer = Normalizer()
        output_text = normalizer.lower_case(input_text)

    elif task == 'remove_punctuations':
        importlib.import_module('Normalizer', 'vnlp')

        normalizer = Normalizer()
        output_text = normalizer.remove_punctuations(input_text)

    elif task == 'remove_accent_marks':
        importlib.import_module('Normalizer', 'vnlp')

        normalizer = Normalizer()
        output_text = normalizer.remove_accent_marks(input_text)

    elif task == 'drop_stop_words':
        importlib.import_module('StopwordRemover', 'vnlp')

        stopword_remover = StopwordRemover()
        output_text = stopword_remover.drop_stop_words(input_text.split())


    # TODO: check if returning prints the result in command line.
    return output_text



if __name__ == "__main__":
    main()