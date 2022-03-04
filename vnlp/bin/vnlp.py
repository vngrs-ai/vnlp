import argparse
import os

# To suppress tensorflow warnings such as below:
"""
This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) 
to use the following CPU instructions in performance-critical operations: 
AVX2 FMA To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags. 
I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176]
None of the MLIR Optimization Passes are enabled (registered 2)
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from vnlp import (StemmerAnalyzer, NamedEntityRecognizer, DependencyParser, PoSTagger,
                  SentimentAnalyzer, SentenceSplitter, Normalizer, StopwordRemover)


def main():

    list_tasks = '\
        # List of available tasks:\n\
        stemming_morph_analysis\n\
        named_entity_recognition\n\
        dependency_parsing\n\
        part_of_speech_tagging\n\
        sentiment_analysis\n\
        split_sentences\n\
        correct_typos\n\
        convert_numbers_to_words\n\
        deasciify\n\
        lower_case\n\
        remove_punctuations\n\
        remove_accent_marks\n\
        drop_stop_words\n\
        '

    example_usage = '\
        $ vnlp --task stemming_morph_analysis --text "Üniversite sınavlarına canla başla çalışıyorlardı."\n\
        $ vnlp --task correct_typos --text "kassıtlı yaezım hatasssı ekliyorumm"\
        '

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, action = 'store', help = 'Task to be performed on text, e.g: sentiment_analysis')
    parser.add_argument('--text', type=str, action = 'store', help = 'Input text to be processed')
    parser.add_argument('--list_tasks', action = 'store_true', help = 'Lists available tasks/functions')
    parser.add_argument('--example_usage', action = 'store_true', help = 'Shows usage examples')
    args = parser.parse_args()

    task = args.task
    input_text = args.text

    if args.list_tasks:
        print(list_tasks)
        return

    if args.example_usage:
        print(example_usage)
        return

    elif task == 'stemming_morph_analysis':
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
        result = str(sentiment_analyzer.predict(input_text))

    elif task == 'split_sentences':
        sentence_splitter = SentenceSplitter()
        result = sentence_splitter.split_sentences(input_text)

    elif task == 'correct_typos':
        normalizer = Normalizer()
        result_as_list = normalizer.correct_typos(input_text.split())
        result = " ".join(result_as_list)

    elif task == 'convert_numbers_to_words':
        normalizer = Normalizer()
        result_as_list = normalizer.convert_numbers_to_words(input_text.split())
        result = " ".join(result_as_list)

    elif task == 'deasciify':
        result_as_list = Normalizer.deasciify(input_text.split())
        result = " ".join(result_as_list)

    elif task == 'lower_case':
        result = Normalizer.lower_case(input_text)

    elif task == 'remove_punctuations':
        result = Normalizer.remove_punctuations(input_text)

    elif task == 'remove_accent_marks':
        result = Normalizer.remove_accent_marks(input_text)

    elif task == 'drop_stop_words':
        stopword_remover = StopwordRemover()
        result_as_list = stopword_remover.drop_stop_words(input_text.split())
        result = " ".join(result_as_list)

    return result


if __name__ == "__main__":
    main()