import tensorflow as tf


def create_sentiment_analysis_model(word_embedding_vocab_size, word_embedding_vector_size, word_embedding_matrix,
                                    num_rnn_units, num_rnn_stacks, dropout):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim = word_embedding_vocab_size, output_dim = word_embedding_vector_size, 
                                        embeddings_initializer=tf.keras.initializers.Constant(word_embedding_matrix), trainable = False))
    for n in range(num_rnn_stacks - 1):
        model.add(tf.keras.layers.GRU(num_rnn_units, return_sequences = True))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.GRU(num_rnn_units, return_sequences = False))
    model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(num_rnn_units // 8, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

    return model


def lower_case(text):
    turkish_lowercase_dict = {"İ": "i", "I": "ı", "Ğ": "ğ", "Ü": "ü", "Ö": "ö", "Ş": "ş", "Ç": "ç"}
    for k, v in turkish_lowercase_dict.items():
        text = text.replace(k, v)

    return text.lower()

def convert_numbers_to_zero(text_: str):
    text_ = str(text_) # in case input is not string
    text = ""
    for char in text_:
        if char.isnumeric():
            text += "0"
        else:
            text += char
    return text

def preprocess_text(text):
    text = text.replace('’', "'")
    text = lower_case(text) # TODO: use normalizer.lower_case() when done with development
    text = convert_numbers_to_zero(text)
    return text