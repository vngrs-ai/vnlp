import tensorflow as tf
import numpy as np


def create_spucbigru_sentiment_model(
    TEXT_MAX_LEN,
    VOCAB_SIZE,
    WORD_EMBEDDING_DIM,
    WORD_EMBEDDING_MATRIX,
    NUM_RNN_UNITS,
    NUM_RNN_STACKS,
    DROPOUT,
):
    inp_layer = tf.keras.layers.Input(shape=(TEXT_MAX_LEN))
    model = tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=WORD_EMBEDDING_DIM,
        embeddings_initializer=tf.keras.initializers.Constant(
            WORD_EMBEDDING_MATRIX
        ),
        trainable=False,
    )(inp_layer)

    for n in range(NUM_RNN_STACKS - 1):
        model = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                NUM_RNN_UNITS, dropout=DROPOUT, return_sequences=True
            )
        )(model)

    model = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            NUM_RNN_UNITS, dropout=DROPOUT, return_sequences=True
        )
    )(model)
    model = tf.keras.layers.GlobalAveragePooling1D()(model)
    model = tf.keras.layers.Dropout(DROPOUT)(model)
    model = tf.keras.layers.Dense(NUM_RNN_UNITS // 8, activation="relu")(model)
    model = tf.keras.layers.Dropout(DROPOUT)(model)
    out_layer = tf.keras.layers.Dense(1, activation="sigmoid")(model)

    model = tf.keras.models.Model(inputs=inp_layer, outputs=out_layer)

    return model


def process_text_input(text, tokenizer, TEXT_MAX_LEN):
    integer_tokenized_text = tokenizer.encode_as_ids(text)

    # Padding.
    # tf.keras.utils.pad_sequences dictates that data must be truncated, either "pre" or "post",
    # whereas I don't want any truncation since I use recursive splitting.
    # Hence, padding is carried out manually here.
    len_diff = TEXT_MAX_LEN - len(integer_tokenized_text)
    if len_diff > 0:
        integer_tokenized_text = [0] * len_diff + integer_tokenized_text

    padded_text = np.array([integer_tokenized_text])
    return padded_text
