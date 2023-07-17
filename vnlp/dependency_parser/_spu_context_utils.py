import tensorflow as tf
import numpy as np

from ..utils import create_rnn_stacks, process_word_context

TOKEN_PIECE_MAX_LEN = 8  # 0.9995 quantile is 8 for 16k_vocab, 7 for 32k_vocab
SENTENCE_MAX_LEN = 40  # 0.998 quantile is 42


def create_spucontext_dp_model(
    TOKEN_PIECE_MAX_LEN,
    SENTENCE_MAX_LEN,
    VOCAB_SIZE,
    ARC_LABEL_VECTOR_LEN,
    WORD_EMBEDDING_DIM,
    WORD_EMBEDDING_MATRIX,
    NUM_RNN_UNITS,
    NUM_RNN_STACKS,
    FC_UNITS_MULTIPLIER,
    DROPOUT,
):
    # Current Word
    # WORD_RNN is common model for processing all word tokens
    word_rnn = tf.keras.models.Sequential(name="WORD_RNN")
    word_rnn.add(tf.keras.layers.InputLayer(input_shape=(TOKEN_PIECE_MAX_LEN)))
    word_rnn.add(
        tf.keras.layers.Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=WORD_EMBEDDING_DIM,
            embeddings_initializer=tf.keras.initializers.Constant(
                WORD_EMBEDDING_MATRIX
            ),
            trainable=False,
            name="WORD_EMBEDDING",
        )
    )
    word_rnn.add(create_rnn_stacks(NUM_RNN_STACKS, NUM_RNN_UNITS, DROPOUT))

    # Left Context Words
    left_context_rnn = tf.keras.models.Sequential(name="LEFT_CONTEXT_RNN")
    left_context_rnn.add(
        tf.keras.layers.InputLayer(
            input_shape=(SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN)
        )
    )
    left_context_rnn.add(tf.keras.layers.TimeDistributed(word_rnn))
    left_context_rnn.add(
        create_rnn_stacks(NUM_RNN_STACKS, NUM_RNN_UNITS, DROPOUT)
    )

    # Right Context Words
    right_context_rnn = tf.keras.models.Sequential(name="RIGHT_CONTEXT_RNN")
    right_context_rnn.add(
        tf.keras.layers.InputLayer(
            input_shape=(SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN)
        )
    )
    right_context_rnn.add(tf.keras.layers.TimeDistributed(word_rnn))
    right_context_rnn.add(
        create_rnn_stacks(
            NUM_RNN_STACKS, NUM_RNN_UNITS, DROPOUT, GO_BACKWARDS=True
        )
    )

    # Previously Processed (Left) Arc-Labels
    lc_arc_label_rnn = tf.keras.models.Sequential(name="PREV_ARC_LABEL_RNN")
    lc_arc_label_rnn.add(
        tf.keras.layers.InputLayer(
            input_shape=(SENTENCE_MAX_LEN, ARC_LABEL_VECTOR_LEN)
        )
    )
    lc_arc_label_rnn.add(
        create_rnn_stacks(NUM_RNN_STACKS, NUM_RNN_UNITS, DROPOUT)
    )

    # FC Layers
    current_left_right_concat = tf.keras.layers.Concatenate()(
        [
            word_rnn.output,
            left_context_rnn.output,
            right_context_rnn.output,
            lc_arc_label_rnn.output,
        ]
    )
    fc_layer_one = tf.keras.layers.Dense(
        NUM_RNN_UNITS * FC_UNITS_MULTIPLIER[0], activation="relu"
    )(current_left_right_concat)
    fc_layer_one = tf.keras.layers.Dropout(DROPOUT)(fc_layer_one)
    fc_layer_two = tf.keras.layers.Dense(
        NUM_RNN_UNITS * FC_UNITS_MULTIPLIER[1], activation="relu"
    )(fc_layer_one)
    fc_layer_two = tf.keras.layers.Dropout(DROPOUT)(fc_layer_two)
    arc_label_output = tf.keras.layers.Dense(
        ARC_LABEL_VECTOR_LEN, activation="sigmoid"
    )(fc_layer_two)

    dp_model = tf.keras.models.Model(
        inputs=[
            word_rnn.input,
            left_context_rnn.input,
            right_context_rnn.input,
            lc_arc_label_rnn.input,
        ],
        outputs=[arc_label_output],
    )

    return dp_model


def vectorize_arc_label(w, arcs, labels, sentence_max_len, tokenizer_label):
    arc = arcs[w]
    label = labels[w]
    arc_vector = tf.keras.utils.to_categorical(
        arc, num_classes=sentence_max_len + 1
    )
    label_vector = tf.keras.utils.to_categorical(
        label, num_classes=len(tokenizer_label.word_index) + 1
    )
    arc_label_vector = np.array(arc_vector.tolist() + label_vector.tolist())

    return arc_label_vector


def process_single_word_input(
    w,
    tokenized_sentence,
    spu_tokenizer_word,
    tokenizer_label,
    arc_label_vector_len,
    arcs,
    labels,
):
    (
        current_word_processed,
        left_context_words_processed,
        right_context_words_processed,
    ) = process_word_context(
        w,
        tokenized_sentence,
        spu_tokenizer_word,
        SENTENCE_MAX_LEN,
        TOKEN_PIECE_MAX_LEN,
    )

    left_context_arc_label_vectors = []
    # Pad Left Context Arc Label Vectors
    for _ in range(SENTENCE_MAX_LEN - w):
        left_context_arc_label_vectors.append(np.zeros(arc_label_vector_len))

    for w_ in range(w):
        left_context_arc_label_vectors.append(
            vectorize_arc_label(
                w_, arcs, labels, SENTENCE_MAX_LEN, tokenizer_label
            )
        )
    left_context_arc_label_vectors = np.array(left_context_arc_label_vectors)

    # Expand dims for sequence processing with batch_size = 1
    current_word_processed = np.expand_dims(current_word_processed, axis=0)
    left_context_words_processed = np.expand_dims(
        left_context_words_processed, axis=0
    )
    right_context_words_processed = np.expand_dims(
        right_context_words_processed, axis=0
    )
    left_context_arc_label_vectors = np.expand_dims(
        left_context_arc_label_vectors, axis=0
    )

    return (
        current_word_processed,
        left_context_words_processed,
        right_context_words_processed,
        left_context_arc_label_vectors,
    )
