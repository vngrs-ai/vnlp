import tensorflow as tf

def create_charner_model(char_vocab_size, embed_size, seq_len_max, num_rnn_stacks, rnn_dim, mlp_dim, num_classes, dropout):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim = char_vocab_size, output_dim = embed_size, input_length=seq_len_max))

    for _ in range(num_rnn_stacks):
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(rnn_dim, return_sequences = True)))
        model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(mlp_dim, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(num_classes, activation = 'softmax'))

    return model