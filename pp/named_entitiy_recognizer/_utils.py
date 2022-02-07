import tensorflow as tf

class CustomNonPaddingTokenLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_ner_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        loss = loss_fn(y_true, y_pred)
        mask = tf.cast((y_true > 0), dtype=tf.float32)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def create_ner_model(char_vocab_size, embed_size, seq_len_max, num_rnn_stacks, rnn_dim, mlp_dim, num_classes, dropout):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim = char_vocab_size, output_dim = embed_size, input_length=seq_len_max))

    for _ in range(num_rnn_stacks):
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(rnn_dim, return_sequences = True)))
        model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(mlp_dim, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(num_classes, activation = 'softmax'))

    return model