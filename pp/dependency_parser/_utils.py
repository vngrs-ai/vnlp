import numpy as np
import tensorflow as tf

from ..stemmer_morph_analyzer import StemmerAnalyzer
from ..normalizer import Normalizer
from utils import WordPunctTokenize

normalizer = Normalizer()

sentence_max_len = 40
tag_max_len = 15

def create_dependency_parser_model(word_embedding_vocab_size, word_embedding_vector_size, word_embedding_matrix,
                                   sentence_max_len, tag_max_len, arc_label_vector_len, num_rnn_stacks, 
                                   tag_num_rnn_units, lc_num_rnn_units, lc_arc_label_num_rnn_units, rc_num_rnn_units,
                                   dropout, tag_embedding_weights = np.zeros((127, 32))):
    """
    Notes:
        - Add can be used instead of Concatenate but reduction of params is small
          and implementation cost is high with several requirements of layer shapes.
          Therefore it's not worth implementing
    """

    word_embedding_layer = tf.keras.layers.Embedding(input_dim = word_embedding_vocab_size + 1, output_dim = word_embedding_vector_size, 
                                              weights=[word_embedding_matrix], trainable = False)
    # ===============================================
    # CURRENT WORD
    # Word
    word_input = tf.keras.layers.Input(shape = (1))
    word_embedded_ = word_embedding_layer(word_input) # shape: (None, 1, embed_size)
    word_embedded = tf.keras.backend.squeeze(word_embedded_, axis = 1) # shape: (None, embed_size)

    # Tag
    # Transferred from learned weights of StemmerAnalyzer - Morphological Disambiguator
    #tag_embedding_weights = sa.model.layers[5].weights[0].numpy().copy() # shape: (127, 32)
    tag_vocab_size = tag_embedding_weights.shape[0]
    tag_embed_size = tag_embedding_weights.shape[1]

    tag_input = tf.keras.layers.Input(shape = (tag_max_len))
    tag_embedding_layer = tf.keras.layers.Embedding(input_dim = tag_vocab_size, output_dim = tag_embed_size,
                                                    weights = [tag_embedding_weights], trainable = False)
    tag_embedded = tag_embedding_layer(tag_input)

    tag_rnn = tf.keras.models.Sequential()
    for n in range(num_rnn_stacks - 1):
        tag_rnn.add(tf.keras.layers.GRU(tag_num_rnn_units, return_sequences = True))
        tag_rnn.add(tf.keras.layers.Dropout(dropout))
    tag_rnn.add(tf.keras.layers.GRU(tag_num_rnn_units))
    tag_rnn.add(tf.keras.layers.Dropout(dropout))
    tag_rnn_output = tag_rnn(tag_embedded)

    word_tag_concatanated = tf.keras.layers.Concatenate()([word_embedded, tag_rnn_output])

    # ===============================================
    # LEFT CONTEXT
    # Left Context Word
    left_context_word_input = tf.keras.layers.Input(shape = (sentence_max_len))
    left_context_word_embedded = word_embedding_layer(left_context_word_input)

    # Left Context Tags
    left_context_tag_input = tf.keras.layers.Input(shape = (sentence_max_len, tag_max_len))
    left_context_tag_embedded = tag_embedding_layer(left_context_tag_input)

    left_context_td_tag_rnn_output = tf.keras.layers.TimeDistributed(tag_rnn, input_shape = (sentence_max_len, tag_max_len, tag_embed_size))(left_context_tag_embedded)
    left_context_word_td_tags_concatenated = tf.keras.layers.Concatenate()([left_context_word_embedded, left_context_td_tag_rnn_output])

    # Left Context Word-Tag Final RNN (Left to Right)
    lc_rnn = tf.keras.models.Sequential()
    for n in range(num_rnn_stacks - 1):
        lc_rnn.add(tf.keras.layers.GRU(lc_num_rnn_units, return_sequences = True))
        lc_rnn.add(tf.keras.layers.Dropout(dropout))
    lc_rnn.add(tf.keras.layers.GRU(lc_num_rnn_units))
    lc_rnn.add(tf.keras.layers.Dropout(dropout))

    lc_output = lc_rnn(left_context_word_td_tags_concatenated)

    # Left-Context Arc-Label (previously processed tokens' arc-label results)
    lc_arc_label_input = tf.keras.layers.Input(shape = (sentence_max_len, arc_label_vector_len))
    lc_arc_label_rnn = tf.keras.models.Sequential()
    for n in range(num_rnn_stacks - 1):
        lc_arc_label_rnn.add(tf.keras.layers.GRU(lc_arc_label_num_rnn_units, return_sequences = True))
        lc_arc_label_rnn.add(tf.keras.layers.Dropout(dropout))
    lc_arc_label_rnn.add(tf.keras.layers.GRU(lc_arc_label_num_rnn_units))
    lc_arc_label_rnn.add(tf.keras.layers.Dropout(dropout))
    lc_arc_label_rnn_output = lc_arc_label_rnn(lc_arc_label_input)

    # ===============================================
    # RIGHT CONTEXT
    # Right Context Word
    right_context_word_input = tf.keras.layers.Input(shape = (sentence_max_len))
    right_context_word_embedded = word_embedding_layer(right_context_word_input)

    # Right Context Tags
    right_context_tag_input = tf.keras.layers.Input(shape = (sentence_max_len, tag_max_len))
    right_context_tag_embedded = tag_embedding_layer(right_context_tag_input)

    right_context_td_tag_rnn_output = tf.keras.layers.TimeDistributed(tag_rnn, input_shape = (sentence_max_len, tag_max_len, tag_embed_size))(right_context_tag_embedded)
    right_context_word_td_tags_concatenated = tf.keras.layers.Concatenate()([right_context_word_embedded, right_context_td_tag_rnn_output])

    # Right Context Word-Tag Final RNN (Right to Left)
    rc_rnn = tf.keras.models.Sequential()
    for n in range(num_rnn_stacks - 1):
        rc_rnn.add(tf.keras.layers.GRU(rc_num_rnn_units, return_sequences = True, go_backwards = True)) # Right to Left!
        rc_rnn.add(tf.keras.layers.Dropout(dropout))
    rc_rnn.add(tf.keras.layers.GRU(rc_num_rnn_units, go_backwards = True))
    rc_rnn.add(tf.keras.layers.Dropout(dropout))

    rc_output = rc_rnn(right_context_word_td_tags_concatenated)
    current_left_right_concat = tf.keras.layers.Concatenate()([word_tag_concatanated, lc_output, lc_arc_label_rnn_output, rc_output])
    
    # ===============================================
    # FC LAYERS
    fc_layer_one = tf.keras.layers.Dense(tag_num_rnn_units * 8, activation = 'relu')(current_left_right_concat)
    fc_layer_one = tf.keras.layers.Dropout(dropout)(fc_layer_one)
    fc_layer_two = tf.keras.layers.Dense(tag_num_rnn_units * 4, activation = 'relu')(current_left_right_concat)
    fc_layer_two = tf.keras.layers.Dropout(dropout)(fc_layer_two)
    arc_label_output = tf.keras.layers.Dense(arc_label_vector_len, activation = 'sigmoid')(fc_layer_two)

    model = tf.keras.models.Model(inputs = [word_input, tag_input, left_context_word_input, left_context_tag_input, 
                                        lc_arc_label_input, right_context_word_input, right_context_tag_input],
                              outputs = [arc_label_output])

    return model


def preprocess_word(word):
    word = word.replace('’', "'")
    word = normalizer.lower_case(word)
    return word


def fit_label_tokenizer(files):
    labels = []
    for file in files:
        with open(file, encoding = 'utf-8') as f:
            for line in f:
                line = line.strip().replace(u'\u200b', '')
                split_line = line.split("\t")
                    
                if (line != "") and (line[0] != "#"):
                    if "-" in split_line[0]:
                        continue
                    else:
                        label = split_line[-3]
                        labels.append([label])

    # Label tokenizer
    tokenizer_label = tf.keras.preprocessing.text.Tokenizer(lower = False, filters = None)
    tokenizer_label.fit_on_texts(labels)

    return tokenizer_label


def load_triplets(files):
    data = []
    for file in files:
        data_ = []
        with open(file, encoding = 'utf-8') as f:
            sentence = []
            shift_by = []
            for line in f:
                line = line.strip().replace(u'\u200b', '')
                split_line = line.split("\t")
                    
                if (line != "") and (line[0] != "#"):
                    if "-" in split_line[0]:
                        continue
                    else:
                        token = split_line[1]
                        token = preprocess_word(token)
                        arc = int(split_line[-4])
                        label = split_line[-3]
                        word_punct_tokenized_tokens = WordPunctTokenize(token)
                        # WordPunctTokenization causes shift in arc values so shift_by process is required to fix it back
                        if len(word_punct_tokenized_tokens) > 1:
                            shift_by.append((len(sentence) + 1, len(word_punct_tokenized_tokens) - 1)) # len(sentence) is the number of tokens in the sentence that are processed so far
                        for token in word_punct_tokenized_tokens:
                            sentence.append([token, arc, label])
                else:
                    if sentence:
                        # Correcting shift caused by WordPunctTokenization
                        for duo in shift_by:
                            arc_anchor = duo[0]
                            shift = duo[1]
                            for t, triplet in enumerate(sentence):
                                arc = triplet[1]
                                if arc > arc_anchor:
                                    arc += shift
                                    sentence[t][1] = arc
                        # Done with the sentence now, and adding
                        data_.append(sentence)
                    sentence = []
                    shift_by = []
        data += data_
    return data

def process_single_word_input(t, sentence_analysis_result, 
                              sentence_max_len, tag_max_len, arc_label_vector_len, 
                              tokenizer_word, tokenizer_tag, tokenizer_label, arcs, labels):
    
    num_tokens_in_sentence = len(sentence_analysis_result)
    analysis_result = sentence_analysis_result[t]
    word_and_tags = analysis_result.split('+')
    
    word = word_and_tags[0]
    word = tokenizer_word.texts_to_sequences([[word]]) # this wont have padding
    
    tags = word_and_tags[1:]
    tags = tokenizer_tag.texts_to_sequences([tags])
    tags = tf.keras.preprocessing.sequence.pad_sequences(tags, 
                                                     maxlen = tag_max_len, 
                                                     padding = 'pre')[0]
    
    # ===============================================
    # Left Context
    if t == 0: # there's no left context yet
        left_context_words = np.zeros(sentence_max_len)
        left_context_tags = np.zeros((sentence_max_len, tag_max_len))
        left_context_arc_label_vectors = np.zeros((sentence_max_len, arc_label_vector_len))
    else:
        left_context_words = [analysis_result.split('+')[0] for analysis_result in sentence_analysis_result[:t]]
        left_context_words = tokenizer_word.texts_to_sequences([left_context_words])
        left_context_words = tf.keras.preprocessing.sequence.pad_sequences(left_context_words, 
                                                                    maxlen = sentence_max_len, 
                                                                    padding = 'pre')[0]

        left_context_tags = [analysis_result.split('+')[1:] for analysis_result in sentence_analysis_result[:t]]
        left_context_tags = tokenizer_tag.texts_to_sequences(left_context_tags)
        left_context_tags = tf.keras.preprocessing.sequence.pad_sequences(left_context_tags, 
                                                                    maxlen = tag_max_len, 
                                                                    padding = 'pre')

        left_context_arcs = arcs[:t]
        left_context_labels = labels[:t]

        left_context_arc_label_vectors = []
        for left_context_arc_, left_context_label_ in zip(left_context_arcs, left_context_labels):
            left_context_one_hot_arc = tf.keras.utils.to_categorical(left_context_arc_, num_classes = sentence_max_len)
            left_context_one_hot_label = tf.keras.utils.to_categorical(left_context_label_, num_classes = len(tokenizer_label.word_index) + 1)
            left_context_arc_label_vector = left_context_one_hot_arc.tolist() + left_context_one_hot_label.tolist()
            left_context_arc_label_vectors.append(left_context_arc_label_vector)
        
        # 2D PRE-padding for left context tags and left_context_arc_vectors
        # final shapes will be: 
        # left_context_tags: (sentence_max_len, tag_max_len)
        # left_context_arc_vectors: (sentence_max_len, sentence_max_len)
        left_context_tags_ = []
        left_context_tags = left_context_tags.tolist()

        left_context_arc_label_vectors_ = []
        for _ in range(max(0, sentence_max_len - len(left_context_tags))):
            left_context_tags_.append(np.zeros(tag_max_len))
            left_context_arc_label_vectors_.append(np.zeros(arc_label_vector_len))
        left_context_tags = np.array(left_context_tags_ + left_context_tags)
        left_context_arc_label_vectors = np.array(left_context_arc_label_vectors_ + left_context_arc_label_vectors)
    # ===============================================
    # Right Context
    if t == (num_tokens_in_sentence -1): # there's no right context for last token
        right_context_words = np.zeros(sentence_max_len)
        right_context_tags = np.zeros((sentence_max_len, tag_max_len))
    else:
        right_context_words = [analysis_result.split('+')[0] for analysis_result in sentence_analysis_result[t+1:]]
        right_context_words = tokenizer_word.texts_to_sequences([right_context_words])
        right_context_words = tf.keras.preprocessing.sequence.pad_sequences(right_context_words, 
                                                                    maxlen = sentence_max_len, 
                                                                    padding = 'post')[0]

        right_context_tags = [analysis_result.split('+')[1:] for analysis_result in sentence_analysis_result[t+1:]]
        right_context_tags = tokenizer_tag.texts_to_sequences(right_context_tags)
        right_context_tags = tf.keras.preprocessing.sequence.pad_sequences(right_context_tags, 
                                                                    maxlen = tag_max_len, 
                                                                    padding = 'pre') # last dimension padding is pre

        # 2D POST-padding for right context tags
        # final shape will be: (sentence_max_len, tag_max_len)
        right_context_tags = right_context_tags.tolist()
        for _ in range(max(0, sentence_max_len - len(right_context_tags))):
            right_context_tags.append(np.zeros(tag_max_len))
        right_context_tags = np.array(right_context_tags)

    word = np.array(word[0]).reshape(1, 1)
    tags = tags.reshape(1, tag_max_len)
    left_context_words = left_context_words.reshape(1, sentence_max_len)
    left_context_tags = left_context_tags.reshape(1, sentence_max_len, tag_max_len)
    left_context_arc_label_vectors = left_context_arc_label_vectors.reshape(1, sentence_max_len, arc_label_vector_len)
    right_context_words = right_context_words.reshape(1, sentence_max_len)
    right_context_tags = right_context_tags.reshape(1, sentence_max_len, tag_max_len)
    
    return (word, tags, left_context_words, left_context_tags, left_context_arc_label_vectors,
            right_context_words, right_context_tags)


def process_single_sentence(sentence, sentence_max_len, tag_max_len, arc_label_vector_len, 
                            tokenizer_word, tokenizer_tag, tokenizer_label, sa):
    """
    sentence is list of List of triplets: [[token, arc, label]] loaded by load_triplets()
    ==========================================================================================
    word					            (1)						                    #nopadding
    tags					            (tag_max_len)					            #pre
    arc_label_vector			        (arc_label_vector_len)				        #nopadding

    left_context_words			        (sentence_max_len)				            #pre
    left_context_tags			        (sentence_max_len, tag_max_len)			    #pre, pre
    left_context_arc_label_vectors		(sentence_max_len, arc_label_vector_len)	#pre

    right_context_words			        (sentence_max_len)				            #post
    right_context_tags			        (sentence_max_len, tag_max_len)			    #post, pre
    """

    raw_sentence_string_form = " ".join([triplet[0] for triplet in sentence])
    num_tokens_in_sentence = len(sentence)
    sentence_analysis_result = sa.predict(raw_sentence_string_form)
    sentence_analysis_result = [sentence_analysis.replace('^', '+') for sentence_analysis in sentence_analysis_result]

    arcs = [triplet[1] for triplet in sentence]
    labels = [triplet[2] for triplet in sentence]
    labels = tokenizer_label.texts_to_sequences([labels])[0]

    if not len(sentence_analysis_result) == num_tokens_in_sentence:
        raise Exception(sentence, "Length of sentence and sentence_analysis_result don't match")

    batch_of_words_ = []
    batch_of_tags_ = []
    batch_of_arc_label_vectors_ = []

    batch_of_left_context_words_ = []
    batch_of_left_context_tags_ = []
    batch_of_left_context_arc_label_vectors_ = []

    batch_of_right_context_words_ = []
    batch_of_right_context_tags_ = []

    for t in range(num_tokens_in_sentence):
        X = process_single_word_input(t, sentence_analysis_result, 
                                      sentence_max_len, tag_max_len, arc_label_vector_len, 
                                      tokenizer_word, tokenizer_tag, tokenizer_label, arcs, labels)
        # X
        word = X[0][0]
        tags = X[1][0]
        left_context_words = X[2][0]
        left_context_tags = X[3][0]
        left_context_arc_label_vectors = X[4][0]
        right_context_words = X[5][0]
        right_context_tags = X[6][0]

        # y
        arc = arcs[t]
        label = labels[t]
        arc_vector = tf.keras.utils.to_categorical(arc, num_classes = sentence_max_len)
        label_vector = tf.keras.utils.to_categorical(label, num_classes = len(tokenizer_label.word_index) + 1)
        arc_label_vector = np.array(arc_vector.tolist() + label_vector.tolist())
        
        # Appending
        batch_of_words_.append(word)
        batch_of_tags_.append(tags)
        batch_of_arc_label_vectors_.append(arc_label_vector)

        batch_of_left_context_words_.append(left_context_words)
        batch_of_left_context_tags_.append(left_context_tags)
        batch_of_left_context_arc_label_vectors_.append(left_context_arc_label_vectors)

        batch_of_right_context_words_.append(right_context_words)
        batch_of_right_context_tags_.append(right_context_tags)

    return (batch_of_words_, batch_of_tags_, batch_of_arc_label_vectors_, batch_of_left_context_words_,
            batch_of_left_context_tags_, batch_of_left_context_arc_label_vectors_, batch_of_right_context_words_,
            batch_of_right_context_tags_)


def data_generator(files, tokenizer_word, tokenizer_tag, tokenizer_label, 
                   sentence_max_len, tag_max_len, arc_label_vector_len, batch_size):
    
    data = load_triplets(files)

    # We need morphological tags to train
    with tf.device('/cpu:0'):
        sa = StemmerAnalyzer()

    while True:
        batch_of_words = []
        batch_of_tags = []
        batch_of_arc_label_vectors = []

        batch_of_left_context_words = []
        batch_of_left_context_tags = []
        batch_of_left_context_arc_label_vectors = []

        batch_of_right_context_words = []
        batch_of_right_context_tags = []
        while len(batch_of_words) < batch_size:
            rand_sentence_idx = np.random.randint(0, len(data), 1)[0]
            sentence = data[rand_sentence_idx]
            # I won't use sentences longer than sentence_max_len for training
            # because first token can depend on last token and truncating and breaking
            # the tree structure can hurt training.
            # Also there's no way to output arc greater than sentence_max_len
            # so I will skip them
            if len(sentence) > sentence_max_len:
                continue
            else:
                (batch_of_words_, batch_of_tags_, batch_of_arc_label_vectors_, batch_of_left_context_words_,
                batch_of_left_context_tags_, batch_of_left_context_arc_label_vectors_, batch_of_right_context_words_,
                batch_of_right_context_tags_) = process_single_sentence(sentence, sentence_max_len, 
                                                                        tag_max_len, arc_label_vector_len, 
                                                                        tokenizer_word, tokenizer_tag, 
                                                                        tokenizer_label, sa)
                batch_of_words += batch_of_words_
                batch_of_tags += batch_of_tags_
                batch_of_arc_label_vectors += batch_of_arc_label_vectors_

                batch_of_left_context_words += batch_of_left_context_words_
                batch_of_left_context_tags += batch_of_left_context_tags_
                batch_of_left_context_arc_label_vectors += batch_of_left_context_arc_label_vectors_

                batch_of_right_context_words += batch_of_right_context_words_
                batch_of_right_context_tags += batch_of_right_context_tags_

        # Converting to NumPy array and truncating in case it is larger than batch_size
        batch_of_words = np.array(batch_of_words)[:batch_size]
        batch_of_tags = np.array(batch_of_tags)[:batch_size]
        batch_of_arc_label_vectors = np.array(batch_of_arc_label_vectors)[:batch_size]

        batch_of_left_context_words = np.array(batch_of_left_context_words)[:batch_size]
        batch_of_left_context_tags = np.array(batch_of_left_context_tags)[:batch_size]
        batch_of_left_context_arc_label_vectors = np.array(batch_of_left_context_arc_label_vectors)[:batch_size]

        batch_of_right_context_words = np.array(batch_of_right_context_words)[:batch_size]
        batch_of_right_context_tags = np.array(batch_of_right_context_tags)[:batch_size]

        X_train = (batch_of_words, batch_of_tags, batch_of_left_context_words, batch_of_left_context_tags,
                batch_of_left_context_arc_label_vectors, batch_of_right_context_words, batch_of_right_context_tags)
        y_train = batch_of_arc_label_vectors

        yield (X_train, y_train)

def UAS(y_true, y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    
    y_true_arc = np.argmax(y_true[:, :sentence_max_len], axis = -1)
    y_pred_arc = np.argmax(y_pred[:, :sentence_max_len], axis = -1)
    
    tp_count = 0
    total_count = y_true.shape[0]
    for idx in range(total_count):
        if y_true_arc[idx] == y_pred_arc[idx]:
            tp_count += 1
    
    return tp_count/total_count

def LAS(y_true, y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    
    y_true_arc = np.argmax(y_true[:, :sentence_max_len], axis = -1)
    y_pred_arc = np.argmax(y_pred[:, :sentence_max_len], axis = -1)
    
    y_true_label = np.argmax(y_true[:, sentence_max_len:], axis = -1)
    y_pred_label = np.argmax(y_pred[:, sentence_max_len:], axis = -1)
    
    tp_count = 0
    total_count = y_true.shape[0]
    for idx in range(total_count):
        if (y_true_arc[idx] == y_pred_arc[idx]) & (y_true_label[idx] == y_pred_label[idx]):
            tp_count += 1
    
    return tp_count/total_count