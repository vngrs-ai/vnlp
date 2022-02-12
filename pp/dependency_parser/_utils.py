import numpy as np
import tensorflow as tf

from ..stemmer_morph_analyzer import StemmerAnalyzer
from ..normalizer import Normalizer
from tokenizer import TreebankWordTokenize

normalizer = Normalizer()

sentence_max_len = 40
tag_max_len = 15

num_unq_labels = 48 #len(tokenizer_label.word_index)
num_unq_pos_tags = 17 #len(tokenizer_pos.word_index)

def create_dependency_parser_model(word_embedding_vocab_size, word_embedding_vector_size, word_embedding_matrix,
                                   pos_vocab_size, pos_embedding_vector_size,
                                   sentence_max_len, tag_max_len, arc_label_vector_len, num_rnn_stacks, 
                                   tag_num_rnn_units, lc_num_rnn_units, lc_arc_label_num_rnn_units, rc_num_rnn_units,
                                   dropout, tag_embedding_matrix):
    """
    Notes:
        - Add can be used instead of Concatenate but reduction of params is small
          and implementation cost is high with several requirements of layer shapes.
          Therefore it's not worth implementing
    """

    word_embedding_layer = tf.keras.layers.Embedding(input_dim = word_embedding_vocab_size, output_dim = word_embedding_vector_size, 
                                              weights=[word_embedding_matrix], trainable = False)
    pos_embedding_layer = tf.keras.layers.Embedding(input_dim = pos_vocab_size + 1, output_dim = pos_embedding_vector_size)
    
    # ===============================================
    # CURRENT WORD
    # Word
    word_input = tf.keras.layers.Input(shape = (1))
    word_embedded_ = word_embedding_layer(word_input) # shape: (None, 1, embed_size)
    word_embedded = tf.keras.backend.squeeze(word_embedded_, axis = 1) # shape: (None, embed_size)

    # Tag
    # Transferred from learned weights of StemmerAnalyzer - Morphological Disambiguator
    #tag_embedding_matrix = sa.model.layers[5].weights[0].numpy().copy() # shape: (127, 32)
    tag_vocab_size = tag_embedding_matrix.shape[0]
    tag_embed_size = tag_embedding_matrix.shape[1]

    tag_input = tf.keras.layers.Input(shape = (tag_max_len))
    tag_embedding_layer = tf.keras.layers.Embedding(input_dim = tag_vocab_size, output_dim = tag_embed_size,
                                                    weights = [tag_embedding_matrix], trainable = False)
    tag_embedded = tag_embedding_layer(tag_input)

    tag_rnn = tf.keras.models.Sequential()
    for n in range(num_rnn_stacks - 1):
        tag_rnn.add(tf.keras.layers.GRU(tag_num_rnn_units, return_sequences = True))
        tag_rnn.add(tf.keras.layers.Dropout(dropout))
    tag_rnn.add(tf.keras.layers.GRU(tag_num_rnn_units))
    tag_rnn.add(tf.keras.layers.Dropout(dropout))
    tag_rnn_output = tag_rnn(tag_embedded)
    
    # Pos
    pos_input = tf.keras.layers.Input(shape = (1))
    pos_embedded_ = pos_embedding_layer(pos_input)
    pos_embedded = tf.keras.backend.squeeze(pos_embedded_, axis = 1)

    word_tag_pos_concatanated = tf.keras.layers.Concatenate()([word_embedded, tag_rnn_output, pos_embedded])

    # ===============================================
    # LEFT CONTEXT
    # Left Context Word
    left_context_word_input = tf.keras.layers.Input(shape = (sentence_max_len))
    left_context_word_embedded = word_embedding_layer(left_context_word_input)

    # Left Context Tags
    left_context_tag_input = tf.keras.layers.Input(shape = (sentence_max_len, tag_max_len))
    left_context_tag_embedded = tag_embedding_layer(left_context_tag_input)

    left_context_td_tag_rnn_output = tf.keras.layers.TimeDistributed(tag_rnn, input_shape = (sentence_max_len, tag_max_len, tag_embed_size))(left_context_tag_embedded)

    # Left Context POS Tags
    left_context_pos_input = tf.keras.layers.Input(shape = (sentence_max_len))
    left_context_pos_embedded = pos_embedding_layer(left_context_pos_input)
    
    left_context_word_td_tags_pos_concatenated = tf.keras.layers.Concatenate()([left_context_word_embedded, left_context_td_tag_rnn_output
                                                                               ,left_context_pos_embedded])
    
    # Left Context Word-Tag-POS Final RNN (Left to Right)
    lc_rnn = tf.keras.models.Sequential()
    for n in range(num_rnn_stacks - 1):
        lc_rnn.add(tf.keras.layers.GRU(lc_num_rnn_units, return_sequences = True))
        lc_rnn.add(tf.keras.layers.Dropout(dropout))
    lc_rnn.add(tf.keras.layers.GRU(lc_num_rnn_units))
    lc_rnn.add(tf.keras.layers.Dropout(dropout))

    lc_output = lc_rnn(left_context_word_td_tags_pos_concatenated)

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
    
    # Right Context POS Tags
    right_context_pos_input = tf.keras.layers.Input(shape = (sentence_max_len))
    right_context_pos_embedded = pos_embedding_layer(right_context_pos_input)

    right_context_td_tag_rnn_output = tf.keras.layers.TimeDistributed(tag_rnn, input_shape = (sentence_max_len, tag_max_len, tag_embed_size))(right_context_tag_embedded)
    
    right_context_word_td_tags_pos_concatenated = tf.keras.layers.Concatenate()([right_context_word_embedded, right_context_td_tag_rnn_output, right_context_pos_embedded])

    # Right Context Word-Tag-POS Final RNN (Right to Left)
    rc_rnn = tf.keras.models.Sequential()
    for n in range(num_rnn_stacks - 1):
        rc_rnn.add(tf.keras.layers.GRU(rc_num_rnn_units, return_sequences = True, go_backwards = True)) # Right to Left!
        rc_rnn.add(tf.keras.layers.Dropout(dropout))
    rc_rnn.add(tf.keras.layers.GRU(rc_num_rnn_units, go_backwards = True))
    rc_rnn.add(tf.keras.layers.Dropout(dropout))

    rc_output = rc_rnn(right_context_word_td_tags_pos_concatenated)
    current_left_right_concat = tf.keras.layers.Concatenate()([word_tag_pos_concatanated, lc_output, lc_arc_label_rnn_output, rc_output])
    
    # ===============================================
    # FC LAYERS
    fc_layer_one = tf.keras.layers.Dense(tag_num_rnn_units * 8, activation = 'relu')(current_left_right_concat)
    fc_layer_one = tf.keras.layers.Dropout(dropout)(fc_layer_one)
    fc_layer_two = tf.keras.layers.Dense(tag_num_rnn_units * 4, activation = 'relu')(fc_layer_one)
    fc_layer_two = tf.keras.layers.Dropout(dropout)(fc_layer_two)
    arc_label_output = tf.keras.layers.Dense(arc_label_vector_len, activation = 'sigmoid')(fc_layer_two)

    model = tf.keras.models.Model(inputs = [word_input, tag_input, pos_input, left_context_word_input, left_context_tag_input, 
                                            left_context_pos_input, lc_arc_label_input, right_context_word_input, right_context_tag_input,
                                            right_context_pos_input],
                                            outputs = [arc_label_output])

    return model


def preprocess_word(word):
    word = word.replace('’', "'")
    word = normalizer.lower_case(word)
    word = convert_numbers_to_zero(word)
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

def fit_pos_tokenizer(files):
    pos_tags = []
    for file in files:
        with open(file, encoding = 'utf-8') as f:
            for line in f:
                line = line.strip().replace(u'\u200b', '')
                split_line = line.split("\t")
                    
                if (line != "") and (line[0] != "#"):
                    if "-" in split_line[0]:
                        continue
                    else:
                        pos = split_line[3]
                        pos_tags.append([pos])

    # Label tokenizer
    tokenizer_pos = tf.keras.preprocessing.text.Tokenizer(lower = False, filters = None)
    tokenizer_pos.fit_on_texts(pos_tags)

    return tokenizer_pos


def load_quadruplets(files, sentence_max_len):
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
                        pos = split_line[3]

                        word_punct_tokenized_tokens = TreebankWordTokenize(token)
                        # In case Tokenization causes shift in arc values, shift_by process is required to fix it back
                        if len(word_punct_tokenized_tokens) > 1:
                            shift_by.append((len(sentence) + 1, len(word_punct_tokenized_tokens) - 1)) # len(sentence) is the number of tokens in the sentence that are processed so far
                        for token in word_punct_tokenized_tokens:
                            sentence.append([token, arc, label, pos])
                else:
                    if sentence:
                        # Correcting shift caused by WordPunctTokenization
                        for duo in shift_by:
                            arc_anchor = duo[0]
                            shift = duo[1]
                            for t, quadruplet in enumerate(sentence):
                                arc = quadruplet[1]
                                if arc > arc_anchor:
                                    arc += shift
                                    sentence[t][1] = arc
                        # Done with the sentence now, and adding
                        # don't use sentences longer than sentence_max_len
                        if not (len(sentence) > sentence_max_len):
                            data_.append(sentence)
                    sentence = []
                    shift_by = []
        data += data_
    return data

def process_single_word_input(t, whole_words_in_sentence, sentence_analysis_result, 
                              sentence_max_len, tag_max_len, arc_label_vector_len, 
                              tokenizer_word, tokenizer_tag, tokenizer_label, tokenizer_pos,
                              arcs, labels, pos_tags, word_form):
    
    num_tokens_in_sentence = len(sentence_analysis_result)
    analysis_result = sentence_analysis_result[t]
    word_and_tags = analysis_result.split('+')
    
    stems = [analysis_result.split('+')[0] for analysis_result in sentence_analysis_result]
    
    # In case a word does not exist in tokenizer_word dict,
    # its stem is used instead of an OOV token.
    # vice versa for stem case: in case a stem is not recognized
    # whole word is used
    if word_form == 'stem':
        words = []
        for stem, whole_word in zip(stems, whole_words_in_sentence):
            # use whole_word only if stem is not in vocab, but whole_word IS in vocab
            # otherwise use stem anyway
            if ((not (stem in tokenizer_word.word_index)) and (whole_word in tokenizer_word.word_index)):
                words.append(whole_word)
            else:
                words.append(stem)

    elif word_form == 'whole':
        words = []
        for whole_word, stem in zip(whole_words_in_sentence, stems):
            # use stem only if whole_word is not in vocab, but stem IS in vocab
            # otherwise use whole_word anyway (can benefit FastText or similar morphological embedding in future)
            if ((not (whole_word in tokenizer_word.word_index)) and (stem in tokenizer_word.word_index)):
                words.append(stem)
            else:
                words.append(whole_word)

    words = [preprocess_word(word) for word in words]
                
    word = words[t]
    word = tokenizer_word.texts_to_sequences([[word]]) # this wont have padding
    
    tags = word_and_tags[1:]
    tags = tokenizer_tag.texts_to_sequences([tags])
    tags = tf.keras.preprocessing.sequence.pad_sequences(tags, 
                                                     maxlen = tag_max_len, 
                                                     padding = 'pre')[0]

    pos_tags = tokenizer_pos.texts_to_sequences([pos_tags])[0]
    pos = pos_tags[t]
    
    # ===============================================
    # Left Context
    if t == 0: # there's no left context yet
        left_context_words = np.zeros(sentence_max_len)
        left_context_tags = np.zeros((sentence_max_len, tag_max_len))
        left_context_pos_tags = np.zeros(sentence_max_len)
        left_context_arc_label_vectors = np.zeros((sentence_max_len, arc_label_vector_len))
    else:
        left_context_words = words[:t]
        left_context_words = tokenizer_word.texts_to_sequences([left_context_words])
        left_context_words = tf.keras.preprocessing.sequence.pad_sequences(left_context_words, 
                                                                           maxlen = sentence_max_len,
                                                                           truncating = 'pre',
                                                                           padding = 'pre')[0]

        left_context_tags = [analysis_result.split('+')[1:] for analysis_result in sentence_analysis_result[:t]]
        left_context_tags = tokenizer_tag.texts_to_sequences(left_context_tags)
        left_context_tags = tf.keras.preprocessing.sequence.pad_sequences(left_context_tags, 
                                                                    maxlen = tag_max_len, 
                                                                    padding = 'pre')
        
        left_context_pos_tags = pos_tags[:t]
        left_context_pos_tags = tf.keras.preprocessing.sequence.pad_sequences([left_context_pos_tags], 
                                                                    maxlen = sentence_max_len, 
                                                                    padding = 'pre')

        left_context_arcs = arcs[:t]
        left_context_labels = labels[:t]
        

        left_context_arc_label_vectors = []
        for left_context_arc_, left_context_label_ in zip(left_context_arcs, left_context_labels):
            left_context_one_hot_arc = tf.keras.utils.to_categorical(left_context_arc_, num_classes = sentence_max_len + 1)
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
        right_context_pos_tags = np.zeros(sentence_max_len)
    else:
        right_context_words = words[t+1:]
        right_context_words = tokenizer_word.texts_to_sequences([right_context_words])
        right_context_words = tf.keras.preprocessing.sequence.pad_sequences(right_context_words, 
                                                                            maxlen = sentence_max_len,
                                                                            truncating = 'post',
                                                                            padding = 'post')[0]

        right_context_tags = [analysis_result.split('+')[1:] for analysis_result in sentence_analysis_result[t+1:]]
        right_context_tags = tokenizer_tag.texts_to_sequences(right_context_tags)
        right_context_tags = tf.keras.preprocessing.sequence.pad_sequences(right_context_tags, 
                                                                    maxlen = tag_max_len, 
                                                                    padding = 'pre') # last dimension padding is pre
        right_context_pos_tags = pos_tags[t+1:]
        right_context_pos_tags = tf.keras.preprocessing.sequence.pad_sequences([right_context_pos_tags], 
                                                                               maxlen = sentence_max_len, 
                                                                               padding = 'post')[0]

        # 2D POST-padding for right context tags
        # final shape will be: (sentence_max_len, tag_max_len)
        right_context_tags = right_context_tags.tolist()
        for _ in range(max(0, sentence_max_len - len(right_context_tags))):
            right_context_tags.append(np.zeros(tag_max_len))
        right_context_tags = np.array(right_context_tags)

    word = np.array(word[0]).reshape(1, 1).astype(int)
    tags = tags.reshape(1, tag_max_len).astype(int)
    pos = np.array([pos]).reshape(1, 1).astype(int)
    
    left_context_words = left_context_words.reshape(1, sentence_max_len).astype(int)
    left_context_tags = left_context_tags.reshape(1, sentence_max_len, tag_max_len).astype(int)
    left_context_pos_tags = left_context_pos_tags.reshape(1, sentence_max_len).astype(int)
    left_context_arc_label_vectors = left_context_arc_label_vectors.reshape(1, sentence_max_len, arc_label_vector_len).astype(int)
    
    right_context_words = right_context_words.reshape(1, sentence_max_len).astype(int)
    right_context_tags = right_context_tags.reshape(1, sentence_max_len, tag_max_len).astype(int)
    right_context_pos_tags = right_context_pos_tags.reshape(1, sentence_max_len).astype(int)
    
    return (word, tags, pos, left_context_words, left_context_tags, left_context_pos_tags, left_context_arc_label_vectors,
            right_context_words, right_context_tags, right_context_pos_tags)


def process_single_sentence(sentence, sentence_max_len, tag_max_len, arc_label_vector_len, 
                            tokenizer_word, tokenizer_tag, tokenizer_label, tokenizer_pos, sa, word_form):

    raw_sentence_string_form = " ".join([quadruplet[0] for quadruplet in sentence])
    num_tokens_in_sentence = len(sentence)
    sentence_analysis_result = sa.predict(raw_sentence_string_form)
    sentence_analysis_result = [sentence_analysis.replace('^', '+') for sentence_analysis in sentence_analysis_result]

    arcs = [quadruplet[1] for quadruplet in sentence]
    labels = [quadruplet[2] for quadruplet in sentence]
    labels = tokenizer_label.texts_to_sequences([labels])[0]
    pos_tags = [quadruplet[3] for quadruplet in sentence]
    

    if not len(sentence_analysis_result) == num_tokens_in_sentence:
        raise Exception(sentence, "Length of sentence and sentence_analysis_result don't match")

    batch_of_words = []
    batch_of_tags = []
    batch_of_pos = []
    batch_of_arc_label_vectors = []

    batch_of_left_context_words = []
    batch_of_left_context_tags = []
    batch_of_left_context_pos_tags = []
    batch_of_left_context_arc_label_vectors = []

    batch_of_right_context_words = []
    batch_of_right_context_tags = []
    batch_of_right_context_pos_tags = []

    for t in range(num_tokens_in_sentence):
        X = process_single_word_input(t, sentence, sentence_analysis_result, 
                                      sentence_max_len, tag_max_len, arc_label_vector_len, 
                                      tokenizer_word, tokenizer_tag, tokenizer_label, tokenizer_pos,
                                      arcs, labels, pos_tags, word_form)

        # X
        word_ = X[0][0]
        tags_ = X[1][0]
        pos_ = X[2][0]
        left_context_words_ = X[3][0]
        left_context_tags_ = X[4][0]
        left_context_pos_tags_ = X[5][0]
        left_context_arc_label_vectors_ = X[6][0]
        right_context_words_ = X[7][0]
        right_context_tags_ = X[8][0]
        right_context_pos_tags_ = X[9][0]

        # y
        arc_ = arcs[t]
        label_ = labels[t]
        arc_vector_ = tf.keras.utils.to_categorical(arc_, num_classes = sentence_max_len + 1)
        label_vector_ = tf.keras.utils.to_categorical(label_, num_classes = len(tokenizer_label.word_index) + 1)
        arc_label_vector_ = np.array(arc_vector_.tolist() + label_vector_.tolist())
        
        # Appending
        batch_of_words.append(word_)
        batch_of_tags.append(tags_)
        batch_of_pos.append(pos_)

        batch_of_left_context_words.append(left_context_words_)
        batch_of_left_context_tags.append(left_context_tags_)
        batch_of_left_context_pos_tags.append(left_context_pos_tags_)
        batch_of_left_context_arc_label_vectors.append(left_context_arc_label_vectors_)

        batch_of_right_context_words.append(right_context_words_)
        batch_of_right_context_tags.append(right_context_tags_)
        batch_of_right_context_pos_tags.append(right_context_pos_tags_)
        
        batch_of_arc_label_vectors.append(arc_label_vector_)

    return (batch_of_words, batch_of_tags, batch_of_pos, batch_of_left_context_words,
            batch_of_left_context_tags, batch_of_left_context_pos_tags, batch_of_left_context_arc_label_vectors,
            batch_of_right_context_words, batch_of_right_context_tags, batch_of_right_context_pos_tags,
            batch_of_arc_label_vectors)


def data_generator(files, tokenizer_word, tokenizer_tag, tokenizer_label, tokenizer_pos,
                   sentence_max_len, tag_max_len, arc_label_vector_len, batch_size, word_form = 'whole', shuffle = True):
    
    data = load_quadruplets(files, sentence_max_len)

    # We need morphological tags to train
    with tf.device('/cpu:0'):
        sa = StemmerAnalyzer()
        
    # starting from the first sentence
    sentence_idx = 0
        
    batch_of_words_ = []
    batch_of_tags_ = []
    batch_of_pos_ = []

    batch_of_left_context_words_ = []
    batch_of_left_context_tags_ = []
    batch_of_left_context_pos_tags_ = []
    batch_of_left_context_arc_label_vectors_ = []

    batch_of_right_context_words_ = []
    batch_of_right_context_tags_ = []
    batch_of_right_context_pos_tags_ = []

    batch_of_arc_label_vectors_ = []

    while True:
        while len(batch_of_words_) < batch_size:
            if shuffle:
                sentence_idx = np.random.randint(0, len(data), 1)[0]
            
            sentence = data[sentence_idx]
            
            # I won't use sentences longer than sentence_max_len for training
            # because first token can depend on last token and truncating and breaking
            # the tree structure can hurt training.
            # Also there's no way to output arc greater than sentence_max_len
            # so I will skip them
            if len(sentence) > sentence_max_len:
                sentence_idx += 1 # this prevents infinite loop
                continue
            else:
                (batch_of_words__, batch_of_tags__, batch_of_pos__, batch_of_left_context_words__,
                 batch_of_left_context_tags__, batch_of_left_context_pos_tags__, batch_of_left_context_arc_label_vectors__,
                 batch_of_right_context_words__, batch_of_right_context_tags__, batch_of_right_context_pos_tags__, 
                 batch_of_arc_label_vectors__) = process_single_sentence(sentence, sentence_max_len, 
                                                                        tag_max_len, arc_label_vector_len, 
                                                                        tokenizer_word, tokenizer_tag, 
                                                                        tokenizer_label, tokenizer_pos, sa,
                                                                        word_form)
                batch_of_words_ += batch_of_words__
                batch_of_tags_ += batch_of_tags__
                batch_of_pos_ += batch_of_pos__

                batch_of_left_context_words_ += batch_of_left_context_words__
                batch_of_left_context_tags_ += batch_of_left_context_tags__
                batch_of_left_context_pos_tags_ += batch_of_left_context_pos_tags__
                batch_of_left_context_arc_label_vectors_ += batch_of_left_context_arc_label_vectors__

                batch_of_right_context_words_ += batch_of_right_context_words__
                batch_of_right_context_tags_ += batch_of_right_context_tags__
                batch_of_right_context_pos_tags_ += batch_of_right_context_pos_tags__
                
                batch_of_arc_label_vectors_ += batch_of_arc_label_vectors__
                
                # increase the idx one by as single sentence is processed
                sentence_idx += 1
                if sentence_idx >= len(data):
                    sentence_idx = 0

        # Converting to NumPy array and truncating in case it is larger than batch_size
        batch_of_words = np.array(batch_of_words_)[:batch_size]
        batch_of_tags = np.array(batch_of_tags_)[:batch_size]
        batch_of_pos = np.array(batch_of_pos_)[:batch_size]

        batch_of_left_context_words = np.array(batch_of_left_context_words_)[:batch_size]
        batch_of_left_context_tags = np.array(batch_of_left_context_tags_)[:batch_size]
        batch_of_left_context_pos_tags = np.array(batch_of_left_context_pos_tags_)[:batch_size]
        batch_of_left_context_arc_label_vectors = np.array(batch_of_left_context_arc_label_vectors_)[:batch_size]

        batch_of_right_context_words = np.array(batch_of_right_context_words_)[:batch_size]
        batch_of_right_context_tags = np.array(batch_of_right_context_tags_)[:batch_size]
        batch_of_right_context_pos_tags = np.array(batch_of_right_context_pos_tags_)[:batch_size]
        
        batch_of_arc_label_vectors = np.array(batch_of_arc_label_vectors_)[:batch_size]

        X_train = (batch_of_words, batch_of_tags, batch_of_pos, 
                   batch_of_left_context_words, batch_of_left_context_tags,
                   batch_of_left_context_pos_tags, batch_of_left_context_arc_label_vectors,
                   batch_of_right_context_words, batch_of_right_context_tags, batch_of_right_context_pos_tags)
        y_train = batch_of_arc_label_vectors
        
        # Continue accumulating from the residual part that was truncated before yield
        batch_of_words_ = batch_of_words_[batch_size:]
        batch_of_tags_ = batch_of_tags_[batch_size:]
        batch_of_pos_ = batch_of_pos_[batch_size:]

        batch_of_left_context_words_ = batch_of_left_context_words_[batch_size:]
        batch_of_left_context_tags_ = batch_of_left_context_tags_[batch_size:]
        batch_of_left_context_pos_tags_ = batch_of_left_context_pos_tags_[batch_size:]
        batch_of_left_context_arc_label_vectors_ = batch_of_left_context_arc_label_vectors_[batch_size:]

        batch_of_right_context_words_ = batch_of_right_context_words_[batch_size:]
        batch_of_right_context_tags_ = batch_of_right_context_tags_[batch_size:]
        batch_of_right_context_pos_tags_ = batch_of_right_context_pos_tags_[batch_size:]

        batch_of_arc_label_vectors_ = batch_of_arc_label_vectors_[batch_size:]
        yield (X_train, y_train)

def convert_numbers_to_zero(text_: str):
    text_ = str(text_) # in case input is not string
    text = ""
    for char in text_:
        if char.isnumeric():
            text += "0"
        else:
            text += char
    return text

def number_of_tokens_in(files, sentence_max_len):
    return sum([len(sent) for sent in load_quadruplets(files, sentence_max_len)])

def UAS(y_true, y_pred):
    if not (isinstance(y_pred, np.ndarray)):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
    
    y_true_arc = np.argmax(y_true[:, :sentence_max_len + 1], axis = -1)
    y_pred_arc = np.argmax(y_pred[:, :sentence_max_len + 1], axis = -1)
    
    tp_count = 0
    total_count = y_true.shape[0]
    for idx in range(total_count):
        if y_true_arc[idx] == y_pred_arc[idx]:
            tp_count += 1
    
    return tp_count/total_count

def LAS(y_true, y_pred):
    if not (isinstance(y_pred, np.ndarray)):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
    
    y_true_arc = np.argmax(y_true[:, :sentence_max_len + 1], axis = -1)
    y_pred_arc = np.argmax(y_pred[:, :sentence_max_len + 1], axis = -1)
    
    y_true_label = np.argmax(y_true[:, sentence_max_len + 1: sentence_max_len + 1 + num_unq_labels + 1], axis = -1)
    y_pred_label = np.argmax(y_pred[:, sentence_max_len + 1: sentence_max_len + 1 + num_unq_labels + 1], axis = -1)
    
    tp_count = 0
    total_count = y_true.shape[0]
    for idx in range(total_count):
        if (y_true_arc[idx] == y_pred_arc[idx]) & (y_true_label[idx] == y_pred_label[idx]):
            tp_count += 1
    
    return tp_count/total_count
