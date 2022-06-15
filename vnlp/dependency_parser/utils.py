import numpy as np

def dp_pos_to_displacy_format(dp_result, pos_result = None):
    """
    Converts Dependency Parser result to Displacy format.
    Displacy requires that PoS tags are also provided, however this implementation allows skipping it.
    """

    # In case PoS result is not provided, use empty strings.
    if pos_result is None:
        pos_result = [(triplet[0], '') for triplet in dp_result]

    dp_result_displacy_format = {'words': [],
                   'arcs': []}
    for dp_res, pos_res in zip(dp_result, pos_result):
        word = dp_res[1]
        pos_tag = pos_res[1]
        
        arc_source = dp_res[0] - 1
        arc_dest = dp_res[2] - 1
        dp_label = dp_res[3]
        
        dp_result_displacy_format['words'].append({'text': word, 'tag': pos_tag})
        if arc_dest < 0:
            continue
        else:
            if arc_source <= arc_dest:
                dp_result_displacy_format['arcs'].append({'start': arc_source, 'end': arc_dest, 'label': dp_label, 'dir': 'right'})
            else:
                dp_result_displacy_format['arcs'].append({'start': arc_dest, 'end': arc_source, 'label': dp_label, 'dir': 'left'})

    return dp_result_displacy_format

def decode_arc_label_vector(logits, SENTENCE_MAX_LEN, LABEL_VOCAB_SIZE):
    """
    Converts the output vector of Dependency Parser model to arc and label values
    """
    arc = np.argmax(logits[:SENTENCE_MAX_LEN + 1]) # +1 is due to reserving of arc 0 for root. Don't confuse it with padding 0!
    label = np.argmax(logits[SENTENCE_MAX_LEN + 1: SENTENCE_MAX_LEN + 1 + LABEL_VOCAB_SIZE + 1])
    
    return arc, label