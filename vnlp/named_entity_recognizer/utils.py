import re


def ner_to_displacy_format(text, ner_result):
    # Obtain Token Start and End indices
    token_loc = {}
    for duo in ner_result:
        word = duo[0]

        # https://stackoverflow.com/a/13989661/4505301
        # https://stackoverflow.com/questions/66400611/
        if not any([s.isalpha() for s in word]):
            continue

        if not word in token_loc:
            token_loc[word] = []
        for match in re.finditer(word, text):
            start, end = match.start(), match.end()
            substring_of_prev_strings = False
            # check if this match is part of another previously matched string and skip it if so
            for token in token_loc:
                list_of_token_indices = token_loc[token]
                for prev_token_indices in list_of_token_indices:
                    prev_token_start = prev_token_indices[0]
                    prev_token_end = prev_token_indices[1]

                    if (start >= prev_token_start) and (end <= prev_token_end):
                        substring_of_prev_strings = True
                        break

            if (not (start, end) in token_loc[word]) and (
                not substring_of_prev_strings
            ):
                token_loc[word].append((start, end))

    # Process for Spacy
    ner_result_displacy_format = {"text": text, "ents": [], "title": None}

    is_continuation = False
    ents = {}
    for idx, duo in enumerate(ner_result):
        word = duo[0]
        entity = duo[1]

        # https://stackoverflow.com/a/13989661/4505301
        # https://stackoverflow.com/questions/66400611/
        if not any([s.isalpha() for s in word]):
            continue

        start, end = token_loc[word][0]
        del token_loc[word][0]

        if not (entity == "O"):
            if not is_continuation:
                ents["start"] = start
                ents["label"] = entity

            if (idx != (len(ner_result) - 1)) and (
                ner_result[idx + 1][1] == entity
            ):
                is_continuation = True
            else:
                ents["end"] = end
                ner_result_displacy_format["ents"].append(ents)
                ents = {}
                is_continuation = False

    return ner_result_displacy_format
