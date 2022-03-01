import re

from ._yildiz_analyzer import (TurkishStemSuffixCandidateGenerator, get_root_from_analysis, get_tags_from_analysis,
                                 to_lower, standardize_tags, capitalize, WordStruct)

def sentence_generator(file_path):
    sentences = []
    # Sentence level iteration
    for sentence in data_generator(file_path):
        surface_words = []
        analyses = []

        num_tokens_in_sentence = len(sentence)
        # Token (WordStruct) level iteration
        for j in range(num_tokens_in_sentence):
            word_struct = sentence[j]
            surface_words.append(word_struct.surface_word)

            roots = word_struct.roots
            tags = word_struct.tags
            num_analysis = len(roots)

            roots_tags = []
            # Analysis level iteration
            for k in range(num_analysis):
                root = roots[k:k+1]
                tag = tags[k]
                if isinstance(tag, str):
                    tag = [tag]
                else:
                    tag = tag
                root_tag = root + tag
                root_tag = "+".join(root_tag)
                roots_tags.append(root_tag)

            analyses.append(roots_tags)

        processed_sentence = [surface_words, analyses]
        sentences.append(processed_sentence)
        
    return sentences

def data_generator(file_path, add_gold_labels=True, case_sensitive=True, max_lines=0):
    sentence = []
    candidate_generator = TurkishStemSuffixCandidateGenerator(case_sensitive=case_sensitive)
    with open(file_path, "r", encoding="UTF-8") as f:
        for i, line in enumerate(f):
            if 0 < max_lines < i:
                break
            trimmed_line = line.strip(" \r\n\t")
            trimmed_line = trimmed_line.replace("s", "s")
            if trimmed_line.startswith("<S>") or trimmed_line.startswith("<s>"):
                sentence = []
            elif trimmed_line.startswith("</S>") or trimmed_line.startswith("</s>"):
                if len(sentence) > 0:
                    yield sentence
            elif len(trimmed_line) == 0 or "<DOC>" in trimmed_line or trimmed_line.startswith(
                    "</DOC>") or trimmed_line.startswith("<TITLE>") or trimmed_line.startswith("</TITLE>"):
                pass
            else:
                parses = re.split(r"[\t ]", trimmed_line)
                surface = parses[0]
                candidates = candidate_generator.get_analysis_candidates(surface)
                roots = []
                suffixes = []
                tags = []
                ambiguity_level = 0
                if add_gold_labels:
                    analyzes = parses[1:]
                    gold_root = get_root_from_analysis(analyzes[0])
                    gold_root = to_lower(gold_root)
                    roots.append(gold_root)
                    gold_suffix = surface[len(gold_root):]
                    if not case_sensitive:
                        gold_suffix = to_lower(gold_suffix)
                    suffixes.append(gold_suffix)
                    gold_tag = standardize_tags(get_tags_from_analysis(analyzes[0]))
                    tags.append(gold_tag)
                    ambiguity_level = len(analyzes)
                    for candidate_root, candidate_suffix, candidate_tag in candidates:
                        if to_lower(candidate_root) != to_lower(gold_root) or "".join(candidate_tag) != "".join(gold_tag):
                            roots.append(to_lower(candidate_root))
                            suffixes.append(candidate_suffix)
                            tags.append(candidate_tag)
                        elif candidate_suffix != gold_suffix and candidate_root == gold_root:
                            suffixes[0] = candidate_suffix
                else:
                    for candidate_root, candidate_suffix, candidate_tag in candidates:
                        roots.append(candidate_root)
                        suffixes.append(candidate_suffix)
                        tags.append(candidate_tag)
                    if len(roots) == 0:
                        if TurkishStemSuffixCandidateGenerator.STARTS_WITH_UPPER.match(surface):
                            candidate_tags = candidate_generator.get_tags("", stem_tags=["Noun", "Noun+Prop"])
                        else:
                            candidate_tags = candidate_generator.get_tags("", stem_tags=["Noun"])
                        for candidate_tag in candidate_tags:
                            if "Prop" in candidate_tag:
                                roots.append(capitalize(surface))
                                suffixes.append("")
                                tags.append(candidate_tag)
                            else:
                                roots.append(to_lower(surface))
                                suffixes.append("")
                                tags.append(candidate_tag)
                if not case_sensitive:
                    surface = to_lower(surface)
                    roots = [to_lower(root) for root in roots]
                    suffixes = [to_lower(suffix) for suffix in suffixes]
                current_word = WordStruct(surface, roots, suffixes, tags, ambiguity_level)
                sentence.append(current_word)

