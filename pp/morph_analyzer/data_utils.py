# -*- coding: utf-8 -*-
import re

import math
import pandas as pd
from .candidate_generators import TurkishStemSuffixCandidateGenerator
from .utils import get_root_from_analysis, get_tags_from_analysis, WordStruct, \
    convert_tag_list_to_str, to_lower, standardize_tags, capitalize


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


def load_data(file_path, max_sentence=0, add_gold_labels=True, case_sensitive=False):
    sentences = []
    sentence = []
    candidate_generator = TurkishStemSuffixCandidateGenerator(case_sensitive=case_sensitive)
    with open(file_path, "r", encoding="UTF-8") as f:
        for i, line in enumerate(f):
            if 0 < max_sentence < i:
                break
            trimmed_line = line.strip(" \r\n\t")
            trimmed_line = trimmed_line.replace("s", "s")
            if trimmed_line.startswith("<S>") or trimmed_line.startswith("<s>"):
                sentence = []
            elif trimmed_line.startswith("</S>") or trimmed_line.startswith("</s>"):
                if len(sentence) > 0:
                    sentences.append(sentence)
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
                    ambiguity_level = len(analyzes)
                    gold_root = get_root_from_analysis(analyzes[0])
                    gold_root = to_lower(gold_root)
                    roots.append(gold_root)
                    gold_suffix = surface[len(gold_root):]
                    if not case_sensitive:
                        gold_suffix = to_lower(gold_suffix)
                    suffixes.append(gold_suffix)
                    gold_tag = standardize_tags(get_tags_from_analysis(analyzes[0]))
                    tags.append(gold_tag)

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
                current_word = WordStruct(surface, roots, suffixes, tags,ambiguity_level)
                sentence.append(current_word)
    return sentences


def extract_non_existing_stems(file_path):
    non_existing_stems = dict()
    candidate_generator = TurkishStemSuffixCandidateGenerator(case_sensitive=True)
    stem_lookup_table = set()
    tag_flag_map = {v:k for k, v in candidate_generator.TAG_FLAG_MAP.items()}
    for k, _ in candidate_generator.stem_dic.items():
        stem_lookup_table.add(k)
    with open(file_path, "r", encoding="UTF-8") as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                print("Line: {}".format(i))
            trimmed_line = line.strip(" \r\n\t")
            trimmed_line = trimmed_line.replace("s", "s")
            if trimmed_line.startswith("<"):
                continue
            else:
                parses = re.split(r"[\t ]", trimmed_line)
                analyzes = parses[1:]
                gold_stem = get_root_from_analysis(analyzes[0])
                gold_tag_sequence = standardize_tags(get_tags_from_analysis(analyzes[0]))
                if len(gold_tag_sequence) > 1 and gold_tag_sequence[1] == "Prop":
                    gold_pos_tag = gold_tag_sequence[0] + "+" + gold_tag_sequence[1]
                else:
                    gold_pos_tag = gold_tag_sequence[0]
                if gold_pos_tag in ["Punc", "Num", "ESTag", "BDTag"]:
                    continue
                flag = tag_flag_map[gold_pos_tag]
                flag = math.pow(2, flag)

                if gold_stem not in stem_lookup_table and re.match(r"^[A-Za-zİŞĞÜÇÖıüğişçö]+$", gold_stem):
                    if gold_stem not in non_existing_stems:
                        non_existing_stems[gold_stem] = flag
                    else:
                        if non_existing_stems[gold_stem] == flag or \
                                non_existing_stems[gold_stem] - flag in [math.pow(2, i)
                                                                         for i in range(len(tag_flag_map))]:
                            continue
                        else:
                            non_existing_stems[gold_stem] += flag
    return non_existing_stems


def extract_non_existing_tags(file_path):
    non_existing_tags = dict()
    candidate_generator = TurkishStemSuffixCandidateGenerator(case_sensitive=False)
    tag_lookup_table = set()
    for _, v in candidate_generator.suffix_dic.items():
        tag_lookup_table.update(v)
    with open(file_path, "r", encoding="UTF-8") as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                print("Line: {}".format(i))
            trimmed_line = line.strip(" \r\n\t")
            trimmed_line = trimmed_line.replace("s", "s")
            if trimmed_line.startswith("<"):
                continue
            else:
                parses = re.split(r"[\t ]", trimmed_line)
                analyzes = parses[1:]
                gold_tag = convert_tag_list_to_str(standardize_tags(get_tags_from_analysis(analyzes[0])))
                if gold_tag not in tag_lookup_table:
                    if gold_tag not in non_existing_tags:
                        non_existing_tags[gold_tag] = 1
                    else:
                        non_existing_tags[gold_tag] += 1
    return non_existing_tags


def evaluate_candidate_generation(file_path, max_words=0, case_sensitive=True):
    candidate_generator = TurkishStemSuffixCandidateGenerator(case_sensitive=case_sensitive)
    result = []
    with open(file_path, "r", encoding="UTF-8") as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                print("Line: {}".format(i))
            if 0 < max_words < i:
                break
            trimmed_line = line.strip(" \r\n\t")
            if trimmed_line.startswith("<"):
                continue
            else:
                parses = re.split(r"[\t ]", trimmed_line)
                surface = parses[0]
                candidates = candidate_generator.get_analysis_candidates(surface)
                roots = []
                suffixes = []
                tags = []
                analyzes = parses[1:]
                gold_root = get_root_from_analysis(analyzes[0])
                if not case_sensitive:
                    gold_root = to_lower(gold_root)
                gold_tag = convert_tag_list_to_str(standardize_tags(get_tags_from_analysis(analyzes[0])))
                does_contain = False
                for candidate_root, candidate_suffix, candidate_tag in candidates:
                    roots.append(candidate_root)
                    suffixes.append(candidate_suffix)
                    tags.append(convert_tag_list_to_str(candidate_tag))
                    if candidate_root == gold_root and convert_tag_list_to_str(candidate_tag) == gold_tag:
                        does_contain = True
                if not does_contain:
                    if gold_root in roots:
                        correct_root_candidate = gold_root
                    else:
                        correct_root_candidate = "Not Exist"

                    if gold_root in roots:
                        found_analyzes = "\n".join(tags)
                    else:
                        found_analyzes = ""
                    result.append({"Surface Word": surface,
                                   "Gold root": gold_root,
                                   "Gold Tags": gold_tag,
                                   "Selected root candidate": correct_root_candidate,
                                   "Found Tag Sequences": found_analyzes})
    df = pd.DataFrame(result, index=None, columns=["Surface Word", "Gold root",
                                                   "Gold Tags", "Selected root candidate",
                                                   "Found Tag Sequences"])
    df.to_excel("Candidate Generation Error Analysis.xlsx")


if __name__ == "__main__":
    with open("data/data.train.candidates.txt", "w", encoding="UTF-8") as f:
        data = data_generator("data/data.train.txt", add_gold_labels=True)
        for sentence in data:
            for word in sentence:
                f.write("{}\t{}\n".format(word.surface_word, " ".join([root + "+" + "+".join(tag).replace("+DB", "^DB")
                                                                       for root, tag in zip(word.roots, word.tags)])))
    # evaluate_candidate_generation("data/data.train.txt")
    # non_existing_tags = extract_non_existing_tags("data/Morph.Dis.Test.Hand.Labeled-20K.txt")
    # df = pd.DataFrame(non_existing_tags, index=[0])
    # df = df.transpose()
    # df.to_csv("Non existing tags.csv")
    #
    # non_existing_stems = extract_non_existing_stems("data/Morph.Dis.Test.Hand.Labeled-20K.txt")
    # df = pd.DataFrame(non_existing_stems, index=[0])
    # df = df.transpose()
    # df.to_csv("Non existing stems.csv")

