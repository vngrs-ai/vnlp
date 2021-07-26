# -*- coding: utf-8 -*-
import re
from collections import namedtuple

WordStruct = namedtuple("WordStruct", ["surface_word", "roots", "suffixes", "tags", "ambiguity_level"])
ANALYSIS_REGEX = re.compile(r"^([^\+]*)\+(.+)$")
TAG_SEPARATOR_REGEX = re.compile(r"[\+\^]")
SPLIT_ROOT_TAGS_REGEX = re.compile(r"^([^\+]+)\+(.+)$")
ADVERB_STANDARDIZER_REGEX = re.compile(r"Adv([^e])")


def to_lower(text):
    text = text.replace("İ", "i")
    text = text.replace("I", "ı")
    text = text.replace("Ğ", "ğ")
    text = text.replace("Ü", "ü")
    text = text.replace("Ö", "ö")
    text = text.replace("Ş", "ş")
    text = text.replace("Ç", "ç")
    return text.lower()


def capitalize(text):
    if len(text) > 1:
        return text[0] + to_lower(text[1:])
    else:
        return text


def asciify(text):
    text = text.replace("İ", "I")
    text = text.replace("Ç", "C")
    text = text.replace("Ğ", "G")
    text = text.replace("Ü", "U")
    text = text.replace("Ş", "S")
    text = text.replace("Ö", "O")
    text = text.replace("ı", "i")
    text = text.replace("ç", "c")
    text = text.replace("ğ", "g")
    text = text.replace("ü", "u")
    text = text.replace("ş", "s")
    text = text.replace("ö", "ö")
    return text


def get_tags_from_analysis(analysis):
    if analysis.startswith("+"):
        return TAG_SEPARATOR_REGEX.split(analysis[2:])
    else:
        return TAG_SEPARATOR_REGEX.split(ANALYSIS_REGEX.sub(r"\2", analysis))


def get_root_from_analysis(analysis):
    if analysis.startswith("+"):
        return "+"
    else:
        return ANALYSIS_REGEX.sub(r"\1", analysis)


def get_pos_from_analysis(analysis):
    tags = get_tags_str_from_analysis(analysis)
    if "^" in tags:
        tags = tags[tags.rfind("^") + 4:]
    return tags.split("+")[0]


def get_tags_str_from_analysis(analysis):
    if analysis.startswith("+"):
        return analysis[2:]
    else:
        return SPLIT_ROOT_TAGS_REGEX.sub(r"\2", analysis)


def standardize_tags(tags):
    new_tags = []
    for tag in tags:
        new_tag = ADVERB_STANDARDIZER_REGEX.sub(r"Adverb\1", tag)
        new_tags.append(new_tag)
    return new_tags


def convert_tag_list_to_str(tags):
    res = "+".join(tags)
    res = res.replace("+DB", "^DB")
    return res