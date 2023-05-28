import os.path as ospath
import re
from os import listdir
from typing import Dict

import pandas as pd

"""
Functions for tagging of bytes like text files weighted by article sections
Date: 1/May/2023 
Author: Sam Boerlijst
"""

# define regex patterns for each section
abstract_pattern = r"abstract:\s*(.*?)\s"
keywords_pattern = r"key(?:word| words)(?::|\s+index:)?\s*(.*?)\s"
introduction_pattern = r"introduction\s*(.*?)\s"
methods_pattern = r"(materials(?:\s+&)?\s+)?methods\s*(.*)"
results_pattern = r"results\s*(.*?)\s"
discussion_pattern = r"discussion\s*(.*?)\s"
conclusion_pattern = r"conclusion\s*(.*?)\s"
references_pattern = r"(?<!taxonomic )(?:taxonomic\s)?(?:references cited|references(?!\s*[A-Z][^a-z]))(?:,(?!$)|.(?!$)|.(?!\s\w)|[^.,\s])\s*([^.,\s]+)"

# create dictionary of patterns and a dictionary to store the sections in
patterns = {
    "abstract": abstract_pattern,
    "keywords": keywords_pattern,
    "introduction": introduction_pattern,
    "methods": methods_pattern,
    "results": results_pattern,
    "discussion": discussion_pattern,
    "conclusion": conclusion_pattern,
    "references": references_pattern,
}
sections = {
    "abstract": "",
    "keywords": "",
    "introduction": "",
    "methods": "",
    "results": "",
    "discussion": "",
    "conclusion": "",
    "references": "",
}

# subroutines
## data preprocessing

def open_file(file: str) -> str:
    text = open(file, "rb").readlines()
    return text


def prepare_bytes_for_pattern(text: str) -> str:
    """
    removes the artifacts from bytes decoding from a given string.

    Parameters:
    text: The bytes-like object to prepare.

    Returns:
    str: The prepared string.
    """
    # remove prepended artifact
    text = text.replace('"b', 'b').replace("b\\'", "").replace("b'", "")
    # remove appended artifact
    text = text.replace('\\n"', '').replace("\\''", "")
    return text

def remove_typographic_line_breaks(text):
    pattern = r'(?<=[a-zA-Z0-9])- (?=[a-zA-Z0-9])'
    return re.sub(pattern, '', text)

def extract_sections(text: str, patterns: dict = patterns, sections: dict = sections) -> dict:
    """
    Extracts the sections from the input text and returns a dictionary
    where each key is a section and the value is the text for that section.

    Parameters:
    text (str): The source string
    patterns (dict): dictionary of regex headers to match
    sections (dict): empty dictionary of headers

    Returns:
    sections (dict): Dictionary of found sections 
    """
    sorted_matches = {}
    for section, pattern in patterns.items():
        match = re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        last_match = None
        for m in match:
            last_match = m
        if last_match:
            sorted_matches[last_match.end()] = section

    last_index = len(text)
    for index in sorted(sorted_matches.keys(), reverse=True):
        section = sorted_matches[index]
        sections[section] = text[index:last_index].strip()
        last_index = index

    sections["abstract"] = text[:last_index].strip()

    return sections


def clean_end_section(patterns: dict = patterns, sections: dict = sections) -> dict:
    """
    Trim end of section by matching with the beginning of the next section

    Parameters:
    sections (dict): The source dictionary

    Returns:
    sections (dict): Dictionary of corrected sections 
    """
    index = []
    for part in sections:
        for key in patterns:
            if key == sections[part]:
                index = list(patterns.keys()).index(key)
                current_section = sections[index]
                next_section = sections[index+1]
                if index == len(sections):
                    return sections
                elif next_section in current_section and next_section != "":
                    string_a = str(current_section)
                    string_b = str(next_section)
                    sections[1] = string_a[:string_a.index(string_b)]
                    break
    return sections

## data processing
def count_keyword_occurrences(section_dictionary: dict, keylist: list) -> dict:
    word_counts = {}

    for section_name, section_text in section_dictionary.items():
        section_word_counts = {}
        words = section_text.split()

        for word in keylist:
            count = words.count(word)
            section_word_counts[word] = count

        word_counts[section_name] = section_word_counts
    return word_counts


## data cleanup
def filter_values(keyword_counts: dict, lower: int = 0) -> dict:
    """
    Remove all keys with a value of 0 from a dictionary (nested or not)

    Parameters:
    keyword_counts (dict): The source dictionary

    Returns:
    dict: The updated dictionary with 0 values removed
    """
    filtered_counts = {}
    for k, v in keyword_counts.items():
        if isinstance(v, dict):
            filtered_v = filter_values(v, lower)
            if filtered_v:
                filtered_counts[k] = filtered_v
        elif v > lower:
            filtered_counts[k] = v
    return filtered_counts


def weigh_keywords(nested_dict) -> dict:
    """
    Weighs a nested dictionary by multiplying the value in the last column based on the first column.
    Weighing is determined as follows:
    Abstract: 4, Discussion: 3, Methods|Results: 2, Introduction:1, References: 0 

    Parameters:
    nested_dict (dict): The source dictionary

    Returns:
    nested_dict (dict): Dictionary with the weighed values 
    """
    weighing_map = {
        "abstract": 4,
        "introduction": 1,
        "methods": 2,
        "results": 2,
        "discussion": 3,
        "references": 0,
    }
    for key, value in nested_dict.items():
        if key in weighing_map and isinstance(value, dict):
            weight = weighing_map[key]
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)):
                    value[sub_key] = sub_value * weight

    return nested_dict


def print_nested_dict(dictionary: dict, indent=0) -> None:
    """
    Prints a dictionary by key: value. If the dictionary is nested it prints it as a line for the key followed by a line of nested key: nested value for each entry within the key.

    Parameters:
    dictionary (dict): The dictionary to be printed

    Returns:
    None
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            print(f"{' ' * indent}{key}: ")
            print_nested_dict(value, indent+2)
        else:
            print(f"{' ' * indent}{key}: {value}")

## data post processing
def denest_and_order_dict(dictionary: dict) -> dict:
    """
    Denests a dictionary and orders it in descending order based on the values of the leaf nodes.

    Parameters:
    dictionary (dict): The source dictionary

    Returns:
    dict: The updated dictionary with all nested keys flattened and sorted in descending order based on leaf node values.
    """
    flat_dict = {}

    # Recursively flatten dictionary and store leaf nodes in a new flat dictionary
    def flatten_dict(dictionary, prefix=''):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                flatten_dict(v, prefix)
            else:
                full_key = prefix + k
                if full_key in flat_dict:
                    flat_dict[full_key] += v
                else:
                    flat_dict[full_key] = v

    flatten_dict(dictionary)

    # Sort the flattened dictionary by values in descending order
    sorted_dict = dict(
        sorted(flat_dict.items(), key=lambda x: x[1], reverse=True))

    return sorted_dict


def nested_dict_to_dataframe(nested_dict: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    rows = []
    for file_name, file_dict in nested_dict.items():
        row = {'filename': file_name}
        row.update(file_dict)
        rows.append(row)
    return pd.DataFrame(rows)


def save_dataframe(dataframe, folder: str):
    """Saves the provided dataframe in the provided folder with headers, adding to the file if already present"""
    dataframe.to_csv(f"{folder}/tagcounts.csv",
                     mode="a", index=False, header=True)

# routines


def split_text_to_sections(text: str) -> dict:
    """
    splits a bytes like text file into sections based on the headers of a scientific article

    Parameters:
    text (str): text to be split into sections

    Returns:
    dict (dict:str): Dictionary of the sections  
    """
    text = open_file(text)
    string_list = [byte.decode('utf-8') for byte in text]
    text = ''.join(string_list)
    text = prepare_bytes_for_pattern(text)
    text = remove_typographic_line_breaks(text)
    dict = extract_sections(text)
    dict = clean_end_section(dict)
    return dict
