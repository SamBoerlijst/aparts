import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from aparts.src.weighted_tagging import split_text_to_sections
from aparts.src.APT import list_filenames, guarantee_csv_exists
import pandas as pd
import os

def remove_repeating_sentences(sentence_tokens:list, sections:dict)->list:
    """
    Removes sentences that occur in all sections, such as footnotes, from a list of tokens.

    Parameters:
    -----------
    sentence_tokens (list): List of tokenized sentences.

    sections (dict): Input text split into sections.

    Returns:
    -----------
    sentence_tokens_filtered (list): List of filtered tokenized sentences.
    """
    sentence_tokens_filtered = []
    for item in sentence_tokens:
        sentence = item.text
        for section in sections.values():
            if sentence not in section:
                sentence_tokens_filtered.append(item)
            else: None
    return sentence_tokens_filtered

def generate_sentence_tokens(text:str) -> tuple[list, dict]:
    """
    Generate a list of tokenized sentences ordered by word frequency. Used as rank during summarization.

    When using the function for the first time, be sure to run: "python -m spacy download en_core_web_sm" to download the language model.
    
    Parameters:
    -----------
    text (str): Source text.

    Returns:
    -----------
    sentence_tokens (list): List of tokenized sentences ordered by word frequency.

    word_frequencies (dict): Dictionary of words and corresponding frequency in the source text.

    """
    nlp = spacy.load('en_core_web_sm')
    doc= nlp(text)
    tokens=[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    return sentence_tokens, word_frequencies

def summarize_tokens(sentence_tokens: list, word_frequencies: dict, amount: int, offset: int) -> str:
    """
    Returns a summary of the requested amount of sentences, by filtering the top sentences scored by the word frequency of the words they are comprised of. 
    
    Parameters:
    -----------
    sentence_tokens (list): List of tokenized sentences.

    word_frequencies (dict): Dictionary containing the corresponding words and their occurence in the source file.

    amount (int): The length of the summary given as the number of sentences.

    offset (int): Excludes the n highest rated sentences from the summary.
    
    Returns:
    -----------
    summary (str): Summarized representation of the source file.
    """
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=amount+offset
    start_position = 0 + offset
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    summary = summary[start_position:]
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    return summary

def summarize_text(text:str, sectiondict:dict, amount:int, offset:int)->str:
    """
    Tokenizes and scores sentences from a string, removes tokens present in all text sections and subsequently generates a summary of the source text.

    Parameters:
    -----------
    text (str): Source text.
    
    sectiondict (dict): Source text split into sections.
    
    amount (int): The length of the summary given as the number of sentences.
    
    offset (int): Excludes the n highest rated sentences from the summary.

    Returns:
    -----------
    summary (str): Summarized representation of the source file.
    """
    text = (text[:99999] + '..') if len(text) > 99999 else text 
    sentence_tokens, word_frequencies = generate_sentence_tokens(text)
    sentence_tokens_filtered = remove_repeating_sentences(sentence_tokens, sectiondict)
    summary = summarize_tokens(sentence_tokens_filtered, word_frequencies, amount, offset)
    return summary

def summarize_file(filepath:str, sections:list, amount:int, offset:int)->str:
    """
    Tokenizes and scores sentences from a source file, removes tokens present in all text sections and subsequently generates a summary of the source text.
    Parameters:
    -----------
    filepath (str): Absolute path to the file to summarize.
    
    sections (list): List of sections to include in summarization.
    
    amount (int): The length of the summary given as the number of sentences.
    
    offset (int): Excludes the n highest rated sentences from the summary.

    Returns:
    -----------
    summary (str): Summarized representation of the source file.
    """
    text = ""
    sectiondict = split_text_to_sections(filepath)
    for item in sections:
        text = text + sectiondict[item]
    text = text.replace("- ", "")
    summary = summarize_text(text, sectiondict, amount, offset)
    return summary

def summarize_csv(outputCSV:str, txtfolder:str, sections:list, amount:int, offset:int, separator: str = ";")->None:
    """
    Generates a summary of the source text for all txt files in the given folder. The generated summary is derived from sentences from the listed sections.
    
    Parameters:
    -----------
    CSV (str): Absolute path to the Excel file the summaries should be exported to.
    
    txtfolder (str): Absolute path to the folder containing the txt files to summarize.
    
    sections (list): List of sections to include in summarization.
    
    amount (int): The length of the summary given as the number of sentences.
    
    offset (int): Excludes the n highest rated sentences from the summary.

    Returns:
    -----------
    summary (str): Summarized representation of the source file.
    """
    header = pd.DataFrame(columns = ["file", "summary"])
    guarantee_csv_exists(outputCSV, header)
    file_list = list_filenames(txtfolder, "*.txt")
    previously_summarized = pd.read_csv(outputCSV, sep = separator)["file"].tolist()
    for item in file_list:
        if str(item) not in previously_summarized:
            filepath = f"{txtfolder}/{item}.txt"
            summary = summarize_file(filepath, sections, amount, offset)
            pd_row = pd.DataFrame({"file": item, "summary": summary}, index=[0])
            pd_row.to_csv(outputCSV, mode='a', index=False, header=False)
            print(f'summarized {item}')
    return

if __name__ == "__main__":
    summarize_csv("C:/NLPvenv/NLP/output/csv/summarized.csv", "C:/NLPvenv/NLP/input/pdf/docs/corrected", ['abstract', 'discussion', 'conclusion'], 2, 2)