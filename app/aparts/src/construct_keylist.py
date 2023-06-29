import os
import re
from operator import itemgetter

import gensim
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from anyascii import anyascii
from cleantext import clean
from fuzzywuzzy import fuzz
from gensim import models
from gensim.parsing.preprocessing import remove_stopwords, strip_short
from gensim.utils import simple_preprocess
from keybert import KeyBERT
from nlp_rake import Rake
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from pybtex.database import parse_file
from pybtex.database.input import bibtex
from six import iteritems
from spacy import load
from yake import KeywordExtractor

""" aparts
Academic Pdf - Automated Reference Tagging System: Collect keywords from (web of science or google scholar) csv list of titles and abstracts using 7 common NLP algorithms, combine those with author given tags and tags present in bib file and export as csv

- Author: Sam Boerlijst
- Date: 9/5/2023

"""

blacklist = ["amount", "at http", "at https", "com", "copyright", "female", "male", "nan",
             "number", "net", "org", "parameter", "result", "species", "study", "total", "www"]

# when using nltk for the first time you might need to download the stopword list Using 'from nltk import download' and 'download('stopwords')'

# build lists to store tags and scores
taglist = []
scorelist = []

# build folder structure


def guarantee_folder_exists(folder: str) -> None:
    """Create folder if not yet present."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    return


def generate_folder_structure() -> None:
    folders = ["input/pdf/docs/corrected",
               "output/bib", "output/csv", "output/md"]
    for folder in folders:
        guarantee_folder_exists(folder)
    return

# input sanitization


def do_clean(text: str) -> str:
    """
    Cleans text given by transliterating to ascii and lowercase.

    Parameters:
    -----------
    text (str): Text to be cleaned.

    Return:
    text (str): Transliterated text.
    -----------
    """
    return clean(
        text,
        lowercase=True,
        extra_spaces=False,
        numbers=False,
        punct=False,
    )


def clean_keywords(keywords: str) -> list:
    keywords = (
        keywords.replace("'", "")
        .replace(",", ";")
        .replace("[", "")
        .replace("]", "")
        .replace("{", "")
        .replace("}", "")
        .split("; ")
    )
    return keywords

# collect keywords from WOS file


def get_original_keywords(input_folder:str, records: str, author_given_keywords: str, original_keywords_txt: str) -> None:
    """
    Retrieves all keywords indexed by Web of Science and cleans any delimiter artifacts.

    Parameters:
    -----------
    records (str): Path to the CSV file containing the Web of Science records.
    original_keywords_txt (str): Path to the text file to store the keywords.

    Returns:
    -----------
    None
    """
    print("Collecting given tags")
    original_keywords_txt_path = f"{input_folder}/{original_keywords_txt}.txt"
    records_path = f"{input_folder}/{records}.csv"
    dataframe = pd.read_csv(records_path)
    author_keywords = str(dataframe[author_given_keywords].to_list())

    # Fix various errors
    author_keywords = do_clean(author_keywords)

    # Convert to list
    author_keywords = clean_keywords(author_keywords)

    # Filter uniques
    author_keywords = list(set(author_keywords))

    wos_keywords_original = sorted(list(set(author_keywords)))

    with open(original_keywords_txt_path, "w", encoding="utf-8") as file:
        file.write(str(wos_keywords_original))

    print("Keywords collected and stored in", original_keywords_txt_path)

    return


# collect keywords using bigrams
def bigram_extraction(records: str, WOScolumn: str, name: str, amount: int, input_folder: str) -> None:
    """
    Determine keywords from the given column for each entry in a csv file using the bigram algorithm. 

    Parameters:
    -----------
    records (str): Path to the csv containing the Web of Science records.

    WOScolumn (str): The column name containing the text to analyse.

    name (str): String stating whether titles or abstracts are being analysed. May be equal to either "wos_a" or "wos_t". Additionally this parameter will be used as output name.

    amount (int): Number of keywords to select.

    Return:
    -----------
    None
    """
    if name == "wos_a":
        print("generating bigram tags from abstracts")
    elif name == "wos_t":
        print("generating bigram tags from titles")
    data = pd.read_csv(records)[WOScolumn].tolist()
    data = str(data).encode(encoding="unicode_escape")
    data = gensim.parsing.preprocessing.remove_stopwords(data)
    doc = strip_short(data, minsize=4)
    # clean input
    doc = doc.replace("[", "").replace("]", "").replace("{", "").replace(
        "}", "").replace("<", "").replace(">", "").replace("%", "")
    doc = do_clean(doc).split(",")
    # NLTK Stop words
    stop_words = stopwords.words("english")
    stop_words.extend(["from", "subject", "re", "edu", "use"])

    def sent_to_words(sentences):
        """Tokenize sentences."""
        for sentence in sentences:
            yield (
                gensim.utils.simple_preprocess(str(sentence), deacc=True)
            )  # deacc=True removes punctuations

    monogram = list(sent_to_words(doc))
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(
        monogram, min_count=2, threshold=2, delimiter=" "
    )  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[monogram], threshold=2)
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    # Define functions for stopwords, bigrams, trigrams and lemmatization

    def remove_stopwords(texts):
        """Remove stopwords from string."""
        return [
            [word for word in simple_preprocess(
                str(doc)) if word not in stop_words]
            for doc in texts
        ]

    def make_bigrams(texts):
        """Generate bigrams from list of strings"""
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        """Generate bigrams from list of strings"""
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    # Remove Stop Words
    monogram_nostops = remove_stopwords(monogram)
    # Form Bigrams
    monograms_bigrams = make_bigrams(monogram_nostops)
    # Create Dictionary
    id2word = corpora.Dictionary(monograms_bigrams)
    # Create Corpus
    texts = monograms_bigrams
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=20,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha="auto",
        per_word_topics=True,
    )
    # Compute Perplexity
    print(
        "\nPerplexity: ", lda_model.log_perplexity(corpus)
    )  # a measure of how good the model is. lower the better.
    #   print(phrase, score)
    bigram_list = bigram.find_phrases(monogram)
    myKeys = list(bigram_list.keys())
    myVal = list(bigram_list.values())
    # myKeys.sort(reverse = True)
    sorted_bigram_list = {i: bigram_list[i] for i in myKeys}
    for doc in bigram_list:
        df = pd.DataFrame.from_dict({"ID": myKeys, "frequency": myVal})
    df = df.sort_values(by="frequency", ascending=False)
    df = df[0:amount]
    df.to_csv(f"{input_folder}/bigram_{name}.csv", index=False)
    return


# collect keywords using keybert
def keybert_extraction(records: str, WOScolumn: str, name: str, amount: int, input_folder: str) -> None:
    """
    Determine keywords from the given column for each entry in a csv file using the keybert algorithm. 

    Parameters:
    -----------
    records (str): Path to the csv containing the Web of Science records.

    WOScolumn (str): The column name containing the text to analyse.

    name (str): String stating whether titles or abstracts are being analysed. May be equal to either "wos_a" or "wos_t". Additionally this parameter will be used as output name.

    amount (int): Number of keywords to select.

    Return:
    -----------
    None
    """
    if name == "wos_a":
        print("generating keybert tags from abstracts")
    elif name == "wos_t":
        print("generating keybert tags from titles")
    sourcefile = pd.read_csv(records)[WOScolumn].tolist()
    data = sourcefile
    data = str(data).encode(encoding="unicode_escape")
    data = remove_stopwords(data)
    doc = strip_short(data)
    # clean input
    do_clean(doc)
    doc = doc.replace("[", "").replace("]", "").replace("{", "").replace(
        "}", "").replace("<", "").replace(">", "").replace("%", "")
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(
        doc,
        keyphrase_ngram_range=(1, 1),
        stop_words="english",
        use_mmr=True,
        diversity=0.4,
        nr_candidates=40,
        top_n=amount,
    )
    for doc in keywords:
        df = pd.DataFrame(data=keywords, columns=["ID", "frequency"])
        df.to_csv(f"{input_folder}/keybert_{name}.csv", index=False)
    return


# collect keywords using RAKE
def rake_extraction(records: str, WOScolumn: str, name: str, Rake_stoppath: str, amount: int, input_folder: str) -> None:
    """
    Determine keywords from the given column for each entry in a csv file using the RAKE algorithm. 

    Parameters:
    -----------
    records (str): Path to the csv containing the Web of Science records.

    WOScolumn (str): The column name containing the text to analyse.

    name (str): String stating whether titles or abstracts are being analysed. May be equal to either "wos_a" or "wos_t". Additionally this parameter will be used as output name.

    Rake_stoppath (str): Path to the file containing stopwords to use.

    amount (int): Number of keywords to select.

    Return:
    -----------
    None
    """
    rake = Rake(
        min_chars=3,
        max_words=3,
        min_freq=2,
        language_code=None,  # 'en'
        stopwords=None,  # {'and', 'of'}
        lang_detect_threshold=50,
        max_words_unknown_lang=2,
        generated_stopwords_percentile=80,
        generated_stopwords_max_len=3,
        generated_stopwords_min_freq=2,
    )
    
    if name == "wos_a":
        print("generating rake tags from abstracts")
    elif name == "wos_t":
        print("generating rake tags from titles")
    data = pd.read_csv(records)[WOScolumn].tolist()
    data = str(data).encode(encoding="unicode_escape")
    data = remove_stopwords(data)
    text = strip_short(data)
    text = do_clean(text)
    text = text.replace("[", "").replace("]", "").replace(
        "{", "").replace("}", "").replace("<", "").replace(">", "").replace("%", "")
    
    rake_keywords = rake.apply(text, text_for_stopwords=text)
    df = pd.DataFrame(data=rake_keywords[0:amount], columns=[
                      "ID", "frequency"])
    df.to_csv(f"{input_folder}/rake_{name}.csv", index=False)
    return


# collect keywords using textrank
def textrank_calculation(text: str, top_n: int = 5, n_gram: int = 3, output_graph: bool = False)-> (tuple[nx.Graph, pd.DataFrame] or pd.DataFrame):
    """
    Determine key-phrases from the provided text using the textrank algorithm. 

    Parameters:
    -----------
    text (str): String of the phrases to analyse.

    top_n (int): The number of key-phrases to extract.

    n_gram (int): Length of the key-phrases in words

    output_graph (bool): wherher or not a graph item should be returned for vidualization.

    Return:
    -----------
    graph (nx.Graph): Graph item representing the distance among key-phrases

    keyphrases (pd.DataFrame): Selected keyphrases and their frequency
    """
    sentences = sent_tokenize(text)
    extractor = KeywordExtractor(n=n_gram)
    candidate_keywords = [
        keyword[0] for sentence in sentences for keyword in extractor.extract_keywords(sentence)]

    graph = nx.Graph()
    graph.add_nodes_from(candidate_keywords)
    for sentence in sentences:
        sentence_keywords = [kw for kw in candidate_keywords if kw in sentence]
        for u in sentence_keywords:
            for v in sentence_keywords:
                if u != v:
                    if graph.has_edge(u, v):
                        graph[u][v]['weight'] += 1
                    else:
                        graph.add_edge(u, v, weight=1)

    scores = nx.pagerank(graph)
    top_keywords = sorted(scores, key=scores.get, reverse=True)[:top_n]
    graph.remove_nodes_from(set(graph.nodes) - set(top_keywords))

    keyphrases = pd.DataFrame({'ID': top_keywords, 'score': [scores[kw] for kw in top_keywords]})
    if output_graph == True:
        return graph, keyphrases
    else:
        return keyphrases

def visualize_textrank_graph(graph:nx.Graph) -> None:
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, k=0.3, iterations=50)
    node_sizes = [300 * graph.degree(node) for node in graph.nodes]
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes,
                           alpha=0.8, node_color='skyblue')
    edge_widths = [0.1 * graph[u][v]['weight'] for u, v in graph.edges]
    nx.draw_networkx_edges(graph, pos, width=edge_widths,
                           alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(graph, pos, font_size=8)
    plt.margins(0.1)
    plt.axis('off')
    plt.show()
    return None

def textrank_extraction(records: str, WOScolumn: str, name: str, amount: int, input_folder: str) -> None:
    """
    Determine keywords from the given column for each entry in a csv file using the TextRank algorithm. 

    Parameters:
    -----------
    records (str): Path to the csv containing the Web of Science records.

    WOScolumn (str): The column name containing the text to analyse.

    name (str): String stating whether titles or abstracts are being analysed. May be equal to either "wos_a" or "wos_t". Additionally this parameter will be used as output name.

    amount (int): Number of keywords to select.

    Return:
    -----------
    None
    """
    if name == "wos_a":
        print("generating textrank tags from abstracts")
    elif name == "wos_t":
        print("generating textrank tags from titles")
    nlp = load("en_core_web_sm")
    data = pd.read_csv(records)[WOScolumn].tolist()
    data = str(data).encode(encoding="unicode_escape")
    data = remove_stopwords(data)
    text = strip_short(data)
    # clean input
    text = do_clean(text)
    text = text.replace("[", "").replace("]", "").replace(
        "{", "").replace("}", "").replace("<", "").replace(">", "").replace("%", "")
    nlp.max_length = len(text) + 100
    keyphrases = textrank_calculation(text, amount, 1, False)
    keyphrases.to_csv(f"{input_folder}/TextR_{name}.csv", index=False)
    return


# collect keywords using topicrank
def topicrank_calculation(text:str="", top_n:int=30, n_gram:int = 2) -> pd.DataFrame:
    """
    Determine key-phrases from the provided text using the topicrank algorithm. 

    Parameters:
    -----------
    text (str): String of the phrases to analyse.

    top_n (int): The number of key-phrases to extract.

    n_gram (int): Length of the key-phrases in words

    Return:
    -----------
    keyphrases (pd.DataFrame): Selected keyphrases and their frequency
    """
    extractor = KeywordExtractor(n=n_gram)
    keywords = extractor.extract_keywords(text)
    topicrank_keywords = [(keyword, score) for keyword, score in keywords][:top_n]
    keyphrases = pd.DataFrame(topicrank_keywords, columns=["ID", "frequency"])
    return keyphrases

def topicrank_extraction(records: str, WOScolumn: str, name: str, amount: int, input_folder: str) -> None:
    """
    Determine keywords from the given column for each entry in a csv file using the TopicRank algorithm. 

    Parameters:
    -----------
    records (str): Path to the csv containing the Web of Science records.

    WOScolumn (str): The column name containing the text to analyse.

    name (str): String stating whether titles or abstracts are being analysed. May be equal to either "wos_a" or "wos_t". Additionally this parameter will be used as output name.

    amount (int): Number of keywords to select.

    Return:
    -----------
    None
    """
    if name == "wos_a":
        print("generating topicrank tags from abstracts")
    elif name == "wos_t":
        print("generating topicrank tags from titles")
    nlp = load("en_core_web_sm")
    nlp.max_length = 1500000
    data = pd.read_csv(records)[WOScolumn].tolist()
    data = str(data).encode(encoding="unicode_escape")
    data = remove_stopwords(data)
    text = strip_short(data)
    # clean input
    do_clean(text)
    text = text.replace("[", "").replace("]", "").replace(
        "{", "").replace("}", "").replace("<", "").replace(">", "").replace("%", "")
    df = topicrank_calculation(text, amount, 1)
    df.to_csv(f"{input_folder}/topicR_{name}.csv", index=False)
    return


# collect keywords using tf-idf
def tf_idf_extraction(records: str, WOScolumn: str, name: str, input_folder: str) -> None:
    """
    Determine keywords from the given column for each entry in a csv file using the TF-IDF algorithm. 

    Parameters:
    -----------
    records (str): Path to the csv containing the Web of Science records.

    WOScolumn (str): The column name containing the text to analyse.

    name (str): String stating whether titles or abstracts are being analysed. May be equal to either "wos_a" or "wos_t". Additionally this parameter will be used as output name.

    Return:
    -----------
    None
    """
    if name == "wos_a":
        print("generating tf_idf tags from abstracts")
    elif name == "wos_t":
        print("generating tf_idf tags from titles")
    data = pd.read_csv(records)[WOScolumn].tolist()
    data = str(data).encode(encoding="unicode_escape")
    data = remove_stopwords(data)
    doc = strip_short(data)
    doc = do_clean(doc)
    doc = doc.replace("[", "").replace("]", "").replace("{", "").replace(
        "}", "").replace("<", "").replace(">", "").replace("%", "")
    doc = list(pd.Series(data))
    # Create the Tokens, Dictionary and Corpus
    text_tokens = [[tok for tok in doc.split(",")] for doc in doc]
    mydict = corpora.Dictionary([simple_preprocess(line) for line in doc])
    corpus = [mydict.doc2bow(simple_preprocess(line)) for line in doc]
    # Create the TF-IDF model
    tfidf = models.TfidfModel(corpus, smartirs="ntc")
    for doc in tfidf[corpus]:
        locals()["data_{0}".format(doc)] = pd.DataFrame(
            data=[[mydict[id], np.around(freq, decimals=2)]
                  for id, freq in doc],
            columns=["ID", "frequency"],
        )
        locals()["data_{0}".format(doc)] = locals()["data_{0}".format(doc)].sort_values(
            by="frequency", ascending=False
        )
        locals()["data_{0}".format(doc)] = locals()["data_{0}".format(doc)][
            locals()["data_{0}".format(doc)]["frequency"] > 0.01
        ]
        locals()["data_{0}".format(doc)].to_csv(
            f"{input_folder}/tf-idf_{name}.csv", index=False
        )
    return


# collect keywords using yake
def yake_extraction(records: str, WOScolumn: str, name: str, amount: int, input_folder: str) -> None:
    """
    Determine keywords from the given column for each entry in a csv file using the YAKE algorithm. 

    Parameters:
    -----------
    records (str): Path to the csv containing the Web of Science records.

    WOScolumn (str): The column name containing the text to analyse.

    name (str): String stating whether titles or abstracts are being analysed. May be equal to either "wos_a" or "wos_t". Additionally this parameter will be used as output name.

    amount (int): Number of keywords to select.

    Return:
    -----------
    None
    """
    if name == "wos_a":
        print("generating yake tags from abstracts")
    elif name == "wos_t":
        print("generating yake tags from titles")
    data = pd.read_csv(records)[WOScolumn].tolist()
    data = str(data).encode(encoding="unicode_escape")
    data = remove_stopwords(data)
    text = strip_short(data)
    text = do_clean(text)
    text = text.replace("[", "").replace("]", "").replace(
        "{", "").replace("}", "").replace("<", "").replace(">", "").replace("%", "")
    language = "en"
    max_ngram_size = 2
    deduplication_threshold = 0.9
    deduplication_algo = "seqm"
    windowSize = 1
    numOfKeywords = amount
    custom_kw_extractor = KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedupLim=deduplication_threshold,
        dedupFunc=deduplication_algo,
        windowsSize=windowSize,
        top=numOfKeywords,
        features=None,
    )
    keywords = custom_kw_extractor.extract_keywords(text)
    df = pd.DataFrame(data=keywords, columns=["ID", "frequency"])
    df.to_csv(f"{input_folder}/yake_{name}.csv", index=False)
    return


# import .bib file
def import_bib(input_folder: str, bibfile: str, libtex_csv: str) -> None:
    """
    Extracts metadata for each entry in a .bib file and stores it in a csv file in the given folder.

    Parameters:
    -----------
    bibfile (str): Path to the bib file to be extracted.

    libtex_csv (str): Path to the csv file to write the output to.

    Return:
    -----------
    None

    """
    print("importing .bib file")
    parser = bibtex.Parser()
    bib_data = parse_file(f"{input_folder}/{bibfile}.bib")
    dataset = pd.DataFrame(
        columns=[
            "entry",
            "author",
            "title",
            "keywords",
            "abstract",
            "doi",
            "date",
            "file",
        ]
    )
    # build dataframe by looping over entries
    for entry in bib_data.entries.values():
        bibdf = pd.DataFrame()
        authorlist = []
        columnlist = []
        entryname = str(entry.key)
        for val in range(len(entry.fields.keys())):
            name = list(entry.fields.keys())[val]
            if str(name) == "file":
                value = list(entry.fields.values())[val]
                value = value.split("\\").pop()
                value = value.split(".p").pop(0)
                value = value.split(".P").pop(0)
            else:
                value = list(entry.fields.values())[val]
                # remove special characters from string
                value = (str(value)
                         .replace(":", "_")
                         .replace(";", "")
                         .replace("=", "")
                         .replace(".", "_")
                         .replace("{", "")
                         .replace("}", "")
                         .replace("?", "")
                         .replace(",", "")
                         .replace("\\\\", "")
                         .replace("\\", "")
                         .replace("/", "")
                         .replace("*", "")
                         .replace('"', "")
                         .replace("'", ""))
            dfcolumn = pd.DataFrame({name: value}, index=[0])
            bibdf = pd.concat([bibdf, dfcolumn], axis=1, ignore_index=True)
            columnlist.append(str(name))
        for name in entry.persons["author"]:
            authorlist.append(str(name))
        authorlist = str(" and ".join(authorlist))
        dfcolumn = pd.DataFrame({str("author"): authorlist}, index=[0])
        bibdf.columns = list(columnlist)
        bibdf["author"] = dfcolumn["author"]
        dfcolumn = pd.DataFrame({str("entry"): entryname}, index=[0])
        bibdf["entry"] = dfcolumn["entry"]
        dataset = pd.concat([dataset, bibdf], ignore_index=True)
    dataset.to_csv(f"{input_folder}/{libtex_csv}.csv", index=False)
    return


# collect keywords using all algorithms
def extract_tags(records, column, name, Rake_stoppath, amount, input_folder) -> None:
    """
    Determine keywords from the given column for each entry in a csv file using all seven algorithms: bigram, keybert, rake, textrank, topicrank, tf-idf and yake. 

    Parameters:
    -----------
    records (str): Path to the csv containing the Web of Science records.

    WOScolumn (str): The column name containing the text to analyse.

    name (str): String stating whether titles or abstracts are being analysed. May be equal to either "wos_a" or "wos_t". Additionally this parameter will be used as output name.

    Rake_stoppath (str): Path to the file containing stopwords to use.

    amount (int): Number of keywords to select.

    Return:
    -----------
    None
    """
    records_path = f"{input_folder}/{records}.csv"
    bigram_extraction(records_path, column, name, amount, input_folder)
    keybert_extraction(records_path, column, name, amount, input_folder)
    rake_extraction(records_path, column, name, Rake_stoppath, amount, input_folder)
    textrank_extraction(records_path, column, name, amount, input_folder)
    topicrank_extraction(records_path, column, name, amount, input_folder)
    tf_idf_extraction(records_path, column, name, input_folder)
    yake_extraction(records_path, column, name, amount, input_folder)
    return


# construct keylist
def construct_keylist(blacklist: list = blacklist, libtex_csv: str = "", bibfile: str = "", output_name: str = "", input_folder:str="", author_given_keywords: str = "", original_keywords_txt:str=""):
    """
    Creates a masterlist of keywords from all seven algorithms, any keywords present in the wos file and any keywords present in the bib file, by filtering for unique keywords and filtering by stem.

    Parameters:
    -----------
    blacklist (list): list of strings to exclude.

    libtex_csv (str): Path to the csv file containing bib formatted citation metadata.

    Return:
    -----------
    None
    """
    print("constructing keyword list from combined output")

    if bibfile != "":
        bib_original = pd.read_csv(libtex_csv)["keywords"].tolist()
    else:
        bib_original = []
    if author_given_keywords != "":
        wos_original = open(f"{input_folder}/{original_keywords_txt}.txt", "r").readlines()
    else:
        wos_original = [""]
    bigram_a = pd.read_csv(f"{input_folder}/bigram_wos_a.csv")["ID"].tolist()
    bigram_t = pd.read_csv(f"{input_folder}/bigram_wos_t.csv")["ID"].tolist()
    keybert_a = pd.read_csv(f"{input_folder}/keybert_wos_a.csv")["ID"].tolist()
    keybert_t = pd.read_csv(f"{input_folder}/keybert_wos_t.csv")["ID"].tolist()
    TextR_a = pd.read_csv(f"{input_folder}/TextR_wos_a.csv")["ID"].tolist()
    TextR_t = pd.read_csv(f"{input_folder}/TextR_wos_t.csv")["ID"].tolist()
    tf_idf_a = pd.read_csv(f"{input_folder}/tf-idf_wos_a.csv")["ID"].tolist()
    tf_idf_t = pd.read_csv(f"{input_folder}/tf-idf_wos_t.csv")["ID"].tolist()
    topicR_a = pd.read_csv(f"{input_folder}/topicR_wos_a.csv")["ID"].tolist()
    topicR_t = pd.read_csv(f"{input_folder}/topicR_wos_t.csv")["ID"].tolist()
    yake_a = pd.read_csv(f"{input_folder}/yake_wos_a.csv")["ID"].tolist()
    yake_t = pd.read_csv(f"{input_folder}/yake_wos_t.csv")["ID"].tolist()
    newlist = list()
    cor_bigram_a = []
    cor_bigram_t = []
    cor_keybert_a = []
    cor_keybert_t = []
    cor_TextR_a = []
    cor_TextR_t = []
    cor_tf_idf_a = []
    cor_tf_idf_t = []
    cor_topicR_a = []
    cor_topicR_t = []
    cor_yake_a = []
    cor_yake_t = []
    stem_bigram_a = []
    stem_bigram_t = []
    stem_keybert_a = []
    stem_keybert_t = []
    stem_TextR_a = []
    stem_TextR_t = []
    stem_tf_idf_a = []
    stem_tf_idf_t = []
    stem_topicR_a = []
    stem_topicR_t = []
    stem_yake_a = []
    stem_yake_t = []
    stemcor_bigram_t = []
    stemcor_bigram_a = []
    stemcor_keybert_a = []
    stemcor_keybert_t = []
    stemcor_TextR_a = []
    stemcor_TextR_t = []
    stemcor_tf_idf_a = []
    stemcor_tf_idf_t = []
    stemcor_topicR_a = []
    stemcor_topicR_t = []
    stemcor_yake_a = []
    stemcor_yake_t = []
    blacklist1 = re.compile("|".join([re.escape(word) for word in blacklist]))
    finallist = []

    def filterblacklist(newlist):
        for i in range(len(newlist)):
            if fuzz.partial_ratio(newlist[i], blacklist) < 80:
                finallist.append(newlist[i])
        return

    comparelist = [
        "bigram_a",
        "bigram_t",
        "keybert_a",
        "keybert_t",
        "TextR_a",
        "TextR_t",
        "tf_idf_a",
        "tf_idf_t",
        "topicR_a",
        "topicR_t",
        "yake_a",
        "yake_t",
    ]

    # filter by length
    for item in comparelist:
        test = eval(item)
        for i in range(len(test)):
            if not len(eval(item)[i]) > 3:
                eval(item)[i] = ""

    # create stem list and filter by stem
    ps = PorterStemmer()
    for item in comparelist:
        test = eval(item)
        stem = 'stem_' + item
        for i in range(len(test)):
            eval(stem).append(ps.stem(test[i]))
    for item in comparelist:
        test = eval(item)
        stem = 'stem_' + item
        cor = 'cor_' + item
        stemcor = 'stemcor_' + item
        for i in range(len(test)):
            if not eval(stem)[i] in eval(stemcor):
                eval(stemcor).append(eval(stem)[i])
                eval(cor).append(eval(item)[i])

    # combine lists
    for item in comparelist:
        cor = 'cor_' + item
        test = eval(cor)
        for value in test:
            if value in taglist:
                for i in taglist:
                    if i == value:
                        scorelist[taglist.index(
                            i)] = scorelist[taglist.index(i)]+1
            else:
                taglist.append(value)
                scorelist.append(1)
    tagmatrix = []

    # filter items that occur in between 2-4 of the lists (common, but not too common)
    for i in range(len(scorelist)):
        if scorelist[i] > 1 and scorelist[i] < 5:
            tagmatrix.append([taglist[i], scorelist[i]])
            newlist.append(taglist[i])

    # append author given keywords and user given keywords
    for i in range(len(wos_original)):
        item = str(wos_original[i])
        item = anyascii(item)
        newlist.append(item)
    for i in range(len(bib_original)):
        item = str(bib_original[i])
        item = anyascii(item)
        newlist.append(item)

    # filter uniques and replace special characters
    newlist = (
        str(sorted(list(set(newlist))))
        .replace('"', "")
        .replace("'", "")
        .replace(",", ";")
        .replace("[", "")
        .replace("]", "")
        .replace("<", "")
        .replace(">", "")
        .replace("{", "")
        .replace("}", "")
        .replace("*", "")
        .replace("&", "and")
        .replace(":", "")
        .replace("\\", "")
        .replace("\_", " ")
        .replace("/", " ")
        .replace("_", " ")
        .split("; ")
    )
    # fuzzy search blacklist
    filterblacklist(newlist)
    finallist = str(sorted(list(set(finallist)))).replace("'", "").split(",")
    # direct search blacklist
    finallist = [word for word in finallist if not blacklist1.search(
        word) and len(word) > 2]
    # write to csv
    finallist = pd.DataFrame(finallist, columns=["ID"])
    finallist = (
        finallist.sort_values(by="ID", axis=0, ascending=True)
        .drop_duplicates("ID", keep="last")
        .reset_index(drop=True)
    )
    finallist.to_csv(f"{input_folder}/{output_name}.csv", index=False)
    return


# complete keylist routine
def generate_keylist(input_folder="", records="", titlecolumn="Article Title", abstactcolumn="Abstract", bibfile="", libtex_csv="", output_name="", original_keywords_txt="", blacklist=blacklist, amount=50, author_given_keywords="", Rake_stoppath=""):
    """
    Gerenates a keyword list using the Web of Science records by 1) extracting indexed keywords 2) filtering article titles for keywords using all seven algorithms,  3) filtering article abstracts for keywords using all seven algorithms, 4) extracting keywords present in a bib file and 5) filtering for unique values excluding keywords present in the blacklist.
    All parameters but the WOS file path and bibfile path have default values.

    Parameters:
    -----------
    input_path (str): Path to the folder containing all files used as source.

    records (str): Filename within the input_path to the csv containing the Web of Science records.

    titlecolumn (str): The column name containing the titles to analyse.

    abstactcolumn (str): The column name containing the abstracts to analyse.

    bibfile (str): Filename within the input_path to the bib file to be extracted.

    libtex_csv (str): Filename within the input_path to the csv file to write the output to.

    output_path (str):

    blacklist (list): list of strings to exclude.

    amount (int): Number of keywords to select.

    Return:
    -----------
    None
    """
    if author_given_keywords != "":
        get_original_keywords(input_folder=input_folder, records=records, author_given_keywords=author_given_keywords, original_keywords_txt=original_keywords_txt)
    extract_tags(name="wos_t", column=titlecolumn, records=records, Rake_stoppath=Rake_stoppath, amount=amount, input_folder=input_folder)
    extract_tags(name="wos_a", column=abstactcolumn, records=records, Rake_stoppath=Rake_stoppath, amount=amount, input_folder=input_folder)
    if bibfile != "":
        import_bib(bibfile, libtex_csv)
        construct_keylist(blacklist=blacklist, author_given_keywords=author_given_keywords, input_folder=input_folder, output_name=output_name, original_keywords_txt=original_keywords_txt)
    else:
        construct_keylist(blacklist=blacklist, bibfile=bibfile, author_given_keywords=author_given_keywords, input_folder=input_folder, output_name=output_name, original_keywords_txt=original_keywords_txt)
    return


if __name__ == "__main__":
    generate_keylist(input_folder= "C:/NLPvenv/nlp/input", records="records", bibfile="library", libtex_csv="savedrecs", output_name="keylist", original_keywords_txt="wos_original_tags",
                     author_given_keywords="Author Keywords", Rake_stoppath="C:/NLPvenv/RAKE/data/stoplists/SmartStoplist.txt")
