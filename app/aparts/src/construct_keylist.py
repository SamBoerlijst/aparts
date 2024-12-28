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

taglist = []
scorelist = []


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
    author_keywords = do_clean(author_keywords)
    author_keywords = clean_keywords(author_keywords)
    author_keywords = list(set(author_keywords))
    wos_keywords_original = sorted(list(set(author_keywords)))

    with open(original_keywords_txt_path, "w", encoding="utf-8") as file:
        file.write(str(wos_keywords_original))

    print("Keywords collected and stored in", original_keywords_txt_path)

    return


def bigram_extraction(records: str, WOScolumn: str, name: str, amount: int, input_folder: str) -> None:
    """
    Determine keywords from the given column for each entry in a csv file using the bigram algorithm. 

    Note: When using nltk for the first time you might need to download the stopword list Using 'from nltk import download' and 'download('stopwords')'
    
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

    doc = doc.replace("[", "").replace("]", "").replace("{", "").replace(
        "}", "").replace("<", "").replace(">", "").replace("%", "")
    doc = do_clean(doc).split(",")

    stop_words = stopwords.words("english")
    stop_words.extend(["from", "subject", "re", "edu", "use"])

    def sent_to_words(sentences):
        """Tokenize sentences."""
        for sentence in sentences:
            yield (
                gensim.utils.simple_preprocess(str(sentence), deacc=True)
            )

    monogram = list(sent_to_words(doc))
    bigram = gensim.models.Phrases(
        monogram, min_count=2, threshold=2, delimiter=" "
    )
    trigram = gensim.models.Phrases(bigram[monogram], threshold=2)

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

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


    monogram_nostops = remove_stopwords(monogram)

    monograms_bigrams = make_bigrams(monogram_nostops)

    id2word = corpora.Dictionary(monograms_bigrams)

    texts = monograms_bigrams

    corpus = [id2word.doc2bow(text) for text in texts]

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
    print(
        "\nPerplexity: ", lda_model.log_perplexity(corpus)
    )
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

    keyphrases = pd.DataFrame({'n-gram': top_keywords, 'score': [scores[kw] for kw in top_keywords]})
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

    text_tokens = [[tok for tok in doc.split(",")] for doc in doc]
    mydict = corpora.Dictionary([simple_preprocess(line) for line in doc])
    corpus = [mydict.doc2bow(simple_preprocess(line)) for line in doc]

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
            elif str(name) == "doi":
                value = list(entry.fields.values())[val]
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


def construct_keylist(blacklist: list = blacklist, libtex_csv: str = "", bibfile: str = "", output_name: str = "keyword_list", input_folder: str = "", author_given_keywords: str = "", original_keywords_txt: str = ""):
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
    def read_files(file_list: list, method_list: list) -> None:
        """Read data from CSV files based on the provided file_list and store them in lists with names derived from the method_list."""
        for i in range(len(method_list)):
            method = method_list[i]
            file = file_list[i]
            file_path = f"{input_folder}/{file.split('_')[0]}_wos_{file[-1]}.csv"
            column_name = 'ID' if 'TextR' not in file else 'n-gram'
            globals()[method] = pd.read_csv(file_path)[column_name].tolist()
        return

    def generate_empty_lists(method_list: list, prefix_list: list = None):
        """Generate empty lists with the given method names with optional prefixes."""
        for item in method_list:
            if prefix_list:
                for prefix in prefix_list:
                    globals()[f"{prefix}{item}"] = []
            else:
                globals()[item] = []

    def clean_list(item_list: list) -> list:

        """Clean up a list by removing unwanted characters and formatting."""
        null_chars = ['"', "'", "[", "]", "<", ">", "{", "}", "*", "&", ":", "\\"]
        space_chars = [",", "/", ""]
        cleaned_list = []

        for item in item_list:
            cleaned_item = "".join(
                " " if char in space_chars else char for char in item if char not in null_chars
            )
            cleaned_list.append(cleaned_item)
        return cleaned_list

    def asciify(source_list: list) -> None:
        """Convert items in the source list to ASCII characters."""
        destination_list = []
        for item in source_list:
            item = anyascii(str(item))
            destination_list.append(item)
        return destination_list

    def filter_by_length(item_list: list) -> None:
        """Filter out items in a list based on their length."""
        for item in item_list:
            test = eval(item)
            for i in range(len(test)):
                if isinstance(test[i], str) and len(test[i]) < 3:
                        eval(item)[i] = ""
                        print("removed", test[i])
        return

    def filter_lists_by_stem(method_list: list) -> None:
        """Apply stemming to items in the given method_list."""
        ps = PorterStemmer()
        for item in method_list:
            test = eval(item)
            stem = 'stem_' + item
            cor = 'cor_' + item
            stemcor = 'stemcor_' + item

            eval(stem).extend(ps.stem(word) for word in test if isinstance(word, str))
            eval(stemcor).extend(test[i] for i in range(
                len(eval(stem))) if eval(stem)[i] not in eval(stemcor))
            eval(cor).extend(eval(item)[i] for i in range(
                len(eval(stem))) if eval(stem)[i] not in eval(stemcor))
        return

    def filterblacklist(source_list: list, fuzz_ratio: int = 80) -> None:
        """Filter out items from the source list based on a fuzziness ratio."""
        target_list = []
        for item in source_list:
            if fuzz.partial_ratio(item, blacklist) < fuzz_ratio:
                target_list.append(item)
        return target_list

    def merge_lists(method_list: list) -> tuple[list, list]:
        """Merge lists from the given method_list and count occurrences."""
        tag_list = []
        score_list = []
        for item in method_list:
            cor = 'cor_' + item
            test = eval(cor)

            for value in test:
                if value in tag_list:
                    index = tag_list.index(value)
                    score_list[index] += 1
                else:
                    tag_list.append(value)
                    score_list.append(1)
        return tag_list, score_list

    def filter_items_by_overlap(tag_list: list, score_list: list, minimum: int, maximum: int) -> dict:
        """Filter items based on their occurrence counts within a specified range."""
        tag_matrix = {}
        output_list = []
        for i in range(len(score_list)):
            if score_list[i] >= minimum and score_list[i] <= maximum:
                j = len(tag_matrix)
                tag_matrix[j] = {tag_list[i]: score_list[i]}
                output_list.append(tag_list[i])
        return output_list, tag_matrix

    newlist = []
    finallist = []
    bib_original = []
    wos_original = []
    prefixes = ["cor_", "stem_", "stemcor_"]
    filelist = [
        "bigram_a",
        "bigram_t",
        "keybert_a",
        "keybert_t",
        "TextR_a",
        "TextR_t",
        "tf-idf_a",
        "tf-idf_t",
        "topicR_a",
        "topicR_t",
        "yake_a",
        "yake_t",
    ]
    methodlist = [
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

    print("constructing keyword list from combined output")
    read_files(filelist, methodlist)
    generate_empty_lists(methodlist, prefixes)
    
    if bibfile:
        bib_original = pd.read_csv(libtex_csv)["keywords"].tolist()

    if author_given_keywords:
        wos_original = open(
            f"{input_folder}/{original_keywords_txt}.txt", "rb").readlines()

    #filter_by_length(methodlist)
    filter_lists_by_stem(methodlist)
    taglist, scorelist = merge_lists(methodlist)
    newlist, x = filter_items_by_overlap(taglist, scorelist, 2, 4)
    woslist = asciify(wos_original)
    biblist = asciify(bib_original)
    newlist = newlist + woslist + biblist
    newlist = clean_list(newlist)
    finallist = filterblacklist(newlist)
    finallist = [word for word in finallist if len(word) > 2]
    finallist = pd.DataFrame(finallist, columns=["ID"])
    finallist = (
        finallist.sort_values(by="ID", axis=0, ascending=True)
        .drop_duplicates("ID", keep="last")
        .reset_index(drop=True)
    )
    finallist.to_csv(f"{input_folder}/{output_name}.csv", index=False)
    return finallist


# complete keylist routine
def generate_keylist(input_folder="", records="", titlecolumn="Article Title", abstactcolumn="Abstract", bibfile="", libtex_csv="", output_name="keyword_list", original_keywords_txt="", blacklist=blacklist, amount=50, author_given_keywords="", Rake_stoppath=""):
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
    generate_keylist(input_folder= "C:/NLPvenv/nlp/input", records="savedrecs_lianas", bibfile="", libtex_csv="savedrecs_lianas_out", output_name="keyword_list", original_keywords_txt="wos_original_tags",
                     author_given_keywords="Author Keywords", Rake_stoppath="C:/NLPvenv/RAKE/data/stoplists/SmartStoplist.txt")
