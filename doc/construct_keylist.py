import re
from collections import OrderedDict
from operator import itemgetter
from anyascii import anyascii

import gensim
import gensim.corpora as corpora
import numpy as np
import pandas as pd
from cleantext import clean, fix_bad_unicode
from fuzzywuzzy import fuzz
from gensim import models
from gensim.parsing.preprocessing import remove_stopwords, strip_short
from gensim.utils import simple_preprocess
from keybert import KeyBERT
from nlp_rake import rake
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pke.unsupervised import TextRank, TopicRank
from pybtex.database.input import bibtex
from pybtex.database import parse_file
from six import iteritems
from spacy import load
from spacy.lang.en.stop_words import STOP_WORDS
from yake import KeywordExtractor

""" APART
- 2023/1/18 Sam Boerlijst
- Academic Pdf - Automated Reference Tagging: Collect keywords from (web of science or google scholar) csv list of titles and abstracts using 7 common NLP algorithms, combine those with author given tags and tags present in bib file and export as csv
"""


""" Vragen:
- klopt l647-651? werkt eval() om string naar corresponderende variabele te verwijzen? in matrix append of n+1 --> filter > 2
"""


# setting columns from sourcefile and location for output
KeylistPath = "input/keylist.csv"
blacklist = ["amount","at http","at https","com","copyright","female","male","nan","number","net","org","parameter","result","species","study","total", "www"]

# when using nltk for the first time you might need to download the stopword list Using 'from nltk import download' and 'download('stopwords')'

#build lists to store tags and scores
taglist = []
scorelist = []

# input sanitization
def do_clean(text : str) -> str:
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
        # Transliterate to closest ASCII representation
        to_ascii=True,
        # Lowercase text
        lower=True,
        # Fully strip line breaks as opposed to only normalizing them
        no_line_breaks=False,
        # Replace all URLs with a special token
        no_urls=False,
        # Replace all email addresses with a special token
        no_emails=False,
        # Replace all phone numbers with a special token
        no_phone_numbers=True,
        # Replace all numbers with a special token
        no_numbers=False,
        # Replace all digits with a special token
        no_digits=False,
        # Replace all currency symbols with a special token
        no_currency_symbols=False,
        # Fully remove punctuation
        no_punct=False,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<ACCOUNT>",
        replace_with_number="<NUMBER>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        # Set to 'de' for German special handling
        lang="en",
    )

## collect keywords from WOS file
def WOS_get_keywords(WOSfile: str, original_keywords_txt: str = "input/wos_original_tags.txt") -> None:
    """
    Retrieves all keywords indexed by Web of Science and cleans any delimiter artifacts.

    Parameters:
    -----------
    WOSfile (str): Path to the csv containing the Web of Science records.

    original_keywords_txt (str): path to the text file in which to store the keywords,

    Return:
    -----------
    None
    """
    print("collecting given tags")
    author_keywords = str(pd.read_csv(WOSfile)["Author Keywords"].to_list())
    plus_keywords = str(pd.read_csv(WOSfile)["Keywords Plus"].to_list())
    # Fix various errors
    author_keywords = do_clean(fix_bad_unicode(author_keywords))
    plus_keywords = do_clean(fix_bad_unicode(plus_keywords))
    # convert to list
    author_keywords = (
        author_keywords.replace("'", "")
        .replace(",", ";")
        .replace("[", "")
        .replace("]", "")
        .replace("{", "")
        .replace("}", "")
        .split("; ")
    )
    plus_keywords = (
        plus_keywords.replace("'", "")
        .replace(",", ";")
        .replace("[", "")
        .replace("]", "")
        .replace("{", "")
        .replace("}", "")
        .split("; ")
    )
    # filter uniques
    author_keywords = list(set(author_keywords))
    plus_keywords = list(set(plus_keywords))
    wos_keywords_original = author_keywords + plus_keywords
    wos_keywords_original = sorted(list(set(wos_keywords_original)))
    wos_original = open(original_keywords_txt, "wb")
    wos_original.write(str(wos_keywords_original).encode("utf-8"))
    wos_original.close()
    return


## collect keywords using bigrams
def bigram_extraction(WOSfile: str, WOScolumn: str, name: str, amount:int) -> None:
    """
    Determine keywords from the given column for each entry in a csv file using the bigram algorithm. 

    Parameters:
    -----------
    WOSfile (str): Path to the csv containing the Web of Science records.

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
    data = pd.read_csv(WOSfile)[WOScolumn].tolist()
    data = str(data).encode(encoding="unicode_escape")
    data = gensim.parsing.preprocessing.remove_stopwords(data)
    doc = strip_short(data, minsize=4)
    # clean input
    doc = doc.replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("<", "").replace(">", "").replace("%", "")
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
            [word for word in simple_preprocess(str(doc)) if word not in stop_words]
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
    df.to_csv("input/bigram_" + name + ".csv", index=False)
    return 


## collect keywords using keybert
def keybert_extraction(WOSfile: str, WOScolumn: str, name: str, amount: int) -> None:
    """
    Determine keywords from the given column for each entry in a csv file using the keybert algorithm. 

    Parameters:
    -----------
    WOSfile (str): Path to the csv containing the Web of Science records.

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
    sourcefile = pd.read_csv(WOSfile)[WOScolumn].tolist()
    data = sourcefile
    data = str(data).encode(encoding="unicode_escape")
    data = remove_stopwords(data)
    doc = strip_short(data)
    # clean input
    do_clean(doc)
    doc = doc.replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("<", "").replace(">", "").replace("%", "")
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
        df.to_csv("input/keybert_" + name + ".csv", index=False)
    return


## collect keywords using RAKE
def rake_extraction(WOSfile: str, WOScolumn: str, name: str, Rake_stoppath: str, amount: int) -> None:
    """
    Determine keywords from the given column for each entry in a csv file using the RAKE algorithm. 

    Parameters:
    -----------
    WOSfile (str): Path to the csv containing the Web of Science records.

    WOScolumn (str): The column name containing the text to analyse.

    name (str): String stating whether titles or abstracts are being analysed. May be equal to either "wos_a" or "wos_t". Additionally this parameter will be used as output name.

    Rake_stoppath (str): Path to the file containing stopwords to use.

    amount (int): Number of keywords to select.

    Return:
    -----------
    None
    """
    if name == "wos_a":
        print("generating rake tags from abstracts")
    elif name == "wos_t":
        print("generating rake tags from titles")
    data = pd.read_csv(WOSfile)[WOScolumn].tolist()
    data = str(data).encode(encoding="unicode_escape")
    data = remove_stopwords(data)
    text = strip_short(data)
    text = do_clean(text)
    text = text.replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("<", "").replace(">", "").replace("%", "")
    sentenceList = rake.split_sentences(text)
    stopwords = rake.load_stop_words(Rake_stoppath)
    stopwordpattern = rake.build_stop_word_regex(Rake_stoppath)
    phraseList = rake.generate_candidate_keywords(
        sentenceList, stopwordpattern, stopwords, max_words_length=3, min_char_length=4
    )
    wordscores = rake.calculate_word_scores(phraseList)
    # output grouped scores
    keywordcandidates = rake.generate_candidate_keyword_scores(
        phraseList, wordscores, min_keyword_frequency=1
    )
    sortedKeywords = sorted(
        iteritems(keywordcandidates), key=itemgetter(1), reverse=True
    )
    df = pd.DataFrame(data=sortedKeywords[0:amount], columns=["ID", "frequency"])
    df.to_csv("input/rake_" + name + ".csv", index=False)
    return


## collect keywords using textrank
def textrank_extraction(WOSfile: str, WOScolumn: str, name: str, amount: int) -> None:
    """
    Determine keywords from the given column for each entry in a csv file using the TextRank algorithm. 

    Parameters:
    -----------
    WOSfile (str): Path to the csv containing the Web of Science records.

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
    data = pd.read_csv(WOSfile)[WOScolumn].tolist()
    data = str(data).encode(encoding="unicode_escape")
    data = remove_stopwords(data)
    text = strip_short(data)
    # clean input
    text = do_clean(text)
    text = text.replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("<", "").replace(">", "").replace("%", "")
    nlp.max_length = len(text) + 100
    extractor = TextRank()
    extractor.candidate_filtering(
        minimum_length=4, maximum_word_number=3, valid_punctuation_marks="."
    )
    # define the set of valid Part-of-Speeches
    pos = {"NOUN", "PROPN", "ADJ"}
    sentences = []
    doc = nlp(text)
    for sent in doc.sents:
        selected_words = []
        for token in sent:
            if token.pos_ in pos and token.is_stop is False:
                selected_words.append(token)
        sentences.append(selected_words)
    keyphrases = pd.DataFrame(columns=["ID", "frequency"])

    class TextRank4Keyword:
        def __init__(self):
            self.d = 0.85  # damping coefficient, usually is .85
            self.min_diff = 1e-5  # convergence threshold
            self.steps = 10  # iteration steps
            self.node_weight = None  # save keywords and its weight

        def set_stopwords(self, stopwords):
            """Set stop words"""
            for word in STOP_WORDS.union(set(stopwords)):
                lexeme = nlp.vocab[word]
                lexeme.is_stop = True

        def sentence_segment(self, doc, candidate_pos, lower):
            """Store those words only in cadidate_pos"""
            sentences = []
            for sent in doc.sents:
                selected_words = []
                for token in sent:
                    # Store words only with cadidate POS tag
                    if token.pos_ in candidate_pos and token.is_stop is False:
                        if lower is True:
                            selected_words.append(token.text.lower())
                        else:
                            selected_words.append(token.text)
                sentences.append(selected_words)
            return sentences

        def get_vocab(self, sentences):
            """Get all tokens"""
            vocab = OrderedDict()
            i = 0
            for sentence in sentences:
                for word in sentence:
                    if word not in vocab:
                        vocab[word] = i
                        i += 1
            return vocab

        def get_token_pairs(self, window_size, sentences):
            """Build token_pairs from windows in sentences"""
            token_pairs = list()
            for sentence in sentences:
                for i, word in enumerate(sentence):
                    for j in range(i + 1, i + window_size):
                        if j >= len(sentence):
                            break
                        pair = (word, sentence[j])
                        if pair not in token_pairs:
                            token_pairs.append(pair)
            return token_pairs

        def symmetrize(self, a):
            return a + a.T - np.diag(a.diagonal())

        def get_matrix(self, vocab, token_pairs):
            """Get normalized matrix"""
            # Build matrix
            vocab_size = len(vocab)
            g = np.zeros((vocab_size, vocab_size), dtype="float")
            for word1, word2 in token_pairs:
                i, j = vocab[word1], vocab[word2]
                g[i][j] = 1
            # Get Symmeric matrix
            g = self.symmetrize(g)
            # Normalize matrix by column
            norm = np.sum(g, axis=0)
            g_norm = np.divide(
                g, norm, where=norm != 0
            )  # this is ignore the 0 element in norm
            return g_norm

        def get_keywords(self, number=30):
            """Print top number keywords"""
            node_weight = OrderedDict(
                sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True)
            )
            for i, (key, value) in enumerate(node_weight.items()):
                keyphrases.loc[len(keyphrases.index)] = [key, value]
                if i > number:
                    break

        def analyze(
            self,
            text,
            candidate_pos=["NOUN", "PROPN"],
            window_size=4,
            lower=False,
            stopwords=list(),
        ):
            """Main function to analyze text"""
            # Set stop words
            self.set_stopwords(stopwords)
            # Pare text by spaCy
            doc = nlp(text)
            # Filter sentences
            sentences = self.sentence_segment(
                doc, candidate_pos, lower
            )  # list of list of words
            # Build vocabulary
            vocab = self.get_vocab(sentences)
            # Get token_pairs from windows
            token_pairs = self.get_token_pairs(window_size, sentences)
            # Get normalized matrix
            g = self.get_matrix(vocab, token_pairs)
            # Initionlization for weight(pagerank value)
            pr = np.array([1] * len(vocab))
            # Iteration
            previous_pr = 0
            for epoch in range(self.steps):
                pr = (1 - self.d) + self.d * np.dot(g, pr)
                if abs(previous_pr - sum(pr)) < self.min_diff:
                    break
                else:
                    previous_pr = sum(pr)
            # Get weight for each node
            node_weight = dict()
            for word, index in vocab.items():
                node_weight[word] = pr[index]
            self.node_weight = node_weight

    tr4w = TextRank4Keyword()
    tr4w.analyze(text, candidate_pos=["NOUN", "PROPN"], window_size=4, lower=False)
    tr4w.get_keywords(amount)
    keyphrases.to_csv("input/TextR_" + name + ".csv", index=False)
    return


## collect keywords using topicrank
def topicrank_extraction(WOSfile: str, WOScolumn: str, name: str, amount: int) -> None:
    """
    Determine keywords from the given column for each entry in a csv file using the TopicRank algorithm. 

    Parameters:
    -----------
    WOSfile (str): Path to the csv containing the Web of Science records.

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
    data = pd.read_csv(WOSfile)[WOScolumn].tolist()
    data = str(data).encode(encoding="unicode_escape")
    data = remove_stopwords(data)
    text = strip_short(data)
    # clean input
    do_clean(text)
    text = text.replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("<", "").replace(">", "").replace("%", "")
    # define the set of valid Part-of-Speeches
    pos = {"NOUN", "PROPN", "ADJ"}
    sentences = []
    # 1. create a TopicRank extractor.
    nlp.max_length = len(text) + 100
    extractor = TopicRank()
    
    # 2. load the content of the document.
    extractor.load_document(input=text, language="en", normalization=None, spacy_model= nlp)
    # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
    # and adjectives (i.e. `(Noun|Adj)*`)
    extractor.candidate_selection()
    extractor.candidate_filtering(
        minimum_length=4, maximum_word_number=2, valid_punctuation_marks="."
    )
    # candidate weighting, in the case of TopicRank: using a random walk algorithm
    extractor.candidate_weighting()
    # N-best selection, keyphrases contains the 10 highest scored candidates as (keyphrase, score) tuples
    keyphrases = extractor.get_n_best(n=amount, redundancy_removal=True)
    for doc in keyphrases:
        df = pd.DataFrame(data=keyphrases, columns=["ID", "frequency"])
        df.to_csv("input/topicR_" + name + ".csv", index=False)
    return


## collect keywords using tf-idf
def tf_idf_extraction(WOSfile: str, WOScolumn: str, name: str) -> None:
    """
    Determine keywords from the given column for each entry in a csv file using the TF-IDF algorithm. 

    Parameters:
    -----------
    WOSfile (str): Path to the csv containing the Web of Science records.

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
    data = pd.read_csv(WOSfile)[WOScolumn].tolist()
    data = str(data).encode(encoding="unicode_escape")
    data = remove_stopwords(data)
    doc = strip_short(data)
    doc = do_clean(doc)
    doc = doc.replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("<", "").replace(">", "").replace("%", "")
    doc = list(pd.Series(data))
    # Create the Tokens, Dictionary and Corpus
    text_tokens = [[tok for tok in doc.split(",")] for doc in doc]
    mydict = corpora.Dictionary([simple_preprocess(line) for line in doc])
    corpus = [mydict.doc2bow(simple_preprocess(line)) for line in doc]
    # Create the TF-IDF model
    tfidf = models.TfidfModel(corpus, smartirs="ntc")
    for doc in tfidf[corpus]:
        locals()["data_{0}".format(doc)] = pd.DataFrame(
            data=[[mydict[id], np.around(freq, decimals=2)] for id, freq in doc],
            columns=["ID", "frequency"],
        )
        locals()["data_{0}".format(doc)] = locals()["data_{0}".format(doc)].sort_values(
            by="frequency", ascending=False
        )
        locals()["data_{0}".format(doc)] = locals()["data_{0}".format(doc)][
            locals()["data_{0}".format(doc)]["frequency"] > 0.01
        ]
        locals()["data_{0}".format(doc)].to_csv(
            "input/tf-idf_" + name + ".csv", index=False
        )
    return


## collect keywords using yake
def yake_extraction(WOSfile: str, WOScolumn: str, name: str, amount: int) -> None:
    """
    Determine keywords from the given column for each entry in a csv file using the YAKE algorithm. 

    Parameters:
    -----------
    WOSfile (str): Path to the csv containing the Web of Science records.

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
    data = pd.read_csv(WOSfile)[WOScolumn].tolist()
    data = str(data).encode(encoding="unicode_escape")
    data = remove_stopwords(data)
    text = strip_short(data)
    text = do_clean(text)
    text = text.replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("<", "").replace(">", "").replace("%", "")
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
    df.to_csv("input/yake_" + name + ".csv", index=False)
    return 


## import .bib file
def import_bib(bibfile: str, libtex_csv: str) -> None:
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
    bib_data = parse_file(bibfile)
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
                #remove special characters from string
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
    dataset.to_csv(libtex_csv, index=False)
    return 


##collect keywords using all algorithms
def extract_tags(WOSfile, column, name, Rake_stoppath, amount) -> None:
    """
    Determine keywords from the given column for each entry in a csv file using all seven algorithms: bigram, keybert, rake, textrank, topicrank, tf-idf and yake. 

    Parameters:
    -----------
    WOSfile (str): Path to the csv containing the Web of Science records.

    WOScolumn (str): The column name containing the text to analyse.

    name (str): String stating whether titles or abstracts are being analysed. May be equal to either "wos_a" or "wos_t". Additionally this parameter will be used as output name.

    Rake_stoppath (str): Path to the file containing stopwords to use.

    amount (int): Number of keywords to select.

    Return:
    -----------
    None
    """
    bigram_extraction(WOSfile, column, name, amount)
    keybert_extraction(WOSfile, column, name, amount)
    rake_extraction(WOSfile, column, name, Rake_stoppath, amount)
    textrank_extraction(WOSfile, column, name, amount)
    topicrank_extraction(WOSfile, column, name, amount)
    tf_idf_extraction(WOSfile, column, name)
    yake_extraction(WOSfile, column, name, amount)
    return


## construct keylist
def construct_keylist(blacklist:list=blacklist, libtex_csv:str = "input/savedrecs.csv"):
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
    bigram_a = pd.read_csv("input/bigram_wos_a.csv")["ID"].tolist()
    bigram_t = pd.read_csv("input/bigram_wos_t.csv")["ID"].tolist()
    keybert_a = pd.read_csv("input/keybert_wos_a.csv")["ID"].tolist()
    keybert_t = pd.read_csv("input/keybert_wos_t.csv")["ID"].tolist()
    TextR_a = pd.read_csv("input/TextR_wos_a.csv")["ID"].tolist()
    TextR_t = pd.read_csv("input/TextR_wos_t.csv")["ID"].tolist()
    tf_idf_a = pd.read_csv("input/tf-idf_wos_a.csv")["ID"].tolist()
    tf_idf_t = pd.read_csv("input/tf-idf_wos_t.csv")["ID"].tolist()
    topicR_a = pd.read_csv("input/topicR_wos_a.csv")["ID"].tolist()
    topicR_t = pd.read_csv("input/topicR_wos_t.csv")["ID"].tolist()
    yake_a = pd.read_csv("input/yake_wos_a.csv")["ID"].tolist()
    yake_t = pd.read_csv("input/yake_wos_t.csv")["ID"].tolist()
    bib_original = pd.read_csv(libtex_csv)["keywords"].tolist()
    wos_original = open("input/wos_original_tags.txt", "r").readlines()
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


    #combine lists
    for item in comparelist:
        cor = 'cor_' + item
        test = eval(cor)
        for value in test:
            if value in taglist:
                for i in taglist:
                    if i == value:
                        scorelist[taglist.index(i)] = scorelist[taglist.index(i)]+1
            else:
                taglist.append(value)
                scorelist.append(1)
    tagmatrix = []

    #filter items that occur in between 2-4 of the lists (common, but not too common)
    for i in range(len(scorelist)):
        if scorelist[i] > 1 and scorelist[i] < 5 :
            tagmatrix.append([taglist[i],scorelist[i]])
            newlist.append(taglist[i])


    #append author given keywords and user given keywords
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
    #fuzzy search blacklist
    filterblacklist(newlist)
    finallist = str(sorted(list(set(finallist)))).replace("'", "").split(",")
    #direct search blacklist
    finallist = [word for word in finallist if not blacklist1.search(word) and len(word)>2]
    # write to csv
    finallist = pd.DataFrame(finallist, columns=["ID"])
    finallist = (
        finallist.sort_values(by="ID", axis=0, ascending=True)
        .drop_duplicates("ID", keep="last")
        .reset_index(drop=True)
    )
    finallist.to_csv(KeylistPath, index=False)
    return


### complete keylist routine
def generate_keylist(WOSfile="input/WOSselect.csv", titlecolumn="Article Title", abstactcolumn="Abstract", bibfile="input/library.bib", libtex_csv="input/savedrecs.csv", blacklist=blacklist, amount=40, Rake_stoppath="C:/NLPvenv/RAKE/data/stoplists/SmartStoplist.txt"):
    """
    Gerenates a keyword list using the Web of Science records by 1) extracting indexed keywords 2) filtering article titles for keywords using all seven algorithms,  3) filtering article abstracts for keywords using all seven algorithms, 4) extracting keywords present in a bib file and 5) filtering for unique values excluding keywords present in the blacklist.
    All parameters but the WOS file path and bibfile path have default values.

    Parameters:
    -----------
    WOSfile (str): Path to the csv containing the Web of Science records.

    titlecolumn (str): The column name containing the titles to analyse.

    abstactcolumn (str): The column name containing the abstracts to analyse.

    name (str): String stating whether titles or abstracts are being analysed. May be equal to either "wos_a" or "wos_t". Additionally this parameter will be used as output name.

    bibfile (str): Path to the bib file to be extracted.

    libtex_csv (str): Path to the csv file to write the output to.

    blacklist (list): list of strings to exclude.

    amount (int): Number of keywords to select.

    Return:
    -----------
    None
    """
    WOS_get_keywords(WOSfile)
    extract_tags(name = "wos_t", column = titlecolumn, WOSfile = WOSfile, Rake_stoppath = Rake_stoppath, amount = amount)
    extract_tags(name = "wos_a", column = abstactcolumn, WOSfile = WOSfile, Rake_stoppath = Rake_stoppath, amount = amount)
    import_bib(bibfile, libtex_csv)
    construct_keylist(blacklist)
    return


if __name__ == "__main__":
    generate_keylist(WOSfile = "input/WOSselect1.csv", bibfile = "input/pc_Library_1-5-2023.bib")