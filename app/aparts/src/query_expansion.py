import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from nltk.corpus import wordnet
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from aparts.src.deduplication import count_tag_occurrence
from aparts.src.subsampling import (assign_group, generate_binary_item_matrix,
                                    transform_dataframe)
from itertools import product


def import_df(filepath:str)->pd.DataFrame:
    """import csv or xlsx file based on filepath"""
    file_extension = filepath.split('.')[-1]
    if file_extension == 'csv':
        df = pd.read_csv(filepath)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(
            f"Unsupported file format for {filepath}. Use DataFrame, CSV or Excel files.")
    return df


def pca_biplot(X: np.ndarray, y, targets: list, features: list):
    """
    Description:
    ------------
    Generate a biplot for Principal Component Analysis (PCA) results. This function takes the results of a PCA, including the scaled and reduced data, and creates a biplot for visualizing the relationships between samples and features in a two-dimensional space.

    Parameters:
    -----------
    X (np.ndarray): The input data matrix.

    y (array-like): Class labels or group assignments for each sample.

    targets (list): List of target names or labels corresponding to each class.

    features (list): List of feature names corresponding to each dimension in the input data.
    """
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2).fit(X_scaled)
    X_reduced = pca.transform(X_scaled)

    scores = X_reduced[:, :2]
    loadings = pca.components_[:2].T
    pvars = pca.explained_variance_ratio_[:2] * 100

    k = 10

    tops = (loadings * pvars).sum(axis=1).argsort()[-k:]
    arrows = loadings[tops]

    arrows /= np.sqrt((arrows ** 2).sum(axis=0))

    plt.figure(figsize=(5, 5))
    for i, name in enumerate(targets):
        plt.scatter(*zip(*scores[y == y[i]]), label=name, s=8, alpha=0.5)
    plt.legend(title='Class')

    width = -0.005 * np.min([np.subtract(*plt.xlim()),
                            np.subtract(*plt.ylim())])
    for i, arrow in zip(tops, arrows):
        plt.arrow(0, 0, *arrow, color='k', alpha=0.75, width=width, ec='none',
                  length_includes_head=True)
        plt.text(*(arrow * 1.15), features[i], ha='center', va='center')

    for i, axis in enumerate('xy'):
        getattr(plt, f'{axis}ticks')([])
        getattr(plt, f'{axis}label')(f'PC{i + 1} ({pvars[i]:.2f}%)')
    return


def pca_tags(CSV: str, variables: str, id: str, tag_length: int, number_of_records: int):
    """
    Description
    -----------
    Perform Principal Component Analysis (PCA) on tag data and generate a biplot for visualization. This function takes a CSV file containing tag data, performs PCA on a binary item matrix derived from the data, and produces a biplot to visualize the relationships between tags and records in a reduced-dimensional space.

    Parameters
    ----------
    CSV (str): Path to the CSV file containing tag data.

    variables (str): Column name in the CSV file containing tag variables.

    id (str): Column name in the CSV file containing record identifiers.

    tag_length (int): Length of the tag variables.

    number_of_records (int): Number of records to consider for PCA.
    """
    matrix, id_list = generate_binary_item_matrix(
        CSV, variables, id, tag_length, number_of_records)
    counts = count_tag_occurrence(matrix)
    column = [item[0] for item in counts]
    groups = assign_group(matrix, column)
    group_list = set(groups)
    X, y, targets, features = transform_dataframe(matrix, groups, group_list)
    y = np.asarray(y)
    pca_biplot(X, y, targets, features)
    return


def merge_words_by_stem_and_wildcards(text: str, min_similarity=0.6, min_prefix_length=3) -> list:
    """
    This function takes an input text containing comma-separated words and merges similar words by stem with wildcards
    based on a similarity threshold.

    Parameters:
    -----------
    text (str): The input text containing comma-separated words.

    min_similarity (float, optional): The minimum similarity threshold to consider for merging words (default is 0.6).

    min_prefix_length (int, optional): The minimum prefix length to consider for similarity (default is 3).

    Returns:
    -----------
    merged_words (list of str): A list of merged words where similar words are merged by stem with wildcards.
    """
    def calculate_similarity(word1, word2, min_prefix_length=min_prefix_length):
        common_prefix = os.path.commonprefix([word1, word2])
        if len(common_prefix) < min_prefix_length:
            return 0.0
        return len(common_prefix) / max(len(word1), len(word2))

    # Tokenize the input text into words
    words = [word.strip() for word in text.split(", ")]
    merged_words = []

    for word1 in words:
        if word1 not in merged_words:
            similar_words = [word1]
            for word2 in words:
                if word2 != word1 and word2 not in similar_words:
                    if calculate_similarity(word1, word2) >= min_similarity:
                        similar_words.append(word2)
            if len(similar_words) > 1:
                merged_word = similar_words[0]
                for i in range(1, len(similar_words)):
                    j = 0
                    while j < len(merged_word) and j < len(similar_words[i]) and merged_word[j] == similar_words[i][j]:
                        j += 1
                    # Check if there's a space before the asterisk in the merged word
                    if j > 0 and merged_word[j - 1] == ' ':
                        # Append all similar words separately
                        merged_words.extend(similar_words)
                        break
                    else:
                        merged_word = merged_word[:j] + '*'
                else:
                    merged_words.append(merged_word)
            else:
                merged_words.append(word1)
    merged_words = set(merged_words)
    return merged_words


def group_related_words(word_list: list[str]):
    """
    Group words based on WordNet synsets without expanding the list with additional synonyms.

    Parameters:
    -----------
    word_list (list): A list of words to be grouped.

    Returns:
    -----------
    grouped_words (list): A list with their elements grouped by their synsets.
    """
    synsets = {}

    for word in word_list:
        synsets[word] = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synsets[word].add(lemma.name())

    grouped_words = []
    processed = set()

    for word in word_list:
        if word not in processed:
            related = synsets[word]
            group = [w for w in word_list if w in related and w != word]
            group.append(word)
            grouped_words.extend(group)
            processed.update(group)

    return list(grouped_words)


def group_fuzzy_words(word_list: list[str], threshold: int = 80):
    """
    Group words based on fuzzy matches.

    Parameters:
    -----------
    word_list (list): A list of words to be grouped.

    threshold (int): fuzz ratio

    Returns:
    -----------
    grouped_words (list): A list with their elements grouped by their similarity.
    """
    grouped_words = []
    processed = set()

    for word in word_list:
        if word not in processed:
            group = [w for w in word_list if fuzz.ratio(
                word, w) >= threshold and w != word]
            group.append(word)
            grouped_words.extend(group)
            processed.update(group)

    return list(grouped_words)


def expand_query_with_tag_similarity(query: str, tags: set, threshold: float = 0.2, top_k: int = 1, batch_size: int = 20):
    """
    Expand a query based on tag similarity.

    Parameters:
    -----------
    query (str): The original query to be expanded.

    tags (list or set): A list or set of tags used for similarity comparison.

    threshold (float, optional): The similarity threshold to consider for expansion (default is 0.2).

    top_k (int, optional): The maximum number of terms to add to the query (default is 1).

    Returns:
    -----------
    expanded_query (list): The expanded query with similar tags added.
    """
    if isinstance(tags, set):
        tags = list(tags)

    vectorizer = TfidfVectorizer()
    num_batches = len(tags) // batch_size + (len(tags) % batch_size > 0)
    expanded_terms = []

    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        batch_tags = tags[batch_start:batch_end]

        tag_matrix = vectorizer.fit_transform(batch_tags)
        query_vec = vectorizer.transform([query])
        query_vec = query_vec.toarray()
        tag_similarity = cosine_similarity(query_vec, tag_matrix)

        if tag_similarity.shape[0] == 0:
            return [query]

        tag_similarity = tag_similarity[0]

        sorted_indices = np.argsort(tag_similarity)[::-1]
        sorted_terms = [batch_tags[i] for i in sorted_indices]

        expanded_terms.extend(
            [term for term in sorted_terms if tag_similarity[batch_tags.index(term)] > threshold][:top_k])

    return expanded_terms


def group_synonyms(words, fuzzy_threshold=""):
    """
    Description:
    ------------
    Group synonyms within a list of words to create logical combinations. This function utilizes WordNet for synonym extraction and allows for optional fuzzy matching to select words with similar meanings. The result is a list of grouped synonyms and individual words.

    Parameters:
    -----------
    words (list of str): List of words to be grouped based on synonyms.

    fuzzy_threshold (str, optional): The threshold for fuzzy matching. If supplied, words with similarity above this threshold are grouped together. Defaults to an empty string, indicating no fuzzy matching.

    Return:
    -------
    grouped_list (list of str): List of grouped synonyms, where each group is represented as a string. Individual words without synonyms are also included in the list.
    """
    grouped_list = []
    words_without_synonyms = []

    def get_synsets(word):
        return [synset for synset in wordnet.synsets(word) if ' ' not in word]

    for i, word1 in enumerate(words):
        synsets1 = get_synsets(word1)
        if not synsets1:
            words_without_synonyms.append(word1)
            continue

        for j in range(i + 1, len(words)):
            word2 = words[j]
            synsets2 = get_synsets(word2)
            if not synsets2:
                words_without_synonyms.append(word2)
                continue

            common_synonyms = set()
            for synset1 in synsets1:
                for synset2 in synsets2:
                    common_synonyms.update(
                        set(synset1.lemma_names()).intersection(synset2.lemma_names()))

            if common_synonyms:
                key = f"({word1} OR {word2})"
                grouped_list.append(key)
            elif fuzzy_threshold:  # Check for fuzzy matches if treshold is supplied
                similarity = fuzz.token_sort_ratio(word1, word2)
                if similarity >= fuzzy_threshold:
                    key = f"({word1} OR {word2})"
                    grouped_list.append(key)

    words_without_synonyms = list(set(words_without_synonyms))
    grouped_list = grouped_list + words_without_synonyms
    return grouped_list


def similarity_feedback(original_query: str = "Culex pipiens AND population dynamics", keyword_column: str = 'keywords', threshold: float = 0.2, top_k: int = 3, input_path: str = "", ingroup: pd.DataFrame = "") -> list[str]:
    """
    Expand a query based on tag co-occurrence in the source csv.

    Parameters:
    -----------
    query (str): The original query to be expanded.

    tags (list or set): A list or set of tags used for similarity comparison.

    tag_matrix (sparse matrix): A TF-IDF matrix representing tag vectors.

    threshold (float, optional): The similarity threshold to consider for expansion (default is 0.2).

    top_k (int, optional): The maximum number of terms to add to the query (default is 1).

    Returns:
    -----------
    expanded_query (str): The expanded query with similar tags added.
    """
    if input_path:
        df = pd.read_csv(input_path)
    else:
        df = ingroup
    df = df.dropna(subset=[keyword_column])
    tags = set(df[keyword_column].tolist())
    vectorizer = CountVectorizer()
    vectorizer.fit(tags)
    expanded_query = expand_query_with_tag_similarity(
        original_query, tags, threshold, top_k)
    return expanded_query


def pseudo_relevance_feedback(original_query: str, n_tags: int = 10, n_articles: int = 20, print_weights: bool = False, input_file: str = None, ingroup: pd.DataFrame = None, outgroup: pd.DataFrame = None, batch_size: int = 20) -> list[str]:
    """
    Perform pseudo-relevance feedback on a collection of documents using the Rocchio algorithm on TF-IDF weights.

    Parameters:
    -----------

    original_query (str): The original query string for retrieval.

    n_tags (int): Number of top relevant tags to keep for query expansion.

    n_articles (int): Number of top relevant articles to display.

    input_file (str): Path to the CSV file containing the document items (title and abstract) to scan.

    ingroup (pd.DataFrame): Optional, Data containing the items that should be represented.

    outgroup (pd.DataFrame):  Optional, Data containing the items that should be represented.

    Returns:
    -----------
    updated_query_str (str): The updated query string after pseudo-relevance feedback.
    """
    def load_data(input_file, trainingset):
        sourcefile = pd.read_csv(input_file)
        matches, non_matches = trainingset
        length = sourcefile.shape[0]
        ingroup_records = int((matches / 100) * length)
        outgroup_records = int((non_matches / 100) * length)
        ingroup = sourcefile.head(ingroup_records).copy()
        outgroup = sourcefile.tail(outgroup_records).copy()
        df = pd.concat([ingroup, outgroup], ignore_index=True)
        return df, ingroup, outgroup

    def process_batch(tfidf_vectorizer, updated_query_array, batch_df):
        batch_df = batch_df.dropna(subset=['Keywords'])

        if not batch_df.empty:
            tfidf_matrix_batch = tfidf_vectorizer.transform(
                batch_df['Keywords'])
            cosine_similarities_batch = cosine_similarity(
                updated_query_array, tfidf_matrix_batch)
            batch_df.loc[:, 'cosine_similarity'] = cosine_similarities_batch[0].copy()

        return batch_df

    def calculate_updated_query(tfidf_vectorizer: any, original_query: str, ingroup: pd.DataFrame, outgroup: pd.DataFrame):
        alpha = 1.0
        beta = 0.75
        gamma = 0.25

        tfidf_matrix = tfidf_vectorizer.fit_transform(
            pd.concat([ingroup, outgroup], ignore_index=True)['Keywords'])
        expanded_query_vector = tfidf_vectorizer.transform([original_query])
        updated_query = alpha * expanded_query_vector + beta * \
            np.mean(tfidf_matrix[ingroup.index, :], axis=0) - \
            gamma * np.mean(tfidf_matrix[outgroup.index, :], axis=0)
        updated_query_array = np.asarray(updated_query)

        return updated_query_array

    def get_top_relevant_tags(tfidf_vectorizer, updated_query_array, n_tags=5, print_weights=False):
        updated_query_tags = tfidf_vectorizer.inverse_transform(updated_query_array)[
            0]
        relevant_tags = [(tag, weight) for tag, weight in zip(
            updated_query_tags, updated_query_array[0])]
        relevant_tags.sort(key=lambda x: x[1], reverse=True)
        top_relevant_tags = [tag for tag, _ in relevant_tags][:n_tags]

        if print_weights:
            print("Relevant Tags with TF-IDF Weights:")
            for tag, weight in relevant_tags[:n_tags]:
                print(f"{tag}: {weight}")

        return top_relevant_tags

    def batch_process(df, tfidf_vectorizer, updated_query_array, batch_size=100):
        batch_start = 0
        df_new = pd.DataFrame()
        while batch_start < df.shape[0]:
            batch_end = min(batch_start + batch_size, df.shape[0])
            batch_df = df.iloc[batch_start:batch_end].copy()
            batch_df_new = process_batch(
                tfidf_vectorizer, updated_query_array, batch_df)
            df_new = pd.concat([df_new, batch_df_new])
            batch_start += batch_size
        return df_new

    if input_file and ingroup == None:
        raise ValueError(
            "Please supply either input file and trainingset or ingroup and outgroup")

    df = pd.concat([ingroup, outgroup], ignore_index=True)

    tfidf_vectorizer = TfidfVectorizer()
    updated_query_array = calculate_updated_query(
        tfidf_vectorizer, original_query, ingroup, outgroup)
    top_relevant_tags = get_top_relevant_tags(
        tfidf_vectorizer, updated_query_array, n_tags, print_weights)
    df = batch_process(df, tfidf_vectorizer, updated_query_array, batch_size)
    df = df.sort_values(by='cosine_similarity', ascending=False)
    top_articles = df.head(n_articles)

    return top_relevant_tags, top_articles


def count_title_matches(file1: pd.DataFrame, file2: pd.DataFrame, file1_column: str, file2_column: str, show_missing: bool = False):
    """
    Count the number of matching titles between two files.

    Parameters:
    -----------
    file1 (pd.Dataframe): Dataframe of the first file.

    file2 (pd.DataFrame): Dataframe of the second file.

    file1_column (str): Column containing titles in the first file.

    file2_column (str): Column containing titles in the second file.

    show_missing (bool, optional): Whether to display titles that are in the first file but not in the second file (default is False).

    Returns:
    -----------
    None
    """
    if type(file1) == pd.DataFrame:
        df1 = file1
    elif type(file1) == str:
        file1_extension = file1.split('.')[-1]
        if file1_extension == 'csv':
            df1 = pd.read_csv(file1)
        elif file1_extension in ['xls', 'xlsx']:
            df1 = pd.read_excel(file1)
        else:
            raise ValueError(
                f"Unsupported file format for {file1}. Use dataframe or path to CSV or Excel files.")
    if type(file2) == pd.DataFrame:
        df2 = file2
    elif type(file2) == str:
        file2_extension = file2.split('.')[-1]
        if file2_extension == 'csv':
            df2 = pd.read_csv(file2)
        elif file2_extension in ['xls', 'xlsx']:
            df2 = pd.read_excel(file2)
        else:
            raise ValueError(
                f"Unsupported file format for {file2}. Use CSV or Excel files.")

    titles1 = df1[file1_column].astype(str)
    titles2 = df2[file2_column].astype(str)

    match_count = titles1.isin(titles2).sum()

    not_in_titles2 = titles1[~titles1.isin(titles2)]

    if show_missing:
        [print(item) for item in not_in_titles2]

    total_titles1 = len(titles1)
    total_titles2 = len(titles2)

    print(f"{match_count} out of {total_titles1} matches.")
    return match_count, total_titles2


def count_title_matches_from_list(file: pd.DataFrame, selected_list: list, file1_column: str, show_missing: bool = False, show_score: bool = False):
    """
    Count the number of matching titles between a file and a selected list of titles.

    Parameters:
    -----------
    file (pd.DataFrame): Data containing records or path to it (csv or xls(x)).

    selected_list (list): List of titles for comparison.

    file1_column (str): Column containing titles in the file.

    show_missing (bool, optional): Whether to display titles that are in the file but not in the selected list (default is False).

    Returns:
    -----------
    None
    """
    titles1 = file[file1_column].astype(str)
    titles2 = selected_list

    match_count = titles1.isin(titles2).sum()

    not_in_titles2 = titles1[~titles1.isin(titles2)]
    if show_missing:
        [print(item) for item in not_in_titles2]

    total_titles1 = len(titles1)
    total_titles2 = len(titles2)

    if show_score:
        print(f"{match_count} out of {total_titles1} matches.")

    return match_count


def emulate_query(query: str, df: pd.DataFrame, title_column: str, abstract_column: str) -> list:
    """
    Emulate a query on a dataframe based on title and abstract columns.

    Parameters:
    -----------
    query (str): Query string to emulate.

    df (pd.DataFrame): Data containing records.

    title_column (str): Column containing titles in the file.

    abstract_column (str): Column containing abstracts in the file.

    Returns:
    -----------
    filtered_titles (list): List of titles matching the query.
    """
    query_list = query.replace("(", "").replace(")", "").replace(
        " OR ", "|").replace("*", "[a-zA-Z]*").split(" AND ")
    query_list = ["(?:"+item+")" for item in query_list]
    query_list = ' AND '.join(query_list).replace(" AND ", ")(?=.*")
    expr = '(?i)(?=.*{})'
    regex_pattern = r'^{}'.format(''.join(expr.format(query_list)))
    title_matches = df[title_column].str.contains(
        regex_pattern, regex=True, case=False)
    abstract_matches = df[abstract_column].str.contains(
        regex_pattern, regex=True, case=False)
    filtered_df = df[title_matches | abstract_matches]

    filtered_titles = filtered_df[title_column].tolist()

    return filtered_titles


def test_query(query: str, target_file: pd.DataFrame, source_file: pd.DataFrame, target_title_column: str, source_title_column: str, source_abstract_column: str) -> tuple[int, int, list]:
    """
    Test overlap for a given query in a source file against a target file.

    Parameters:
    -----------
    query (str): The query string to be verified.

    target_file (pd.DataFrame): Records that should be matched.

    source_file (pd.DataFrame): Records to be searched.

    test_file (str): Path to the test file acquired from a search engine using the same query.

    target_title_column (str): Column containing titles in the target file.

    source_title_column (str): Column containing titles in the source file.

    source_abstract_column (str): Column containing abstracts in the source file.

    Returns:
    -----------
    score (int): number of target matches.

    filtered_titles (int): Number of source titles matching the queries

    filtered_titles (list): List of source titles matching the query in the test file.

    Example:
    -----------
    verify_query(query = "forest* AND (tropic* OR neotropic*) AND (climber* OR liana* OR vine*) AND (trend* OR change*) AND (ground OR climate OR hyperspectral OR accretion OR precipitation OR reproduction OR environmental)", target_file='D:/Users/Sam/Downloads/lianas_oct24.csv', test_file='D:/Users/Sam/Downloads/savedrecs(6).xls', target_title_column='title', test_title_column='Article Title', test_abstract_column='Abstract')
    """
    filtered_titles = emulate_query(
        query, source_file, source_title_column, source_abstract_column)
    score = count_title_matches_from_list(
        target_file, filtered_titles, target_title_column)
    return score, len(filtered_titles), filtered_titles


def verify_query(query: str, target_file: str, source_file: str, test_file: str, target_title_column: str, test_title_column: str, source_title_column: str, source_abstract_column: str) -> list:
    """
    Verify behavior for a given query against a target and test files.

    Parameters:
    -----------
    query (str): The query string to be verified.

    target_file (str): Path to the target file.

    source_file (str): Path to the source file.

    test_file (str): Path to the test file acquired from a search engine using the same query.

    target_title_column (str): Column containing titles in the target file.

    test_title_column (str): Column containing titles in the test file.

    source_title_column (str): Column containing titles in the source file.

    source_abstract_column (str): Column containing abstracts in the source file.

    Returns:
    -----------
    filtered_titles (list): List of titles matching the query in the test file.

    Example:
    -----------
    verify_query(query = "forest* AND (tropic* OR neotropic*) AND (climber* OR liana* OR vine*) AND (trend* OR change*) AND (ground OR climate OR hyperspectral OR accretion OR precipitation OR reproduction OR environmental)", target_file='D:/Users/Sam/Downloads/lianas_oct24.csv', source_file='D:/Users/Sam/Downloads/lianas_original.xls', test_file='D:/Users/Sam/Downloads/savedrecs(6).xls', target_title_column='title', test_title_column='Article Title', source_title_column='Article Title', source_abstract_column='Abstract')
    """
    source = import_df(source_file)
    target = import_df(target_file)
    filtered_titles = emulate_query(
        query, source, source_title_column, source_abstract_column)
    score = count_title_matches_from_list(
        target, filtered_titles, target_title_column)
    matches = count_title_matches(target, test_file,
                                  target_title_column, test_title_column)
    return score, matches, filtered_titles


def generate_query_combinations(items: list) -> list[str]:
    """
    Description:
    ------------
    Generate all possible combinations from a list using the Cartesian product.
    This function takes a list of items as input and returns a list of combinations.

    Parameters:
    -----------
    items (list): List of items for which combinations need to be generated.

    Return:
    -------
    combinations (list): List of combinations, where each combination is represented as a string.
    """
    combinations = []

    for r in range(1, len(items) + 1):
        for combination in product(items, repeat=r):
            combinations.append(' OR '.join(combination))

    combinations = [
        f'({combination})' for combination in combinations if ' OR ' in combination]

    combinations.extend(items)
    combinations = list(set(combinations))

    and_combinations = [
        f'{combination} AND {item}' for item in items for combination in combinations if item not in combination]
    and_combinations = list(set(and_combinations))

    combinations.extend(and_combinations)
    combinations = list(set(combinations))

    return combinations


def generate_queries(query: str, items: list, generate_combinations: bool = False):
    """
    Description:
    ------------
    Generate a list of queries by combining the input query with each item in the provided list. The input query is combined with each item using "AND" to form a new query.

    Parameters:
    -----------
    query (str): The base query to which items are appended.

    items (list): List of items to be combined with the query.

    generate_combinations (bool): Indicated whether to generate all potential combinations of two tags within the items list. 

    Return:
    -------
    query_list (list of str): List of queries generated by combining the input query with each item.
    """
    query_list = []
    if generate_combinations:
        combinations = generate_query_combinations(items)
    else:
        combinations = items
    for item in combinations:
        if query:
            updated_query = query + ' AND ' + item
        else:
            updated_query = item
        query_list.append(updated_query)
    return query_list


def select_relevant_tags(source_file: str, tag_column: str, record_amount: int) -> list:
    """
    Description:
    ------------
    Select relevant tags from a specified column in a CSV file.

    Parameters:
    -----------
    source_file (str): Path to the CSV file containing the tags.

    tag_column (str): Column name in the CSV file containing tags.

    record_amount (int): Number of records to extract.

    Return:
    -------
    tags_list (list of str): List of relevant tags.
    """
    df = pd.read_csv(source_file)
    tags = df.head(record_amount)[tag_column]
    tags_list = ', '.join(list(tags))
    tags_list = tags_list.split(', ')
    return tags_list


def find_optimal_query_in_batches(query_list: list, target: pd.DataFrame, source: pd.DataFrame, target_title_column: str, source_title_column: str, source_abstract_column: str, max_matches: int, original_query: str) -> tuple[str, float, int]:
    """
    Description:
    ------------
    Find the optimal query by iteratively testing different queries and selecting the shortest one with the highest score and the lowest number of mismatches below a specified threshold.

    Parameters:
    -----------
    query_list (list of str): List of queries to be tested.

    target (pd.Dataframe): Data that should be matched by the queries.

    source (pd.Dataframe): Data to be searched that should not be matched by the queries.

    target_title_column (str): Column name in the target file containing titles.

    source_title_column (str): Column name in the source file containing titles.

    source_abstract_column (str): Column name in the source file containing abstracts.

    max_matches (int): Maximum number of matches allowed.

    Return:
    -------
    best_query (str): The optimal query with the highest score and the lowest number of matches.

    highest_score (float): The highest score among the tested queries.

    lowest_matches (int): The lowest number of matches among the tested queries.
    """
    highest_score = float('-inf')
    lowest_matches = float('inf')
    best_query = ""

    for query in query_list:
        score, matches, _ = test_query(
            query, target, source, target_title_column, source_title_column, source_abstract_column)
        old_length = len(re.split(' AND | OR ', best_query))
        new_length = len(re.split(' AND | OR ', query))
        
        if highest_score == float('-inf'):
            highest_score, lowest_matches, best_query = score, matches, query

        elif (score/(matches + 0.01)) > (highest_score/(lowest_matches+0.01)) and score > round(highest_score-(len(target)/5)): 
            highest_score, lowest_matches, best_query = score, matches, query
        
        elif (score, matches) == (highest_score, lowest_matches) and (old_length > new_length or score > highest_score):
            highest_score, lowest_matches, best_query = score, matches, query

    query = best_query.replace(f"{original_query} AND ", "")
    print(f"{query}: {highest_score}/{lowest_matches}")

    return best_query, highest_score, lowest_matches


def auto_optimize_query(query: str, extra_tags: str, target_title_column: str, source_title_column: str, source_abstract_column: str, max_matches: int, source_file: str = None, target_file: str = None, ingroup: pd.DataFrame = None, outgroup: pd.DataFrame = None, batch_size: int = 20):
    """
    Automatically optimize the query by iteratively removing words and testing the resulting queries until a satisfactory query is found. It utilizes the find_optimal_query function for testing.

    Parameters:
    -----------
    query (str): The initial query to be optimized.

    extra_tags (str): Additional tags to be considered in the optimization process.

    target_title_column (str): Column name in the target file containing titles.

    source_title_column (str): Column name in the source file containing titles.

    source_abstract_column (str): Column name in the source file containing abstracts.

    max_matches (int): Maximum number of matches allowed.

    source_file (str): Optional, path to the source file for testing queries.

    target_file (str): Optional, path to the target file for testing queries.

    ingroup (pd.DataFrmae): Optional, Data containing the items to be matched.

    outgroup (pd.DataFrmae): Optional, Data containing the items to not be matched.

    batch_size (int): Optional, defines the number of queries that should be evaluated simoutltaniously.

    Return:
    -------
    best_query (str): The optimized query.

    highest_score (float): The score of the optimized query.

    lowest_matches (int): The number of matches of the optimized query.
    """
    if ingroup is not None and outgroup is not None:
        target = ingroup
        source = pd.concat([outgroup, ingroup])
    elif target_file is not None and source_file is not None: 
        source = import_df(source_file)
        target = import_df(target_file)
    else:
        raise ValueError(
        "Please supply either input file and trainingset or ingroup and outgroup")

    relevant_tags = extra_tags
    non_relevant_tags = [tag for part in query.replace(
        "(", "").replace(")", "").split(" AND ") for tag in part.split(" OR ")]
    non_relevant_tags += ['and', 'or']
    relevant_tags = [item.lstrip(' ') for item in relevant_tags if not any(
        word in item for word in non_relevant_tags)]
    relevant_tags_merged = [f"{query} AND {item}" for item in relevant_tags]

    best_query, highest_score, lowest_matches = find_optimal_query_in_batches(
                query_list=relevant_tags_merged,
                target=target,
                source=source,
                target_title_column=target_title_column,
                source_title_column=source_title_column,
                source_abstract_column=source_abstract_column,
                max_matches=max_matches,
                original_query=query,
                )
    while True:
        selected_word = best_query.replace(f"{query} AND ", "").replace(f"{query} OR ", "")
        relevant_tags = [tag for tag in relevant_tags if tag not in selected_word]
        relevant_tags_and = [f"{best_query} AND {item}" for item in relevant_tags]
        relevant_tags_or = [f"{query} AND ({selected_word} OR {item})" for item in relevant_tags]
        relevant_tags_merged = relevant_tags_and + relevant_tags_or

        better_query, score, matches = find_optimal_query_in_batches(
            query_list=relevant_tags_merged,
            target=target,
            source=source,
            target_title_column=target_title_column,
            source_title_column=source_title_column,
            source_abstract_column=source_abstract_column,
            max_matches=max_matches,
            original_query=query,
        )
        old_length = len(re.split(' AND | OR ', best_query))
        new_length = len(re.split(' AND | OR ', better_query))

        if (score / (matches + 0.01)) > (highest_score / (lowest_matches + 0.01)) and score > round(highest_score - (len(target)/5)):
            highest_score, lowest_matches, best_query = score, matches, better_query
            best_query = best_query.replace("((", "(").replace("OR (", "OR ")
        elif (score, matches) == (highest_score, lowest_matches) and (old_length > new_length or score > highest_score):
            highest_score, lowest_matches, best_query = score, matches, better_query
        else:
            break
    return best_query, highest_score, lowest_matches


def propose_tags(original_query: str, keyword_column: str, n_tags: int, threshold: float, ingroup: pd.DataFrame = None, outgroup: pd.DataFrame = None, input_file: str = None ):
    """
    Description:
    ------------
    Propose a list of tags based on an original query, input file, and specified parameters. It combines pseudo-relevance feedback and similarity feedback to expand the query and suggests tags.

    Parameters:
    -----------
    original_query (str): The original query to be expanded.

    keyword_column (str): Column name in the input file containing keywords.

    n_tags (int): Number of tags to propose.

    threshold (float): Threshold for similarity feedback.

    ingroup (pd.DataFrame): Optional, Data containing the items that should be represented.

    outgroup (pd.DataFrame): Optional, Data containing the items that should not be represented. 

    input_file (str): Path to the input file for feedback.
    
    Return:
    -------
    merged_terms (list of str): List of proposed tags after merging similar terms.
    """
    expanded_query1, top_articles = pseudo_relevance_feedback(
        original_query=original_query, n_tags=n_tags, ingroup=ingroup, outgroup=outgroup)
    if ingroup is not None:
        expanded_query2 = similarity_feedback(
            ingroup=ingroup, original_query=original_query, keyword_column=keyword_column, threshold=threshold)
    elif input_file is not None:
        expanded_query2 = similarity_feedback(
            input_path=input_file, original_query=original_query, keyword_column=keyword_column, threshold=threshold)
    else:
        raise ValueError( 
            "Please supply either input file or ingroup")

    expanded_query = expanded_query1 + expanded_query2
    expanded_query = ', '.join(expanded_query)
    expanded_terms = merge_words_by_stem_and_wildcards(expanded_query)
    expanded_terms = group_fuzzy_words(expanded_terms, 80)
    expanded_terms = group_related_words(expanded_terms)
    merged_terms = group_synonyms(expanded_terms)
    return merged_terms


def calculate_best_query_by_subset(original_query: str, input_file:str, tag_column: str, n_tags: int, threshold: float, target_title_column: str, source_title_column: str, source_abstract_column: str, max_matches: int, ingroup:pd.DataFrame = None, outgroup:pd.DataFrame = None):
    """
    Description:
    ------------
    Determine the best query for the given training percentage. The ingroup is defined as the top (training percent)% records, the outgroup as the remaining records.

    Parameters:
    -----------
    trainingset (tuple of int): Tuple representing the percentage of documents to consider as the training set and the remaining as the testing set.

    original_query (str): The original query to be optimized.

    tag_column (str): Column name in the input file containing keywords.

    n_tags (int): Number of tags to propose.

    threshold (float): Threshold for similarity feedback.

    ingroup (pd.DataFrame): Data containing the items to be matched.

    outgroup (pd.DataFrame): Data containing the items to not be matched

    Return:
    -------
    best_query (str): The optimized query for the current training set percentage.

    highest_score (float): The score of the optimized query.

    lowest_matches (int): The number of matches of the optimized query.
    """
    merged_terms = propose_tags(
            original_query=original_query, input_file=input_file, keyword_column=tag_column, n_tags=n_tags, threshold=threshold, ingroup=ingroup, outgroup=outgroup)
    best_query, highest_score, lowest_matches = auto_optimize_query(query=original_query, extra_tags=merged_terms, target_title_column=target_title_column, source_title_column=source_title_column, 
                                                                        source_abstract_column=source_abstract_column, max_matches=max_matches, ingroup=ingroup, outgroup=outgroup)
    return best_query, highest_score, lowest_matches


def iteratively_propose_query(original_query: str, input_file: str, trainingset: int, n_tags: int, threshold: float, tag_column: str, target_file: str, target_title_column: str, source_title_column: str, source_abstract_column: str, max_matches: int, method:str = "none", subsample_size:int = 10):
    """
    Description:
    ------------
    Propose an optimized query based on an original query, input file, and specified parameters. It iteratively refines the query by adjusting the training set percentage and utilizing the auto_optimize_query function.

    Parameters:
    -----------
    original_query (str): The original query to be optimized.

    input_file (str): Path to the input file for feedback.

    trainingset (int): Used as initial percentage of documents to consider as the training set, after which it iterates decreasing by 5% each time.

    n_tags (int): Number of tags to propose.

    threshold (float): Threshold for similarity feedback.

    tag_column (str): Column name in the input file containing tags.

    target_file (str): Path to the target file for testing queries.

    target_title_column (str): Column name in the target file containing titles.

    source_title_column (str): Column name in the input file containing titles.

    source_abstract_column (str): Column name in the input file containing abstracts.

    max_matches (int): Maximum number of matches allowed.

    mode (str): Type of subsampling. Can be "random", for ten iterations of random subsamples of the trainingset with size subsample_size; "downsampling" for subsamples starting with the trainingset size and decrasing with subsample_size per iteration or "none" for one iteration at the trainingset size. 

    subsample_size (int): Optional. Indicates size of random sample or stepsize for downsampling. Defaults to 10.

    Return:
    -------
    query_dict (dict): Dictionary containing the best query, highest score, and lowest matches for each iteration of the training set percentage.
    """
    query_dict = {}
    ingroup_size = trainingset/100
    df = pd.read_csv(input_file)
    df_len = len(df)
    in_len = round(df_len*ingroup_size)
    out_len = df_len-in_len
    ingroup = df.head(in_len)
    outgroup = df.tail(out_len)

    if method == "none":
        out_len = df_len-in_len
        ingroup = df.head(in_len)
        outgroup = df.tail(out_len)
        best_query, highest_score, lowest_matches = calculate_best_query_by_subset(
            original_query=original_query, input_file=input_file, ingroup=ingroup, outgroup=outgroup, tag_column=tag_column, n_tags=n_tags, threshold=threshold, target_title_column=target_title_column, source_title_column=source_title_column, source_abstract_column = source_abstract_column, max_matches=max_matches)
        query_dict[in_len] = {"query": best_query,
                            "score": highest_score, "matches": lowest_matches}
        print(f"The current best query for {in_len} {method} items is: {best_query} with a score of {highest_score}/{lowest_matches}")
    elif method == "random":
        for i in range(10):
            sample = ingroup.sample(subsample_size)
            merged_df = pd.merge(df, sample, how='outer', indicator=True)
            outgroup = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
            best_query, highest_score, lowest_matches = calculate_best_query_by_subset(
                original_query=original_query, input_file=input_file, ingroup=sample, outgroup=outgroup, tag_column=tag_column, n_tags=n_tags, threshold=threshold, target_title_column=target_title_column, source_title_column=source_title_column, source_abstract_column = source_abstract_column, max_matches=subsample_size)
            query_dict[f"{in_len}_{i}"] = {"query": best_query,
                                "score": highest_score, "matches": lowest_matches}
            print(f"The current best query for {in_len} {method} items is: {best_query} with a score of {highest_score}/{lowest_matches}")
    elif method == "downsampling":
        while True:
            if in_len < subsample_size:
                break
            out_len = df_len-in_len
            ingroup = df.head(in_len)
            outgroup = df.tail(out_len)
            best_query, highest_score, lowest_matches = calculate_best_query_by_subset(
                original_query=original_query, input_file=input_file, ingroup=ingroup, outgroup=outgroup, tag_column=tag_column, n_tags=n_tags, threshold=threshold, target_title_column=target_title_column, source_title_column=source_title_column, source_abstract_column = source_abstract_column, max_matches=max_matches)
            in_len = in_len - subsample_size
            query_dict[in_len] = {"query": best_query,
                                "score": highest_score, "matches": lowest_matches}
            print(f"The current best query for {in_len} {method} items is: {best_query} with a score of {highest_score}/{lowest_matches}")

        max_entry = max(query_dict.values(), key=lambda x: (x["score"], -x["matches"]))
        key = list(query_dict.keys())[list(query_dict.values()).index(max_entry)]
        print(
            f"The best query was found for a trainingset of {key}% with: {max_entry['query']} with a score of {max_entry['score']}/{max_entry['matches']}")
    else:
        raise ValueError(
            f"Please supply a valid method: random, downsampling or none")
    return query_dict


def calculate_best_query_by_cluster(cluster_column: str, original_query: str, input_file: str, tag_column: str, n_tags: int, threshold: float, target_title_column: str, source_title_column: str, source_abstract_column: str, minimum_tag_length: int = 3):
    """
    Description:
    ------------
    Determine the best query for each cluster in the given column. The ingroup is defined as the top (training percent)% records, the outgroup as the remaining records.

    Parameters:
    -----------
    trainingset (tuple of int): Tuple representing the percentage of documents to consider as the training set and the remaining as the testing set.

    original_query (str): The original query to be optimized.

    input_file (str): Path to the input file for feedback.

    tag_column (str): Column name in the input file containing keywords.

    n_tags (int): Number of tags to propose.

    threshold (float): Threshold for similarity feedback.

    minimum_tag_length (int): Optional, minimum length for a tag to be considered. Defaults to 3.

    Return:
    -------
    best_query (str): The optimized query for the current training set percentage.

    highest_score (float): The score of the optimized query.

    lowest_matches (int): The number of matches of the optimized query.
    """
    def find_tags(df, column):
        tags_list = df[column]
        tags_list = ', '.join(list(tags_list))
        tags_list = tags_list.split(', ')
        return tags_list
    dataframe = pd.read_csv(input_file)
    cluster_dict = {}
    for cluster_value in dataframe[cluster_column].unique():
        data_ingroup = dataframe[dataframe[cluster_column] == cluster_value]
        max_matches = len(data_ingroup)
        data_outgroup = dataframe[dataframe[cluster_column] != cluster_value]
        record_amount = dataframe[dataframe[cluster_column]
                                  == cluster_value].shape[0]

        print(
            f"Cluster {cluster_value}: {max_matches}/{len(dataframe)} records")

        merged_terms = propose_tags(
            original_query, tag_column, n_tags, threshold, ingroup=data_ingroup, outgroup=data_outgroup)
        relevant_tags = find_tags(data_ingroup, tag_column)
        merged_terms += relevant_tags
        merged_terms = [term for term in merged_terms if len(
            term) >= minimum_tag_length]
        best_query, highest_score, lowest_matches = auto_optimize_query(query=original_query, extra_tags=merged_terms, target_title_column=target_title_column, source_title_column=source_title_column, 
                                                                        source_abstract_column=source_abstract_column, max_matches=max_matches, ingroup=data_ingroup, outgroup=data_outgroup)
        cluster_dict[cluster_value] = {
            "query": best_query, "score": highest_score, "matches": lowest_matches}
        print(
            f"The current best query for {cluster_value} is: {best_query} with a score of {highest_score}/{lowest_matches}")
    return cluster_dict


def analyze_clusters(query: str, cluster_range: tuple, cluster_column: str, training_filepath: str, title_column_test_file: str, title_column_training_file: str, abstract_column: str, keyword_column: str, n_tags: int, threshold: float, test_filepath: str = None) -> dict:
    """
    match the amount of titles in the cluster and optimize the query for each cluster within range

    Parameters:
    -----------
    query (str): The original query to be optimized.

    cluster_range (tuple): Range of clusters to be inspected. Generally starts with 1.

    training_filepath (str): Path to the input file for feedback.

    title_column_test_file (str): Column name in the target file containing titles.

    title_column_training_file (str): Column name in the input file containing titles.

    abstract_column (str): Column name in the input file containing abstracts.

    keyword_column (str): Column name in the input file containing tags.

    n_tags (int): Number of tags to propose.

    threshold (float): Threshold for similarity feedback.

    cluster_column (str): if supplied iterates over unique values as ingroup

    test_filepath (str): Optional, path to the target file for files that should be present in output.

    return:
    -------
    query_dict (dict): Dictionary containing the best query, highest score, and lowest matches for each iteration of the training set percentage.
    """
    minimum, maximum = cluster_range
    if test_filepath:
        test_file = import_df(test_filepath)
        for i in range(minimum, maximum):
            training_data = pd.read_csv(training_filepath)
            ingroup = training_data[training_data[cluster_column] == i]
            count_title_matches(
                test_file, ingroup, title_column_test_file, title_column_training_file)
    
    query_dict = calculate_best_query_by_cluster(cluster_column=cluster_column, original_query=query, input_file=training_filepath, tag_column=keyword_column, n_tags=n_tags,
                                                 threshold=threshold, target_title_column=title_column_test_file, source_title_column=title_column_training_file, source_abstract_column=abstract_column)
    return query_dict



if __name__ == "__main__":
    # match the amount of titles in the cluster and optimize the query for each cluster within range
    
    #query_dict = analyze_clusters(query="forest* AND tropic* AND (climber* OR liana* OR vine*) AND (trend* OR change*)", cluster_column="Cluster", cluster_range=(1, 9), training_filepath="C:/NLPvenv/NLP/output/csv/savedrecs_lianas_sorted_all_clusters.csv",
    #                              test_filepath="D:/Users/Sam/Downloads/lianas_oct24.csv", title_column_training_file="Article Title", title_column_test_file="Article Title", abstract_column="Abstract", keyword_column="Keywords", n_tags=30, threshold=0.2)
    df = pd.read_csv("C:/NLPvenv/NLP/output/csv/savedrecs_lianas_sorted_all_clusters.csv")
    df1 = pd.read_csv("D:/Users/Sam/Downloads/lianas_oct24.csv")
    titles = emulate_query(query="forest* AND tropic* AND (climber* OR liana* OR vine*) AND (trend* OR change*) AND ground AND plots", df=df, title_column="Article Title", abstract_column="Abstract")
    selection = df1[df1["Article Title"].isin(titles)]
    print(selection)
# find dominant tags
    # components = retrieve_pca_components(input_file="C:/NLPvenv/NLP/output/csv/savedrecs_lianas.csv", output="C:/NLPvenv/NLP/output/csv/savedrecs_lianas_sorted_deduplicated.csv", variables="Keywords", id="Article Title", tag_length=4, n_components_for_variance=80, number_of_records=35, show_plots="loading and scree and saturation")

# optimize query
    #query_dict = iteratively_propose_query(original_query="forest* AND tropic* AND (climber* OR liana* OR vine*) AND (trend* OR change*)", input_file="C:/NLPvenv/NLP/output/csv/savedrecs_lianas_sorted.csv", trainingset=14, n_tags=30, threshold=0.2,
    #        tag_column="Keywords", target_file='D:/Users/Sam/Downloads/lianas_oct24.csv', target_title_column='Article Title', source_title_column='Article Title', source_abstract_column='Abstract', max_matches=37, method="none", subsample_size=2)
    #for i in query_dict:
    #    print(f"{i};{query_dict[i]['query']};{query_dict[i]['score']};{query_dict[i]['matches']}")

    # subsample
    # titles = subsample_from_csv(CSV_path="D:/Users/Sam/Downloads/savedrecs(7).csv",
    #                            y="Keywords", x="Article Title", n=40, distance_type="similarity")
    # count_title_matches_from_list(file1_path='D:/Users/Sam/Downloads/lianas_oct24.csv', file1_column='title', selected_list=titles,show_score=True)
