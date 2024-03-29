import os

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


def pca_biplot(X: np.ndarray, y, targets: list, features: list):
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


def expand_query_with_tag_similarity(query, tags, threshold=0.2, top_k=1):
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
    tag_matrix = vectorizer.fit_transform(tags)
    query_vec = vectorizer.transform([query])
    query_vec = query_vec.toarray()
    tag_similarity = cosine_similarity(query_vec, tag_matrix)

    if tag_similarity.shape[0] == 0:
        return [query]

    tag_similarity = tag_similarity[0]

    sorted_indices = np.argsort(tag_similarity)[::-1]
    sorted_terms = [tags[i] for i in sorted_indices]

    expanded_terms = [term for term in sorted_terms if tag_similarity[tags.index(
        term)] > threshold][:top_k]
    return expanded_terms


def group_synonyms(words, fuzzy_threshold=""):
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


def similarity_feedback(input: str, original_query: str = "Culex pipiens AND population dynamics", keyword_column: str = 'keywords', threshold: float = 0.2, top_k: int = 3) -> list[str]:
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
    df = pd.read_csv(input)
    df = df.dropna(subset=[keyword_column])
    tags = set(df[keyword_column].tolist())
    vectorizer = CountVectorizer()
    vectorizer.fit(tags)
    expanded_query = expand_query_with_tag_similarity(
        original_query, tags, threshold, top_k)
    return expanded_query


def pseudo_relevance_feedback(input: str, original_query: str, trainingset: tuple[int, int] = (50, 20), n_tags: int = 10, n_articles: int = 20, print_weights:bool = False) -> list[str]:
    """
    Perform pseudo-relevance feedback on a collection of documents using the Rocchio algorithm on TF-IDF weights.

    Parameters:
    -----------
    inputCSV (str): Path to the CSV file containing the document items (title and abstract) to scan.

    original_query (str): The original query string for retrieval.

    trainingset (tuple): A tuple specifying the percentage of documents to use as the "good" and "bad" training sets (ingroup and outgroup). The ingroup is taken from the top, the outgroup from the bottom.

    n_tags (int): Number of top relevant tags to keep for query expansion.

    n_articles (int): Number of top relevant articles to display.

    Returns:
    -----------
    updated_query_str (str): The updated query string after pseudo-relevance feedback.
    """
    df = pd.read_csv(input)
    ingroup, outgroup = trainingset
    length = df.shape[0]
    ingroup_records = int((ingroup/100) * length)
    outgroup_records = int((outgroup/100) * length)
    good_matches = df.head(ingroup_records).copy()
    bad_matches = df.tail(outgroup_records).copy()

    good_matches['tags_str'] = good_matches['Keywords']
    bad_matches['tags_str'] = bad_matches['Keywords']

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Keywords'])

    centroid_good = np.mean(tfidf_matrix[:ingroup], axis=0)
    centroid_bad = np.mean(tfidf_matrix[-outgroup:], axis=0)

    alpha = 1.0
    beta = 0.75
    gamma = 0.25

    expanded_query_vector = tfidf_vectorizer.transform([original_query])
    updated_query = alpha * expanded_query_vector + \
        beta * centroid_good - gamma * centroid_bad
    updated_query_array = np.asarray(updated_query)
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

    cosine_similarities = cosine_similarity(updated_query_array, tfidf_matrix)

    df['cosine_similarity'] = cosine_similarities[0]
    df = df.sort_values(by='cosine_similarity', ascending=False)

    top_articles = df.head(n_articles)
    return top_relevant_tags, top_articles


def count_title_matches(file1_path, file2_path, file1_column, file2_column, show_missing: bool = False):
    """
    Count the number of matching titles between two files.

    Parameters:
    -----------
    file1_path (str): Path to the first file.

    file2_path (str): Path to the second file.

    file1_column (str): Column containing titles in the first file.

    file2_column (str): Column containing titles in the second file.

    show_missing (bool, optional): Whether to display titles that are in the first file but not in the second file (default is False).

    Returns:
    -----------
    None
    """
    file1_extension = file1_path.split('.')[-1]
    file2_extension = file2_path.split('.')[-1]

    if file1_extension == 'csv':
        df1 = pd.read_csv(file1_path)
    elif file1_extension in ['xls', 'xlsx']:
        df1 = pd.read_excel(file1_path)
    else:
        raise ValueError(
            f"Unsupported file format for {file1_path}. Use CSV or Excel files.")

    if file2_extension == 'csv':
        df2 = pd.read_csv(file2_path)
    elif file2_extension in ['xls', 'xlsx']:
        df2 = pd.read_excel(file2_path)
    else:
        raise ValueError(
            f"Unsupported file format for {file2_path}. Use CSV or Excel files.")

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


def count_title_matches_from_list(file1_path: str, selected_list: list, file1_column: str, show_missing: bool = False, show_score: bool = False):
    """
    Count the number of matching titles between a file and a selected list of titles.

    Parameters:
    -----------
    file1_path (str): Path to the file.

    selected_list (list): List of titles for comparison.

    file1_column (str): Column containing titles in the file.

    show_missing (bool, optional): Whether to display titles that are in the file but not in the selected list (default is False).

    Returns:
    -----------
    None
    """
    file1_extension = file1_path.split('.')[-1]

    if file1_extension == 'csv':
        df1 = pd.read_csv(file1_path)
    elif file1_extension in ['xls', 'xlsx']:
        df1 = pd.read_excel(file1_path)
    else:
        raise ValueError(
            f"Unsupported file format for {file1_path}. Use CSV or Excel files.")

    titles1 = df1[file1_column].astype(str)
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


def emulate_query(query: str, file_path: str, title_column: str, abstract_column: str) -> list:
    """
    Emulate a query on a dataframe based on title and abstract columns.

    Parameters:
    -----------
    query (str): Query string to emulate.

    file_path (str): Path to the file containing data.

    title_column (str): Column containing titles in the file.

    abstract_column (str): Column containing abstracts in the file.

    Returns:
    -----------
    filtered_titles (list): List of titles matching the query.
    """
    df = pd.read_excel(file_path)

    query_list = query.replace("(", "").replace(")", "").replace(
        " OR ", "|").replace("*", "").split(" AND ")
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


def test_query(query: str, target_file: str, source_file: str, target_title_column: str, source_title_column: str, source_abstract_column: str) -> list:
    """
    Test overlap for a given query in a source file against a target file.

    Parameters:
    -----------
    query (str): The query string to be verified.

    target_file (str): Path to the target file containing the records that should be matched.

    source_file (str): Path to the source file containing the records to be searched.

    test_file (str): Path to the test file acquired from a search engine using the same query.

    target_title_column (str): Column containing titles in the target file.

    source_title_column (str): Column containing titles in the source file.

    source_abstract_column (str): Column containing abstracts in the source file.

    Returns:
    -----------
    filtered_titles (list): List of titles matching the query in the test file.

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
    filtered_titles = emulate_query(
        query, source_file, source_title_column, source_abstract_column)
    count_title_matches_from_list(
        target_file, filtered_titles, target_title_column)
    count_title_matches(target_file, test_file,
                        target_title_column, test_title_column)
    return filtered_titles


def generate_query_combinations(items: list) -> list[str]:
    """Generate all possible combinations from a list"""
    combinations = [' OR '.join(list(set(combination)))
                    for combination in product(items, repeat=len(items))]

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


def generate_queries(query: str, items: list):
    query_list = []
    combinations = generate_query_combinations(items)
    for item in combinations:
        if query != "":
            updated_query = query + ' AND ' + item
        else:
            updated_query = item
        query_list.append(updated_query)
    return query_list


def find_optimal_query(query_list, target_file: str, source_file: str, target_title_column: str, source_title_column: str, source_abstract_column: str, max_matches: int) -> tuple[str, int]:
    highest_score = float('-inf')
    lowest_matches = float('inf')
    best_query = ""

    for query in query_list:
        score, matches, _ = test_query(
            query, target_file, source_file, target_title_column, source_title_column, source_abstract_column)

        if matches < max_matches:
            if score >= (highest_score - 2):
                highest_score = score
                lowest_matches = matches
                best_query = query

    return best_query, highest_score, lowest_matches


def select_relevant_tags(source_file: str, tag_column: str, record_amount: int) -> list:
    df = pd.read_csv(source_file)
    tags = df.head(record_amount)[tag_column]
    tags_list = ', '.join(list(tags))
    tags_list = tags_list.split(', ')
    return tags_list


def auto_optimize_query(query: str, extra_tags: str, source_file: str, tag_file: str, tag_column: str, record_amount: int, target_file: str, target_title_column: str, source_title_column: str, source_abstract_column: str, max_matches: int):
    relevant_tags = []

    if record_amount != "":
        relevant_tags = select_relevant_tags(tag_file, tag_column, record_amount)

    if extra_tags != "":
        relevant_tags = relevant_tags + extra_tags
    non_relevant_tags = ['forest', 'tropic',
                         'climber', 'liana', 'vine', 'trend', 'change']

    for word in non_relevant_tags:
        relevant_tags = [item for item in relevant_tags if word not in item]
    query_list = relevant_tags.copy()

    best_query, highest_score, lowest_matches = find_optimal_query(
        query_list=query_list,
        target_file=target_file,
        source_file=source_file,
        target_title_column=target_title_column,
        source_title_column=source_title_column,
        source_abstract_column=source_abstract_column,
        max_matches=max_matches
    )

    print(
        f"The current query is: {best_query} with a score of {highest_score}/{lowest_matches}")

    while True:
        higher_score = highest_score
        lower_matches = lowest_matches

        selected_word = best_query.replace(relevant_tags[0], "")
        relevant_tags = [
            item for item in relevant_tags if item != selected_word]

        new_queries = []
        for item in relevant_tags:
            word_list = [best_query, item]
            generated_queries = generate_queries("", word_list)
            new_queries.extend(generated_queries)

        query_list = list(set(new_queries))

        better_query, higher_score, lower_matches = find_optimal_query(
            query_list=query_list,
            target_file=target_file,
            source_file=source_file,
            target_title_column=target_title_column,
            source_title_column=source_title_column,
            source_abstract_column=source_abstract_column,
            max_matches=lowest_matches
        )

        better_query = better_query.replace(') OR', ' OR').replace(
            'OR (', 'OR ').replace('))', ')').replace('((', '(').replace('  ', ' ')
        print(
            f"The current query is: {better_query} with a score of {higher_score}/{lower_matches}")

        if higher_score < highest_score or lower_matches >= lowest_matches:
            break

        best_query = query + " AND " + better_query
        highest_score = higher_score
        lowest_matches = lower_matches

    print(
        f"The best query is: {best_query} with a score of {highest_score}/{lowest_matches}")

    return best_query, highest_score, lowest_matches

def propose_tags(original_query:str, input_file:str, keyword_column:str, trainingset:tuple[int,int], n_tags:int, threshold:float):
    expanded_query1, top_articles = pseudo_relevance_feedback(
        input=input_file, trainingset=trainingset, n_tags=n_tags)
    expanded_query2 = similarity_feedback(input=input_file, original_query=original_query, keyword_column=keyword_column, threshold=threshold)
    expanded_query = expanded_query1 + expanded_query2
    expanded_query = ', '.join(expanded_query)
    expanded_terms = merge_words_by_stem_and_wildcards(expanded_query)
    expanded_terms = group_fuzzy_words(expanded_terms, 80)
    expanded_terms = group_related_words(expanded_terms)
    merged_terms = group_synonyms(expanded_terms)

if __name__ == "__main__":
    pca_tags("C:/NLPvenv/NLP/output/csv/savedrecs_lianas_sorted.csv", "Keywords", "Article Title", 4, 35)
    
    #original_query = "forest* AND tropic* AND (climber* OR liana* OR vine*) AND (trend* OR change*)"
    #merged_terms = propose_tags(original_query=original_query, input_file="C:/NLPvenv/NLP/output/csv/savedrecs_lianas_sorted.csv", keyword_column='Keywords', original_query=original_query, trainingset=(30, 40), n_tags=20, threshold=0.2)
    #best_query, highest_score, lowest_matches = auto_optimize_query(query="forest* AND tropic* AND (climber* OR liana* OR vine*) AND (trend* OR change*)", extra_tags=merged_terms, tag_file="C:/NLPvenv/NLP/output/csv/savedrecs_lianas_sorted.csv", source_file='D:/Users/Sam/Downloads/lianas_original.xls', tag_column="Keywords", record_amount="", target_file='D:/Users/Sam/Downloads/lianas_oct24.csv', target_title_column='title', source_title_column='Article Title', source_abstract_column='Abstract', max_matches=254)
    
    #count_title_matches('D:/Users/Sam/Downloads/lianas_oct24.csv', 'D:/Users/Sam/Downloads/savedrecs(6).xls', 'title', 'Article Title')
# visualize tag clustering
# correct for dependence among tags
