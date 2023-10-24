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


def pca_tags(CSV: str, variables: str, id: str, tag_length: int):
    matrix, id_list = generate_binary_item_matrix(
        CSV, variables, id, tag_length)
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
    return merged_words


def group_synonyms(words, fuzzy_threshold = ""):
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
            elif fuzzy_threshold: # Check for fuzzy matches if treshold is supplied
                similarity = fuzz.token_sort_ratio(word1, word2)
                if similarity >= fuzzy_threshold:
                    key = f"({word1} OR {word2})"
                    grouped_list.append(key)

    words_without_synonyms = list(set(words_without_synonyms))
    grouped_list = grouped_list + words_without_synonyms
    return grouped_list


def group_related_words(word_list):
    synsets = {}
    
    for word in word_list:
        synsets[word] = set([lemma.name() for syn in wordnet.synsets(word) for lemma in syn.lemmas()])
    
    grouped_words = []
    processed = set()
    
    for word in word_list:
        if word not in processed:
            related = synsets[word]
            grouped_words.extend(related)
            processed.update(related)
    
    return list(set(grouped_words))


def expand_query_with_tag_similarity(query, tags, threshold=0.2, top_k=1):
    """
    Expand a query based on tag similarity.

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
    if isinstance(tags, set):
        tags = list(tags)

    vectorizer = TfidfVectorizer()
    tag_matrix = vectorizer.fit_transform(tags)
    query_vec = vectorizer.transform([query])
    query_vec = query_vec.toarray()
    tag_similarity = cosine_similarity(query_vec, tag_matrix)

    if tag_similarity.shape[0] == 0:
        return query

    tag_similarity = tag_similarity[0]

    sorted_indices = np.argsort(tag_similarity)[::-1]
    sorted_terms = [tags[i] for i in sorted_indices]

    expanded_terms = [term for term in sorted_terms if tag_similarity[tags.index(
        term)] > threshold][:top_k]
    return expanded_terms


def similarity_feedback(input: str, original_query: str = "Culex pipiens AND population dynamics", keyword_column: str = 'keywords', threshold:float=0.2, top_k:int=3) -> list[str]:
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


def pseudo_relevance_feedback(input: str, original_query: str, trainingset: tuple[int, int] = (50, 20), n_tags: int = 10, n_articles: int = 20) -> list[str]:
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
    ingroup_percent = int((ingroup/100) * length)
    outgroup_percent = int((outgroup/100) * length)
    good_matches = df.head(ingroup_percent).copy()
    bad_matches = df.tail(outgroup_percent).copy()

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

    print("Relevant Tags with TF-IDF Weights:")
    for tag, weight in relevant_tags[:n_tags]:
        print(f"{tag}: {weight}")

    cosine_similarities = cosine_similarity(updated_query_array, tfidf_matrix)

    df['cosine_similarity'] = cosine_similarities[0]
    df = df.sort_values(by='cosine_similarity', ascending=False)

    top_articles = df.head(n_articles)
    return top_relevant_tags, top_articles


if __name__ == "__main__":
    #pca_tags("C:/NLPvenv/NLP/output/csv/total.csv", "keywords", "title", 4)
    original_query = "forest* AND tropic* AND (climber* OR liana* OR vine*) AND (trend* OR change*)"
    expanded_query1, top_articles = pseudo_relevance_feedback("C:/NLPvenv/NLP/output/csv/savedrecs_lianas.csv", original_query, trainingset=(20,60))
    expanded_query2 = similarity_feedback(input="C:/NLPvenv/NLP/output/csv/savedrecs_lianas.csv", original_query=original_query, keyword_column='Keywords', threshold=0.252)
    expanded_query = expanded_query1 + expanded_query2
    expanded_query = ', '.join(expanded_query)
    expanded_terms = merge_words_by_stem_and_wildcards(expanded_query)
    #merged_terms = group_synonyms(expanded_terms)
    print(expanded_terms)
    expanded_terms = group_related_words(expanded_terms)
    print(expanded_terms)
    """
    merged_terms = ' OR '.join(merged_terms)

    if merged_terms:
        expanded_query = original_query + ' AND "' + merged_terms + '"'
    else:
        expanded_query = original_query
    print(expanded_query)
    expanded_query_1, top_articles = pseudo_relevance_feedback("C:/NLPvenv/NLP/output/csv/savedrecs_lianas.csv", expanded_query, trainingset=(20,60), n_articles=40)
    """


# pca to see tag clustering?

# correct for dependence among tags
