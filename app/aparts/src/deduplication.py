from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from aparts.src.subsampling import (assign_group, generate_binary_item_matrix,
                                    transform_dataframe)


def group_tags_by_dissimilarity(dissimilarity_matrix: np.ndarray, tag_names: list, threshold: float = 0.5, print_output: bool = False) -> list:
    clustering = AgglomerativeClustering(
        n_clusters=None, linkage='average', distance_threshold=threshold, metric='precomputed')
    clusters = clustering.fit_predict(dissimilarity_matrix)

    grouped_columns = {}
    for col_idx, cluster_id in enumerate(clusters):
        if cluster_id not in grouped_columns:
            grouped_columns[cluster_id] = []
        grouped_columns[cluster_id].append(col_idx)

    # Filter clusters to include only those with 2 or more items
    filtered_groups = [
        group for group in grouped_columns.values() if len(group) >= 2]

    # Substitute indices with tag names
    nnamed_groups = [[tag_names[col_idx] for col_idx in group]
                     for group in filtered_groups]

    return nnamed_groups


def generate_tag_dissimilarity(dataframe: pd.DataFrame) -> np.ndarray:
    binary_dataframe = dataframe.astype(int)
    matrix = binary_dataframe.values
    dissimilarity_matrix = pairwise_distances(matrix.T, metric='braycurtis')
    return dissimilarity_matrix


def deduplicate_tag_conjugations(word_list: list, method: str = "", deleted_pairs: list = None) -> list:
    """
    Identify and delete duplicate tags based on their stem from a (nested) list.

    This function takes a nested list of words and performs deduplication by identifying duplicate tags based on their stem.
    It also provides an option to output the identified duplicates so that their data can be merged in their corresponding matrix.

    Parameters:
    -----------
    word_list (list): The nested list of words to be deduplicated.

    method (str): The deduplication method to use. Can be either "deduplicated" or "pairs".

    deleted_pair (list): A list to retain identified duplicate pairs across iterations. Used when method is set to "pairs" and the input is a nested list.

    Returns:
    --------
    list: A deduplicated version of the input nested list. If method is "deduplicated", 
          this list contains the deduplicated words and sublists. If method is "pairs", 
          this list contains pairs of duplicate words and their corresponding stems.
    """
    if deleted_pairs is None:
        deleted_pairs = []

    sorted_words = sorted(word_list, key=len, reverse=True)
    processed_prefixes = set()
    deduplicated = []

    for word in sorted_words:
        pair = ()
        if isinstance(word, list):
            deduplicated_sublist = deduplicate_tag_conjugations(
                word, method, deleted_pairs)
            deduplicated.append(deduplicated_sublist)
        elif isinstance(word, str):
            is_duplicate = any(word.startswith(
                existing_prefix[:len(word)]) for existing_prefix in processed_prefixes)
            if is_duplicate:
                pair = (word, next(existing_prefix for existing_prefix in processed_prefixes if word.startswith(
                    existing_prefix[:len(word)])))
                deleted_pairs.append(pair)
            else:
                deduplicated.append(word)
                processed_prefixes.add(word)

    return deleted_pairs if method == "pairs" else deduplicated


def deduplicate_dataframe(dataframe: pd.DataFrame, pairs_list: list, mode: str = "strict") -> pd.DataFrame:
    """
    Deduplicate a DataFrame based on the provided pairs and mode.

    Parameters:
    -----------
    dataframe (pd.DataFrame): The DataFrame to be deduplicated.

    pairs_list (list): A list of pairs, where each pair is a tuple containing source and sink column names.

    mode (str): The deduplication mode. Can be 'strict' or 'lenient'. In strict mode the contents of the source column are copied to the sink column. 
    The source column is subsequently deleted. In lenient mode, the source column is deleted if its contents are identical to the sink column.

    Returns:
    --------
    pd.DataFrame: The deduplicated DataFrame.
    """
    deduplicated_dataframe = dataframe.copy()

    for pair in pairs_list:
        source, sink = pair

        if source not in deduplicated_dataframe.columns or sink not in deduplicated_dataframe.columns:
            continue

        sourcedata = deduplicated_dataframe[source]
        sinkdata = deduplicated_dataframe[sink]

        if mode == "strict":
            for i in range(len(sourcedata)):
                if sourcedata[i] > 0:
                    sinkdata[i] = sourcedata[i]
            deduplicated_dataframe[sink] = sinkdata
            deduplicated_dataframe.drop(columns=[source], inplace=True)

        if mode == "lenient":
            if sourcedata.equals(sinkdata):
                deduplicated_dataframe.drop(columns=[source], inplace=True)

    return deduplicated_dataframe


def merge_similar_tags_from_dataframe(input_file: str, output: str, variables: str, id: str, tag_length: int, threshold: float = 0.6, manual: bool = False, show_output: bool = False):
    """
    Merges similar tags in a DataFrame based on tag similarity using various deduplication methods.

    Parameters:
    -----------
    input_file (str): Path to the CSV file containing the data.

    output (str): Filename for an output csv. Data is not saved, but only returned, if left blank.

    variables (str): Comma-separated list of tag/column names.

    id (str): Column name containing unique titles/identifiers.

    tag_length (int): Length of tag n-grams for similarity comparison.

    threshold (float, optional): Similarity threshold for grouping tags. Default is 0.6.

    manual (bool): include manual check of potential duplicates by y/n prompt per pair to either merge or discard ('q' to escape). Default is False.

    show_output (bool): Whether to display intermediate output. Default is False.

    Returns:
    --------
    pd.DataFrame: DataFrame with similar tags merged using the specified deduplication methods.
    """
    def generate_tuple_combinations(nested_list: str) -> list:
        """
        Generate all possible combinations of tag pairs from a nested list of tags.
        """
        combination = []

        for item in nested_list:
            product = []
            product = list(combinations(item, 2))
            combination.extend(product)

        return combination

    def calculate_tag_similarity(Dataframe: pd.DataFrame, method: str, treshold: float) -> tuple[list, np.ndarray]:
        """
        Calculate tag similarity and deduplicate similar tags in the DataFrame.
        """
        tag_dissimilarity = generate_tag_dissimilarity(Dataframe)
        tag_names = Dataframe.columns
        grouped_tags = group_tags_by_dissimilarity(
            tag_dissimilarity, tag_names, treshold)
        deduplicated_tags = deduplicate_tag_conjugations(grouped_tags, method)

        return deduplicated_tags, tag_dissimilarity

    def manual_deduplication(tuples, dataframe):
        """
        Suggest potential duplicates and merge or discard based on user input"""
        deduplicated_dataframe = dataframe.copy()

        for source, sink in tuples:
            if source in deduplicated_dataframe.columns and sink in deduplicated_dataframe.columns:
                sourcedata = deduplicated_dataframe[source]
                sinkdata = deduplicated_dataframe[sink]
                answer = input(f"Merge {source} to {sink}? (y/n): ").lower()
                if answer == "y":
                    print(f"Merged {source} to {sink}")
                    sinkdata = sinkdata.add(sourcedata, fill_value=0)
                    deduplicated_dataframe[sink] = sinkdata
                    deduplicated_dataframe.drop(columns=[source], inplace=True)
                elif answer == "n":
                    break
                elif answer == "q":
                    return deduplicated_dataframe
                else:
                    print("Input invalid. Please answer with 'y' or 'n'")

        return deduplicated_dataframe

    matrix = generate_binary_item_matrix(
        input_file, variables, id, tag_length)[0]
    matrix = drop_0_columns(matrix)
    deduplicated_tags = calculate_tag_similarity(matrix, "pairs", threshold)[0]

    matrix_deduplicated = deduplicate_dataframe(
        matrix, deduplicated_tags, "strict")
    deduplicated_tags_control = calculate_tag_similarity(
        matrix_deduplicated, "", threshold)[0]
    potential_duplicates = generate_tuple_combinations(
        deduplicated_tags_control)

    matrix_deduplicated_lenient = deduplicate_dataframe(
        matrix, potential_duplicates, "lenient")
    deduplicated_tags_remaining = calculate_tag_similarity(
        matrix_deduplicated_lenient, "", threshold)[0]
    remaining_duplicates = [
        item for item in deduplicated_tags_remaining if len(item) > 1]
    remaining_duplicate_tuple = generate_tuple_combinations(
        remaining_duplicates)

    if manual:
        matrix_deduplicated_manual = manual_deduplication(
            remaining_duplicate_tuple, matrix_deduplicated_lenient)
    else:
        matrix_deduplicated_manual = matrix_deduplicated_lenient

    if show_output:
        deduplicated_tags_remaining, tag_dissimilarity3 = calculate_tag_similarity(
            matrix_deduplicated_manual, "", threshold)
        plt.figure()
        plt.imshow(tag_dissimilarity3)

    if output:
        matrix_deduplicated_manual.to_csv(output, sep=',')

    return matrix_deduplicated_manual


def count_tag_occurrence(dataframe: pd.DataFrame) -> np.array:
    # Assuming the binary columns start from column index 1
    binary_columns = dataframe.columns[1:]

    counts = []
    for column in binary_columns:
        count = dataframe[column].sum()
        if count > 0:
            counts.append((column, count))
    counts.sort(key=lambda x: x[1], reverse=True)
    return counts


def drop_0_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Delete any columns without observations from the given dataframe.
    """
    zero_columns = dataframe.columns[(dataframe == 0).all()]
    dataframe.drop(zero_columns, axis=1, inplace=True)
    return dataframe


def drop_unique_columns(Dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all single observation columns from a dataframe    
    """
    filtered_dataframe = Dataframe.copy()
    counts = count_tag_occurrence(Dataframe)
    for item, count in counts:
        if count == 1:
            filtered_dataframe.drop(columns=[item], inplace=True)
    return filtered_dataframe


def plot_pca_tags(data: pd.DataFrame, n_components_for_variance: int = 0, show_plots: str = ""):
    column_names = list(data.columns)
    # You need to define assign_group function
    groups = assign_group(data, column_names)
    group_list = set(groups)
    # You need to define transform_dataframe function
    X, y, targets, features = transform_dataframe(data, groups, group_list)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca.fit_transform(X_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    num_components_for_95_variance = np.argmax(cumulative_variance >= 0.95) + 1

    def plots(show_plots):
        plot_mapping = {
            "scree": {"function": plot_scree},
            "saturation": {"function": plot_saturation},
            "loading": {"function": plot_loading}
        }

        selected_plots = show_plots.split("and")
        num_subplots = len(selected_plots)

        fig, axes = plt.subplots(1, num_subplots, figsize=(18, 5))
        if num_subplots == 1:
            axes = [axes]

        for i, selected_plot in enumerate(selected_plots):
            plot_info = plot_mapping[selected_plot.strip()]
            plot_function = plot_info["function"]

            if num_subplots == 1:
                subplot_index = 0
            else:
                subplot_index = i

            ax = axes[subplot_index]
            plot_function(ax)

        plt.tight_layout()
        plt.show()

    def plot_scree(ax):
        ax.plot(range(1, len(explained_variance_ratio) + 1),
                explained_variance_ratio, marker='o')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('Scree Plot')

    def plot_saturation(ax):
        ax.plot(range(1, len(cumulative_variance) + 1),
                cumulative_variance, marker='o', color='r')
        ax.set_xlabel('Number of Principal Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('Saturation Plot')
        ax.text(y=0.05, x=0.5,
                s=f"Number of components needed for 95% explained variance: {num_components_for_95_variance}")

    def plot_loading(ax):
        loading_matrix = pca.components_.T * np.sqrt(pca.explained_variance_)
        for i, feature in enumerate(features):
            ax.arrow(0, 0, loading_matrix[i, 0],
                     loading_matrix[i, 1], color='r', alpha=0.5)
            ax.text(loading_matrix[i, 0],
                    loading_matrix[i, 1], feature, color='g')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('PC1 Loading')
        ax.set_ylabel('PC2 Loading')
        ax.set_title('Loading Plot')

    def find_contributing_tags(components: int):
        loading_matrix = pca.components_.T * np.sqrt(pca.explained_variance_)
        main_tags = []
        for loading in loading_matrix:
            indices_sorted_by_loading = np.argsort(np.abs(loading))[::-1]
            main_tags.append(features[indices_sorted_by_loading[0]])
        main_tags = sorted(set(main_tags[:components]))
        main_tags_str = ', '.join(main_tags)
        print(
            f'Main contributing tags for {components}% explained variance: {main_tags_str}')

    if "all" in show_plots:
        show_plots = "loading and scree and saturation"

    if n_components_for_variance > 0:
        find_contributing_tags(n_components_for_variance)

    plots(show_plots)
    plt.show()

    return


def main(input_file: str, output: str, variables: str, id: str, tag_length: int, n_components_for_variance: int, show_plots: str):
    Dataframe_merged = merge_similar_tags_from_dataframe(
        input_file, output, variables, id, tag_length)
    Dataframe_filtered = drop_unique_columns(Dataframe_merged)
    plot_pca_tags(Dataframe_filtered, n_components_for_variance, show_plots)


if __name__ == "__main__":
    main(input_file="C:/.../output/csv/total.csv", output="C:/NLPvenv/NLP/output/csv/total_deduplicated.csv",
         variables="Keywords", id="Article Title", tag_length=4, n_components_for_variance=40, show_plots="all")
