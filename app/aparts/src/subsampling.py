import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.spatial.distance import braycurtis
from sklearn.cluster import KMeans


def generate_binary_item_matrix(CSV_path: str = "", y: str = "keywords", x: str = "title", keyword_length: int = 3, number_of_records: int = "", delimiter: str = ", ", separator: str = ";") -> tuple[pd.DataFrame, list]:
    """
    Generate a boolean matrix of items by respective tag presence from a csv.

    Parameters:
    -----------
    CSV_path (str): Path to the Excel file containing article metadata.

    y (str): Column name containing the keywords to use for dissimilarity.

    x (str): Column name of the article identifyers.

    keyword_length (int): Minimum tag length.

    number_of_records (int): Select only the top n records.

    delimiter (str): Separator used to delimit tags.

    Returns:
    --------
    binary_matrix_df (numpy.ndarray): boolean matrix of items by tag presence.

    rows_list (list): List of article identifyers.
    """
    dimensions = pd.read_csv(CSV_path, sep = separator)[y]
    if number_of_records:
        dimensions = dimensions.head(number_of_records)
        
    dimensions = dimensions.astype(str).sum()
    dimensions = str(dimensions).split(delimiter)
    dimensions = [item.lstrip(' ').rstrip(' ') for item in dimensions]
    dimensions_uniques = sorted(set(dimensions))
    dimensions_filtered = [
        item for item in dimensions_uniques if len(item) >= keyword_length]
    rows = pd.read_csv(CSV_path, sep = separator)[x]
    rows_list = rows.tolist()
    binary_matrix = np.zeros((len(rows), len(dimensions_filtered)), dtype=int)

    for i, item in enumerate(rows):
        for j, dim in enumerate(dimensions_filtered):
            if dim in dimensions[i]:
                binary_matrix[i, j] = 1

    # Remove rows with only 0 values
    non_zero_rows = np.any(binary_matrix, axis=1)
    binary_matrix = binary_matrix[non_zero_rows]
    rows_list = [row for i, row in enumerate(rows_list) if non_zero_rows[i]]

    # Update indices
    binary_matrix_df = pd.DataFrame(
        binary_matrix, columns=dimensions_filtered)
    binary_matrix_df.index = rows_list

    return binary_matrix_df, rows_list


def generate_bray_curtis_dissimilarity(binary_matrix: pd.DataFrame) -> np.ndarray:
    """
    Calculate 3D bray-curtis dissimilarities for items in a binary matrix.

    Parameters:
    -----------
    binary_matrix (numpy.ndarray): boolean matrix of items by tag presence.

    Returns:
    --------
    continuous_dimensions (numpy.ndarray): 3D dissimilarity matrix of continuous values.
    """
    num_rows = binary_matrix.shape[0]
    dissimilarity_matrix = np.zeros((num_rows, num_rows))

    for i in range(num_rows):
        for j in range(i+1, num_rows):
            dissimilarity = braycurtis(
                binary_matrix.iloc[i], binary_matrix.iloc[j])
            dissimilarity_matrix[i, j] = dissimilarity
            dissimilarity_matrix[j, i] = dissimilarity

    eigenvalues, eigenvectors = np.linalg.eigh(dissimilarity_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    pca_coordinates = eigenvectors[:, sorted_indices[:3]]

    rng = default_rng(42)
    resampled_indices = rng.choice(
        len(pca_coordinates), size=len(pca_coordinates), replace=True)
    continuous_dimensions = pca_coordinates[resampled_indices]
    rng.shuffle(continuous_dimensions)

    return continuous_dimensions


def calculate_euclidean_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the Euclidean distance matrix for a 3D numpy array.

    Parameters:
    -----------
    matrix (np.ndarray): 3D numpy array.

    Returns:
    --------
    distance_matrix (np.ndarray): Euclidean distance matrix.
    """
    num_items = matrix.shape[0]
    distance_matrix = np.zeros((num_items, num_items))

    for i in range(num_items):
        for j in range(i + 1, num_items):
            distance = np.linalg.norm(matrix[i] - matrix[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


def select_items_by_distance(distance_matrix: np.ndarray, n: int, continuous_dimensions: np.ndarray, distance_type: str, start_item: str) -> list:
    """
    Select items based on the highest sum of distances to previously selected items.

    Parameters:
    -----------
    distance_matrix (np.ndarray): Euclidean distance matrix.

    n (int): Number of items to select.

    continuous_dimensions (np.ndarray): Matrix containing the coordinates of the samples.

    distance_type (str): Indicates whether articles should be chosen based on "dissimilarity" or "similarity".

    start_item (str): Indicates whether the first selected item should be the one closest to the "centroid", or in the middle of the largest "cluster".

    Returns:
    --------
    selected_items (list): List of selected item indices.
    """
    type_list = ["dissimilarity", "similarity"]
    start_list = ["centroid", "cluster"]
    assert distance_type in type_list, "Please use a distance_type equal to 'dissimilarity' or 'similarity'"
    assert start_item in start_list, "Please use a start_item equal to 'centroid' or 'cluster'"
    num_items = distance_matrix.shape[0]
    selected_items = []

    # Calculate the centroid of continuous_dimensions
    centroid = np.mean(continuous_dimensions, axis=0)

    # Find the index of the item closest to the centroid
    if start_item == "centroid":
        first_item = np.argmin(np.linalg.norm(
            continuous_dimensions - centroid, axis=1))
        selected_items.append(first_item)
    elif start_item == "cluster":
        kmeans = KMeans(n_clusters=5)  # Set the appropriate number of clusters
        labels = kmeans.fit_predict(continuous_dimensions)
        cluster_sizes = np.bincount(labels)
        largest_cluster_index = np.argmax(cluster_sizes)
        largest_cluster_items = np.where(labels == largest_cluster_index)[0]
        largest_cluster_centroid = np.mean(
            continuous_dimensions[largest_cluster_items], axis=0)
        starting_item = largest_cluster_items[np.argmin(np.linalg.norm(
            continuous_dimensions[largest_cluster_items] - largest_cluster_centroid, axis=1))]
        selected_items.append(starting_item)

    for _ in range(n - 1):
        item_sums = np.sum(
            distance_matrix[selected_items][:, np.newaxis], axis=0).ravel()
        # Set previously selected items' sums to -inf
        item_sums[selected_items] = -np.inf

        masked_sums = np.ma.array(item_sums, mask=False)
        # Mask previously selected items
        masked_sums[selected_items] = np.ma.masked

        if distance_type == "dissimilarity":
            next_item = np.ma.argmax(masked_sums)
        elif distance_type == "similarity":
            next_item = np.ma.argmin(masked_sums)

        selected_items.append(next_item)

    return selected_items


def get_selected_coordinates(selected_items: list, distance_matrix: np.ndarray) -> np.ndarray:
    """
    Get the coordinates for each sample in the list of selected item indices.

    Parameters:
    -----------
    selected_items (list): List of selected item indices.

    distance_matrix (np.ndarray): Euclidean distance matrix.

    Returns:
    --------
    selected_coordinates (np.ndarray): Matrix containing the coordinates of the selected samples.
    """
    selected_coordinates = distance_matrix[selected_items]

    return selected_coordinates


def get_sample_id(indices: list, id_list: list) -> list:
    """
    Collect respective id values by index.

    Parameters:
    -----------
    indices (list): List of sample positions within the dataframe.

    id_list (list): List of identifyers within the dataframe.


    Returns:
    --------
    names (list): List of identifyers for the selected items.
    """
    names = []
    for index in indices:
        item_id = id_list[index]
        names.append(item_id)
    return names


def subsample_from_csv(CSV_path: str = "", y: str = "keywords", x: str = "title", n: int = 100, save_plot: bool = False, distance_type: str = "dissimilarity", start_item: str = "centroid"):
    """
    Subsamples n papers by 3D bray-curtis dissimilarity based on their tags, and evenly selecting items across the euclidean space.

    Parameters:
    -----------
    CSV_path (str): Path to the Excel file containing article metadata.

    y (str): Column name containing the keywords to use for dissimilarity.

    x (str): Column name of the article identifyers.

    n (int): The number of articles to select.

    save_plot (bool): Indicates whether the 3D plot of corpus and selection should be saved to the current folder as png.

    distance_type (str): Indicates whether articles should be chosen based on "dissimilarity" or "similarity".

    start_item (str): Indicates whether the first selected item should be the one closest to the "centroid", or in the middle of the largest "cluster".

    Returns:
    --------
    titles (list): List of identifyers for the selected items.
    """
    matrix, id_list = generate_binary_item_matrix(CSV_path, y, x)
    dissimilarity_array = generate_bray_curtis_dissimilarity(matrix)
    eucllidean_matrix = calculate_euclidean_distance_matrix(
        dissimilarity_array)
    indices = select_items_by_distance(
        eucllidean_matrix, n, dissimilarity_array, distance_type, start_item)
    selected_array = get_selected_coordinates(indices, dissimilarity_array)
    plot_array(dissimilarity_array, selected_array, n, save_plot)
    titles = get_sample_id(indices, id_list)
    return titles


def plot_array(total_set: np.array, selected_set: np.array, n: int, save_plot: bool) -> None:
    """
    Generate a 3D plot of the selected articles superimposed over the corpus.

    Parameters:
    -----------
    total_set (np.array): 3D dissimilarity matrix of continuous values for the entire corpus.

    selected_set (np.array): 3D dissimilarity matrix of continuous values for the selected articles.

    n (int): The number of selected_articles (to include in filename).

    save_plot (bool): Indicates whether the 3D plot of corpus and selection should be saved to the current folder as png.

    Returns:
    --------
    None
    """
    def array_to_3d_array(array: np.array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = []
        y = []
        z = []
        for item in array:
            a, b, c = item
            x.append(a)
            y.append(b)
            z.append(c)
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        return x, y, z
    a, b, c = array_to_3d_array(total_set)
    d, e, f = array_to_3d_array(selected_set)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a, b, -c, zdir='z', c='red', alpha=0.3)
    ax.scatter(d, e, -f, zdir='z', c='black')
    if save_plot == True:
        plt.savefig(f"corpus and selection n{n}.png")
    return


def transform_dataframe(df: pd.DataFrame, target: list, target_list: list) -> tuple[np.ndarray, list, list, list]:
    data = df.values
    target_names = target_list
    feature_names = df.columns.tolist()

    return data, target, target_names, feature_names


def assign_group(binary_dataframe, item_list):
    assigned_items = []
    for index, row in binary_dataframe.iterrows():
        for item in item_list:
            if row[item] == 1:
                assigned_items.append(item)
                break
    return assigned_items