import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.spatial.distance import braycurtis
import matplotlib.pyplot as plt

def generate_binary_matrix(CSV_path: str, y: str, x: str) -> tuple[pd.DataFrame, list]:
    """
    Generate a boolean matrix of items by respective tag presence from a csv.

    Parameters:
    -----------
    CSV_path (str): Path to the Excel file containing article metadata.

    y (str): Column name containing the keywords to use for dissimilarity.

    x (str): Column name of the article identifyers.

    Returns:
    --------
    binary_matrix_df (numpy.ndarray): boolean matrix of items by tag presence.

    rows_list (list): List of article identifyers.
    """
    def remove_empty_values(x: pd.Series):
        return x[x != '']
    dimensions = pd.read_csv(CSV_path)[y].str.replace(",", "").str.split(' ')
    dimensions = dimensions.astype(str)
    dimensions = remove_empty_values(dimensions)
    rows = pd.read_csv(CSV_path)[x]
    dimensions_uniques = sorted(
        list(set([dim for dim_list in dimensions for dim in dim_list])))
    rows_list = rows.tolist()
    binary_matrix = np.zeros((len(rows), len(dimensions_uniques)), dtype=int)

    for i, item in enumerate(dimensions):
        for j, dim in enumerate(dimensions_uniques):
            if dim in item:
                binary_matrix[i, j] = 1

    binary_matrix_df = pd.DataFrame(
        binary_matrix, columns=dimensions_uniques, index=rows)
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


def select_items_by_distance(distance_matrix: np.ndarray, n: int, continuous_dimensions: np.ndarray) -> list:
    """
    Select items based on the highest sum of distances to previously selected items.

    Parameters:
    -----------
    distance_matrix (np.ndarray): Euclidean distance matrix.

    n (int): Number of items to select.

    continuous_dimensions (np.ndarray): Matrix containing the coordinates of the samples.

    Returns:
    --------
    selected_items (list): List of selected item indices.
    """
    num_items = distance_matrix.shape[0]
    selected_items = []

    # Calculate the centroid of continuous_dimensions
    centroid = np.mean(continuous_dimensions, axis=0)

    # Find the index of the item closest to the centroid
    first_item = np.argmin(np.linalg.norm(continuous_dimensions - centroid, axis=1))
    selected_items.append(first_item)

    for _ in range(n - 1):
        item_sums = np.sum(distance_matrix[selected_items][:, np.newaxis], axis=0).ravel()
        item_sums[selected_items] = -np.inf  # Set previously selected items' sums to -inf
        next_item = np.argmax(item_sums)
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


def subsample_from_csv(CSV_path: str = "", y: str = "keywords", x: str = "title", n: int = 100):
    """
    Subsamples n papers by 3D bray-curtis dissimilarity based on their tags, and evenly selecting items across the euclidean space.

    Parameters:
    -----------
    CSV_path (str): Path to the Excel file containing article metadata.

    y (str): Column name containing the keywords to use for dissimilarity.

    x (str): Column name of the article identifyers.

    n (int): The number of articles to select.

    Returns:
    --------
    titles (list): List of identifyers for the selected items.
    """
    matrix, id_list = generate_binary_matrix(CSV_path, y, x)
    dissimilarity_array = generate_bray_curtis_dissimilarity(matrix)
    eucllidean_matrix = calculate_euclidean_distance_matrix(dissimilarity_array)
    indices = select_items_by_distance(eucllidean_matrix, n, dissimilarity_array)
    selected_array = get_selected_coordinates(indices, dissimilarity_array)
    plot_array(dissimilarity_array, selected_array, n, False)
    titles = get_sample_id(indices, id_list)
    return titles

def plot_array(totalset:np.array, selected_set:np.array, n:int, save_plot:bool)->None:
    def array_to_3d_array(array:np.array)-> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    a,b,c = array_to_3d_array(totalset)
    d,e,f = array_to_3d_array(selected_set)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a, b, -c, zdir='z', c= 'red')
    ax.scatter(d, e, -f, zdir='z', c= 'blue')
    if save_plot == True:
        plt.savefig(f"corpus and selection n{n}.png")
    return

if __name__ == "__main__":
    titles = subsample_from_csv(
        CSV_path="C:/NLPvenv/NLP/output/csv/total.csv", n=30)
    for item in titles:
        print(item)
