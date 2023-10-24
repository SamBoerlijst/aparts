import unittest
from unittest.mock import call, patch

import numpy as np
import pandas as pd

from aparts.src.subsampling import (assign_group,
                                    calculate_euclidean_distance_matrix,
                                    generate_binary_item_matrix,
                                    generate_bray_curtis_dissimilarity,
                                    plot_array, select_items_by_distance,
                                    transform_dataframe)


class TestGenerateBinaryItemMatrix(unittest.TestCase):
    def test_generate_binary_item_matrix(self):
        CSV_path = 'C:/NLPvenv/aparts/app/aparts/test/test.csv'
        keyword_length = 1
        delimiter = ', '

        binary_matrix_df, rows_list = generate_binary_item_matrix(
            CSV_path, keyword_length=keyword_length, delimiter=delimiter)

        self.assertIsInstance(binary_matrix_df, pd.DataFrame)
        self.assertIsInstance(rows_list, list)
        self.assertEqual(binary_matrix_df.shape[0], 3)
        self.assertEqual(binary_matrix_df.shape[1], 5)
        self.assertEqual(len(rows_list), len(set(rows_list)))


class TestGenerateBrayCurtisDissimilarity(unittest.TestCase):
    def test_generate_bray_curtis_dissimilarity(self):
        binary_matrix = pd.DataFrame({
            'Tag1': [1, 0, 1],
            'Tag2': [0, 1, 1],
            'Tag3': [1, 1, 0]
        })

        result = generate_bray_curtis_dissimilarity(binary_matrix)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (binary_matrix.shape[0], 3))
        self.assertEqual(result.shape, (binary_matrix.shape[1], 3))


class TestCalculateEuclideanDistanceMatrix(unittest.TestCase):
    def test_calculate_euclidean_distance_matrix(self):

        sample_matrix = np.array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
            [[13, 14, 15], [16, 17, 18]]
        ])
        result = calculate_euclidean_distance_matrix(sample_matrix)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(
            result.shape, (sample_matrix.shape[0], sample_matrix.shape[0]))


class TestSelectItemsByDistance(unittest.TestCase):
    def test_select_items_by_distance(self):
        sample_distance_matrix = np.array([
            [0.0, 14.69693846, 29.39387691], 
            [14.69693846,  0.0, 14.69693846], 
            [29.39387691, 14.69693846,  0.0]])
        sample_continuous_dimensions = np.array([
            [1, 2],
            [4, 5],
            [7, 8]
        ])
        result = select_items_by_distance(
            sample_distance_matrix, 2, sample_continuous_dimensions, "dissimilarity", "centroid")

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)


class TestPlotArray(unittest.TestCase):

    @patch('matplotlib.pyplot.savefig')
    def test_plot_array(self, mock_savefig):
        total_set = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        selected_set = np.array([[2, 3, 4], [5, 6, 7]])

        plot_array(total_set, selected_set, 2, True)

        mock_savefig.assert_called_once_with("corpus and selection n2.png")


class TestTransformDataFrame(unittest.TestCase):
    def test_transform_dataframe(self):
        # Create a sample DataFrame
        df = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Feature2': [4, 5, 6]
        })
        target = [0, 1, 0]
        target_list = ['ClassA', 'ClassB']
        data, returned_target, returned_target_names, returned_feature_names = transform_dataframe(df, target, target_list)

        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (3, 2))
        self.assertListEqual(returned_target, target)
        self.assertListEqual(returned_target_names, target_list)
        self.assertListEqual(returned_feature_names, df.columns.tolist())


class TestAssignGroup(unittest.TestCase):
    def test_assign_group(self):
        binary_dataframe = pd.DataFrame({
            'Item1': [1, 0, 0],
            'Item2': [0, 1, 0],
            'Item3': [0, 0, 1]
        })
        item_list = ['Item1', 'Item2', 'Item3']
        assigned_items = assign_group(binary_dataframe, item_list)
        self.assertIsInstance(assigned_items, list)
        self.assertListEqual(assigned_items, ['Item1', 'Item2', 'Item3'])



if __name__ == '__main__':
    unittest.main()
