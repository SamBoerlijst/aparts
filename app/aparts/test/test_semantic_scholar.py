import os
import shutil
import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

import requests_mock

from aparts.src.semantic_scholar import (
    batch_collect_author_metadata, batch_collect_paper_metadata,
    batch_collect_recommendation_metadata, dict_to_csv,
    fetch_authors_semantic_scholar, fetch_metadata_semantic_scholar,
    fetch_query_semantic_scholar, fetch_recommendations_semantic_scholar,
    json_author_to_dict, json_paper_to_dict, json_recommendation_to_dict)

paper_normal = [
    {
        "externalIds": {"DOI": "10.1234/abcd"},
        "title": "Sample Paper",
        "authors": [{"name": "Author 1"}, {"name": "Author 2"}],
        "year": 2023,
        "abstract": "This is the abstract.",
        "tldr": {"text": "Too long; didn't read."},
        "journal": {"name": "Journal of Samples"},
        "fieldsOfStudy": ["Field 1", "Field 2"],
        "s2FieldsOfStudy": [{"category": "Category 1"}, {"category": "Category 2"}],
        "citationCount": 10,
        "openAccessPdf": {"url": "http://example.com/sample_paper.pdf"}
    }
]

recommendation_normal = [
    {
        "url": "https://example.com/paper1",
        "title": "Paper 1 Title",
        "authors": [
            {"name": "Author A"},
            {"name": "Author B"}
        ],
        "year": 2023,
        "abstract": "Abstract for Paper 1",
        "journal": {"name": "Journal 1"},
        "fieldsOfStudy": ["Field 1", "Field 2"],
        "s2FieldsOfStudy": [{"category": "Category 1"}, {"category": "Category 2"}],
        "citationCount": 10,
        "openAccessPdf": {"url": "https://example.com/paper1.pdf"},
        "source": "Source 1"
    },
    {
        "url": "https://example.com/paper2",
        "title": "Paper 2 Title",
        "authors": [
            {"name": "Author X"},
            {"name": "Author Y"}
        ],
        "year": 2022,
        "abstract": "Abstract for Paper 2",
        "journal": {"name": "Journal 2"},
        "fieldsOfStudy": ["Field 3"],
        "s2FieldsOfStudy": [{"category": "Category 3"}],
        "citationCount": 20,
        "openAccessPdf": {"url": "https://example.com/paper2.pdf"},
        "source": "Source 2"
    }
]

author_normal = [
    {
        "name": "John Doe",
        "aliases": ["Johnny D", "J. Doe"],
        "url": "http://example.com/johndoe",
        "authorId": "12345",
        "externalIds": ["abc123", "def456"],
        "paperCount": 10,
        "citationCount": 100,
        "hIndex": 5
    },
    {
        "name": "Jane Smith",
        "aliases": ["J. Smith"],
        "url": "http://example.com/janesmith",
        "authorId": "67890",
        "externalIds": ["xyz789"],
        "paperCount": 8,
        "citationCount": 80,
        "hIndex": 4
    }
]


class TestFetchPaper(unittest.TestCase):
    def test_json__paper_to_dict(self):
        input = paper_normal
        dict = json_paper_to_dict(input)
        paper_id = list(dict.keys())[0]
        paper_data = dict[paper_id][0]
        assert {'DOI': '10.1234/abcd'} == paper_data['externalIds']
        assert '10.1234/abcd' == paper_data['DOI']
        assert 'Sample Paper' == paper_data['title']
        assert 'Author 1' == paper_data['first_author']
        assert 'Author 1, Author 2' == paper_data['authors']
        assert 2023 == paper_data['year']
        assert 'This is the abstract.' == paper_data['abstract']
        assert "Too long; didn't read." == paper_data['tldr']
        assert 'Journal of Samples' == paper_data['journal']
        assert 'Field 1, Field 2' == paper_data['fieldsOfStudy']
        assert 'Category 1, Category 2' == paper_data['s2FieldsOfStudy']
        assert 10 == paper_data['citationCount']
        assert 'http://example.com/sample_paper.pdf' == paper_data['openAccessPdf']

    def test_json_paper_to_dict_near_empty(self):
        input = [{"externalIds": {"DOI": "10.1234/abcd"}}]
        corpus_data = json_paper_to_dict(input)
        paper_id = list(corpus_data.keys())[0]
        paper_data = corpus_data[paper_id][0]
        control = {'externalIds': {'DOI': '10.1234/abcd'}, 'DOI': '10.1234/abcd', 'title': '', 'authors': '', 'first_author': '', 'year': '',
                   'abstract': '', 'tldr': '', 'journal': '', 'fieldsOfStudy': '', 's2FieldsOfStudy': '', 'citationCount': '', 'openAccessPdf': ''}
        assert control == paper_data

    def test_json__paper_to_dict_empty(self):
        input = [{}]
        dict = json_paper_to_dict(input)
        assert {} == dict


class TestFetchRecommendations(unittest.TestCase):
    def test_json_recommendation_to_dict(self):
        input = recommendation_normal
        corpus_data = json_recommendation_to_dict(input)

        paper_id = list(corpus_data.keys())[0]
        paper_data = corpus_data[paper_id][0]
        assert 'https://example.com/paper1' == paper_data['url']
        assert 'Paper 1 Title' == paper_data['title']
        assert 'Author A' == paper_data['first_author']
        assert 'Author A, Author B' == paper_data['authors']
        assert 2023 == paper_data['year']
        assert 'Abstract for Paper 1' == paper_data['abstract']
        assert 'Journal 1' == paper_data['journal']
        assert 'Field 1, Field 2' == paper_data['fieldsOfStudy']
        assert 'Category 1, Category 2' == paper_data['s2FieldsOfStudy']
        assert 10 == paper_data['citationCount']
        assert 'https://example.com/paper1.pdf' == paper_data['openAccessPdf']
        assert 'Source 1' == paper_data['source']

    def test_json_recommendation_to_dict_near_empty(self):
        input = [{"url": "https://example.com/paper1"}]
        corpus_data = json_recommendation_to_dict(input)

        paper_id = list(corpus_data.keys())[0]
        paper_data = corpus_data[paper_id][0]
        control = {'url': 'https://example.com/paper1', 'title': '', 'first_author': '', 'authors': '', 'year': '', 'abstract': '',
                   'journal': '', 'fieldsOfStudy': '', 's2FieldsOfStudy': '', 'citationCount': '', 'openAccessPdf': '', 'source': ''}
        assert control == paper_data

    def test_json_recommendation_to_dict_empty(self):
        input = [{}]
        dict = json_recommendation_to_dict(input)
        assert {} == dict


class TestFetcAuthor(unittest.TestCase):
    def test_json_author_to_dict(self):
        input_data = author_normal
        corpus_data = json_author_to_dict(input_data)
        author_id_1 = list(corpus_data.keys())[0]
        author_data_1 = corpus_data[author_id_1][0]
        assert 'http://example.com/johndoe' == author_data_1['url']
        assert 'John Doe' == author_data_1['name']
        assert 'Johnny D' == author_data_1['alias']
        assert '12345' == author_data_1['authorId']
        assert ['abc123', 'def456'] == author_data_1['externalIds']
        assert 10 == author_data_1['paperCount']
        assert 100 == author_data_1['citationCount']
        assert 5 == author_data_1['hIndex']

    def test_json_author_to_dict_empty(self):
        input_data = [{}]
        corpus_data = json_author_to_dict(input_data)
        assert {} == corpus_data

    def test_json_author_to_dict_near_empty(self):
        input_data = [{"authorId": "12345"}]
        corpus_data = json_author_to_dict(input_data)
        paper_id = list(corpus_data.keys())[0]
        paper_data = corpus_data[paper_id][0]
        control = {'name': '', 'alias': '', 'url': '', 'authorId': '12345',
                   'externalIds': '', 'paperCount': '', 'citationCount': '', 'hIndex': ''}
        assert control == paper_data


class TestFetchMetadata(unittest.TestCase):

    def test_fetch_metadata_semantic_scholar(self):
        args = MagicMock()
        args.output = 'test_output.csv'
        doi_list = ['10.1016/j.scitotenv.2022.159716']
        fields = []

        with requests_mock.mock() as m:
            m.post('https://api.semanticscholar.org/graph/v1/paper/batch',
                   status_code=200, json={})
            result = fetch_metadata_semantic_scholar(args, doi_list, fields)
        self.assertIsNotNone(result)

    def test_fetch_authors_semantic_scholar(self):
        input_list = ['Sam Boerlijst']
        fields = []

        with requests_mock.mock() as m:
            m.get('https://api.semanticscholar.org/graph/v1/author/search',
                  status_code=200, json={'data': [{'name': 'Sam Boerlijst'}]})
            result = fetch_authors_semantic_scholar(input_list, fields)
        self.assertIsNotNone(result)

    def test_fetch_recommendations_semantic_scholar(self):
        input_list = ['649def34f8be52c8b66281af98ae884c09aef38b']
        fieldnames = ['title', 'authors', 'year']

        with requests_mock.mock() as m:
            m.get('https://api.semanticscholar.org/recommendations/v1/papers/forpaper/649def34f8be52c8b66281af98ae884c09aef38b?limit=10&fields=title%2Cauthors%2Cyear', status_code=200, json={})
            result = fetch_recommendations_semantic_scholar(
                input_list, fieldnames)
        self.assertIsNotNone(result)


class TestDictToCSV(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_outputs'
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_dict_to_csv(self):
        metadata = {
            'paper1': [
                {'externalIds': '123', 'title': 'Paper 1',
                    'authors': 'Author 1', 'year': '2023'},
                {'externalIds': '124', 'title': 'Paper 2',
                    'authors': 'Author 2', 'year': '2022'},
            ],
            'paper2': [
                {'externalIds': '125', 'title': 'Paper 3',
                    'authors': 'Author 3', 'year': '2021'},
            ]
        }
        output_path = os.path.join(self.test_dir, 'test_output.csv')
        dict_to_csv(metadata, output_path)
        self.assertTrue(os.path.exists(output_path))

        with open(output_path, 'r') as csvfile:
            content = csvfile.read()
            self.assertIn('123,,Paper 1,,Author 1,2023,,,,,,,', content)
            self.assertIn('124,,Paper 2,,Author 2,2022,,,,,,,', content)
            self.assertIn('125,,Paper 3,,Author 3,2021,,,,,,,', content)


def test_batch_collect_paper_metadata(self):
    output_path = os.path.join(self.temp_dir, 'output.csv')

    with patch('aparts.src.semantic_scholar.fetch_metadata_semantic_scholar') as mock_fetch_metadata:
        with patch('aparts.src.semantic_scholar.dict_to_csv') as mock_dict_to_csv:
            batch_collect_paper_metadata(self.input_path, output_path)

            expected_args = Namespace(
                output=output_path, input=self.input_path)

            mock_fetch_metadata.assert_called_with(
                expected_args,
                ['10.123/456', '10.789/123'],
                None
            )
            mock_dict_to_csv.assert_called_with(
                metadata=mock_fetch_metadata.return_value, fields=None, output=output_path
            )


class TestBatchCollectAuthorMetadata(unittest.TestCase):
    def setUp(self):
        self.temp_dir = 'temp_test_dir'
        os.makedirs(self.temp_dir, exist_ok=True)
        self.input_path = os.path.join(self.temp_dir, 'input.csv')
        with open(self.input_path, 'w') as f:
            f.write('doi\n10.123/456\n10.789/123\n')

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_batch_collect_author_metadata(self):
        output_path = os.path.join(self.temp_dir, 'output.csv')

        with patch('aparts.src.semantic_scholar.fetch_authors_semantic_scholar') as mock_fetch_authors:
            with patch('aparts.src.semantic_scholar.json_author_to_dict') as mock_json_author_to_dict:
                with patch('aparts.src.semantic_scholar.dict_to_csv') as mock_dict_to_csv:
                    batch_collect_author_metadata(self.input_path, output_path)

                    mock_fetch_authors.assert_called_with(
                        self.input_path,
                        ['name', 'aliases', 'url', 'authorId', 'externalIds',
                            'paperCount', 'citationCount', 'hIndex']
                    )
                    mock_json_author_to_dict.assert_called_with(
                        mock_fetch_authors.return_value)
                    mock_dict_to_csv.assert_called_with(
                        metadata=mock_json_author_to_dict.return_value,
                        fields=['name', 'alias', 'url', 'authorId',
                                'externalIds', 'paperCount', 'citationCount', 'hIndex'],
                        output=output_path
                    )


class TestBatchCollectRecommendationMetadata(unittest.TestCase):
    def setUp(self):
        self.temp_dir = 'temp_test_dir'
        os.makedirs(self.temp_dir, exist_ok=True)
        self.input_path = os.path.join(self.temp_dir, 'input.csv')
        with open(self.input_path, 'w') as f:
            f.write('doi\n10.123/456\n10.789/123\n')

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_batch_collect_recommendation_metadata(self):
        output_path = os.path.join(self.temp_dir, 'output.csv')

        with patch('aparts.src.semantic_scholar.fetch_recommendations_semantic_scholar') as mock_fetch_recommendations:
            with patch('aparts.src.semantic_scholar.json_recommendation_to_dict') as mock_json_recommendation_to_dict:
                with patch('aparts.src.semantic_scholar.dict_to_csv') as mock_dict_to_csv:
                    batch_collect_recommendation_metadata(
                        self.input_path, output_path)

                    mock_fetch_recommendations.assert_called_with(
                        self.input_path,
                        ['title', 'authors', 'year', 'abstract', 'journal', 'fieldsOfStudy',
                            's2FieldsOfStudy', 'citationCount', 'openAccessPdf', 'url']
                    )
                    mock_json_recommendation_to_dict.assert_called_with(
                        mock_fetch_recommendations.return_value)
                    mock_dict_to_csv.assert_called_with(
                        metadata=mock_json_recommendation_to_dict.return_value,
                        fields=['title', 'authors', 'year', 'abstract', 'journal', 'fieldsOfStudy',
                                's2FieldsOfStudy', 'citationCount', 'openAccessPdf', 'url', 'source'],
                        output=output_path
                    )


class TestFetchQuerySemanticScholar(unittest.TestCase):
    def test_fetch_query_semantic_scholar_default(self):
        result = fetch_query_semantic_scholar(
            'natural language processing', amount=50)
        self.assertIsInstance(result, dict)
        self.assertTrue(all(isinstance(k, str) and isinstance(v, list)
                        for k, v in result.items()))


if __name__ == '__main__':
    unittest.main()
