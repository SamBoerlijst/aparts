import argparse
import csv
import os
from typing import Generator, List, TypeVar
from uuid import uuid1

import dotenv
import pandas as pd
import requests

dotenv.load_dotenv()
T = TypeVar('T')

S2_API_KEY = os.environ.get('S2_API_KEY', '')


def json_to_dict(response) -> dict:
    """
    Converts a list of metadata in JSON format to a dictionary.

    Parameters:
    -----------
    response (list): A list of JSON objects containing paper metadata.

    Returns:
    -----------
    corpus_data (dict): A dictionary where keys are unique identifiers for papers, and values are lists of dictionaries containing paper metadata.
    """
    corpus_data = {}
    for paper in response:
        if not paper:
            continue
        paper_authors = paper.get('authors', [])
        paper_data = {
            'externalIds': paper.get('externalIds', ''),
            'DOI': paper['externalIds'].get('DOI', ''),
            'title': paper.get('title', ''),
            'first_author': paper_authors[0].get('name', ''),
            'authors': ', '.join([author['name'] for author in paper_authors]),
            'year': paper.get('year', ''),
            'abstract': paper.get('abstract', ''),
            'tldr': paper.get('tldr', {}).get('text', '') if paper.get('tldr', {}) else '',
            'journal': paper.get('journal', {}).get('name', '') if paper.get('journal', {}) else '',
            'fieldsOfStudy': ', '.join([studyfield for studyfield in paper.get('fieldsOfStudy', '')]) if paper.get('fieldsOfStudy', '') else '',
            's2FieldsOfStudy': ', '.join([studyfield['category'] for studyfield in paper.get('s2FieldsOfStudy', '')]) if paper.get('s2FieldsOfStudy', '') else '',
            'citationCount': paper.get('citationCount', ''),
            'openAccessPdf': paper['openAccessPdf']['url'] if paper['openAccessPdf'] else '',
        }
        ID = str(uuid1())
        corpus_data[ID] = []
        corpus_data[ID].append(paper_data)
    return corpus_data


def fetch_metadata_semantic_scholar(args: argparse.Namespace, input:list, fieldnames: list = "") -> dict[str, any]:
    """
    Fetches metadata for a list of DOIs from the Semantic Scholar API and stores it in a CSV file.

    Parameters:
    -----------
    args (argparse.Namespace): An argparse.Namespace object containing command-line arguments and the path to the output CSV file where metadata will be stored.

    input (list): List of DOI or other identifyers used to search for the papers,

    fieldnames (list): A list of fields to include in the CSV file.

    Returns:
    -----------
    None
    """

    def batched(items: List[T], batch_size: int) -> List[List[T]]:
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    def get_paper_batch(session: requests.Session, ids: List[str], fields: str = 'paperId,title', **kwargs) -> List[dict]:
        params = {
            'fields': fields,
            **kwargs,
        }
        headers = {
            'X-API-KEY': S2_API_KEY,
        }
        body = {
            'ids': ids,
        }

        url = 'https://api.semanticscholar.org/graph/v1/paper/batch'
        response = session.post(url, params=params, headers=headers, json=body)
        response.raise_for_status()
        return response.json()

    def get_papers(ids: List[str], batch_size: int = 20, **kwargs) -> Generator[dict, None, None]:
        with requests.Session() as session:
            for ids_batch in batched(ids, batch_size=batch_size):
                yield from get_paper_batch(session, ids_batch, **kwargs)

    if fieldnames == "":
        fieldnames = ['DOI', 'title', 'first_author',
                      'year', 'abstract', 'journal', 'tldr']

    dois = input

    with open(args.output, 'w') as csvfile:
        ids = [f'DOI:{DOI}' for DOI in dois]
        fields = 'externalIds,title,authors,year,abstract,tldr,journal,fieldsOfStudy,s2FieldsOfStudy,citationCount,openAccessPdf'
        response = get_papers(ids, fields=fields)
        corpus_data = json_to_dict(response)
        count = len(corpus_data.keys())
    print(f'Fetched {count} results to {args.output}')

    return corpus_data


def dict_to_csv(metadata: dict, output: str, fields: list = None) -> None:
    """
    Write a dictionary of paper metadata to a CSV file, allowing for multiple metadata entries per paper.

    Parameters:
    -----------
    metadata (dict): A dictionary where keys represent unique identifiers for papers, and values are lists of dictionaries containing paper metadata. Each dictionary in the list should have keys corresponding to the fields to be included in the CSV file.

    fields (list): A list of field names to include as columns in the CSV file. The function will write data only for the specified fields. If this list is None, it will default to a predefined list of fields. These include: 'externalIds, title, authors, year, abstract, tldr, journal, fieldsOfStudy, s2FieldsOfStudy, citationCount, and openAccessPdf'.

    output (str): The path to the output CSV file where the metadata will be stored.

    Returns:
    -----------
    None
    """
    if fields is None:
        fields = ['externalIds', 'DOI', 'title', 'first_author', 'authors', 'year', 'abstract',
                  'tldr', 'journal', 'fieldsOfStudy', 's2FieldsOfStudy', 'citationCount', 'openAccessPdf']

    with open(output, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for paper_list in metadata.values():
            for paper in paper_list:
                # Only encode and decode string values
                row = {key: value.encode('utf-8').decode('utf-8') if isinstance(
                    value, str) else value for key, value in paper.items() if key in fields}
                writer.writerow(row)

    print(f'Wrote results to {output}')


def batch_collect_paper_metadata(input: str, output: str, fieldnames: list = None) -> None:
    """
    Collects paper metadata for a list of DOIs and stores it in a CSV file using the Semantic Scholar API.

    Parameters:
    -----------
    input (str, optional): The path to the input file containing DOIs. Default is 'pmid-p53-set.txt'.
    output (str, optional): The path to the output CSV file where metadata will be stored. Default is 'papers.csv'.
    fieldnames (list, optional): A list of field names to include in the CSV file. Default is 'fieldnames'

    Returns:
    -----------
    None
    """
    input_list = pd.read_csv(input)['DOI']
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default=output)
    parser.add_argument('input', nargs='?', default=input)
    args = parser.parse_args()
    corpus = fetch_metadata_semantic_scholar(args, input_list, fieldnames)
    dict_to_csv(metadata=corpus, fields=fieldnames, output=args.output)
    return


def fetch_query_semantic_scholar(query: str, fields: str = "", amount: int = 200) -> dict:
    """
    Fetches paper metadata from the Semantic Scholar API based on a query and specified fields.

    Parameters:
    -----------
    query (str, optional): The query string used to search for papers. If not provided, the function will prompt the user for input.
    fields (str, optional): A comma-separated string specifying the fields to include in the fetched paper metadata. If not provided, a default set of fields will be used. These include externalIds, title, authors, year, abstract, tldr, journal, fieldsOfStudy, s2FieldsOfStudy, citationCount and openAccessPdf.
    amount (int, optional): The maximum number of papers to retrieve. Defaults to 200.

    Returns:
    -----------
    corpus_data (dict): A dictionary where keys are unique identifiers for papers, and values are lists of dictionaries containing paper metadata.
    """
    if fields == "":
        fields = 'externalIds,title,authors,year,abstract,tldr,journal,fieldsOfStudy,s2FieldsOfStudy,citationCount,openAccessPdf'

    papers = None
    while not papers:
        if query == "":
            query = input('Find papers about what: ')
            if not query:
                continue

        rsp = requests.get('https://api.semanticscholar.org/graph/v1/paper/search',
                           headers={'X-API-KEY': S2_API_KEY},
                           params={'query': query, 'limit': amount, 'fields': 'externalIds,title,authors,year,abstract,tldr,journal,fieldsOfStudy,s2FieldsOfStudy,citationCount,openAccessPdf'})
        rsp.raise_for_status()
        results = rsp.json()
        total = results["total"]
        if not total:
            print('No matches found. Please try another query.')
            continue

        print(f'Found {total} results. Showing up to {amount}.')
        papers = results['data']
        corpus_data = json_to_dict(papers)
    return corpus_data


def query_to_csv(query: str, output: str, amount: int = 200) -> None:
    data = fetch_query_semantic_scholar(query=query, amount=amount)
    dict_to_csv(data, output=output)
    return


if __name__ == '__main__':
    batch_collect_paper_metadata(input='papers.csv', output='papers.csv')
    # query_to_csv(query="Culex pipiens AND population dynamics", output = 'papers.csv')
