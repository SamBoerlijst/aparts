from scholarly import ProxyGenerator, scholarly
from scidownl import scihub_download

"""
Collect metadata from Google scholar for author and article by article title and/or author. Missing pdf files may be collected using scihub_download_pdf.
Metadata queries using the scholarly may take a minute to process.
Author: Sam P. Boerlijst
Date: 3/5/2023
"""

TXTfolder = "input/pdf/docs"

# setup proxy to stop ip from being blocked
pg = ProxyGenerator()
pg.FreeProxies()
scholarly.use_proxy(pg)


def lookup_author(author_name: str):
    """
    Retrieves metadata of a given author including, but not limited to, their full name.

    Parameters:
    -----------
    author_name: (Partial) name of the author.

    Return: 
    -----------
    article: Scholarly author type.
    """
    search_query = scholarly.search_author(author_name)
    author = next(search_query)
    return author


def get_author_metadata(author_name: str) -> dict:
    """
    Retrieves metadata for a given author and returns it as dictionary. Metadata includes: full name, affiliation, email domain, scholar id, fields of interest, url picture.

    Parameters:
    -----------
    author_name: (Partial) name of the author.

    Return: 
    -----------
    author_info (dict): Dictionary of author metadata

    """
    author = lookup_author(author_name)
    author_info = {
        'name': author["name"],
        'affiliation': author["affiliation"],
        'email_domain': author["email_domain"],
        'scholar_id': author["scholar_id"],
        'interests': author["interests"],
        'url_picture': author["url_picture"]
    }
    return author_info


def get_article(title: str) -> dict:
    """
    Retrieves metadata for an article with a given title. Metadata includes the fields: title, authors, year, journal and link.

    Parameters:
    -----------
    title (str): Title of the article.

    Return: 
    pub_data (dict): dictionary of metadata of the article.
    -----------

    """
    pub_data = {}
    pub = scholarly.search_single_pub(title, filled=True)
    bib = pub['bib']
    pub_data = {}
    pub_data['title'] = bib['title']
    pub_data['authors'] = bib['author']
    pub_data['year'] = bib['pub_year']
    pub_data['journal'] = bib['journal']
    pub_data['link'] = pub['pub_url']
    return pub_data


def get_author_publications(full_author_name: str) -> list:
    """
    Retrieves titles of articles (co-)written by a given author.

    Parameters:
    -----------
    full_author_name (str): Complete author name (acquired with lookup_author) of the article.

    Return: 
    -----------
    publication_list (list): List of publication titles.

    """
    publication_list = []
    author = lookup_author(full_author_name)
    scholarly.fill(author, sections=['publications'])
    publications = author['publications']
    for item in publications:
        title = item['bib']['title']
        publication_list.append(title)
    return publication_list


def get_author_bibliography(full_author_name: str) -> dict:
    """
    Collects metadata of all found articles for a given author. Metadata includes the fields: title, authors, year, journal and link.

    Parameters:
    -----------
    full_author_name (str): Complete author name (acquired with lookup_author) of the article.

    Return: 
    -----------
    bibliography (dict): Dictionary of all found articles.
    """
    bibliography = {}
    publications = get_author_publications(full_author_name)
    for title in publications:
        pub_data = get_article(title)
        bibliography[pub_data['title']] = pub_data
        print(pub_data)
    return bibliography


def get_article_by_author(full_author_name: str, title: str):
    """
    Retrieves metadata of an article by lookup of a title within an authors bibliography. Metadata includes the fields: title, authors, year, journal and link.

    Parameters:
    -----------
    full_author_name (str): complete author name (acquired with lookup_author) of the article.
    title (str): (partial) article title.

    Return: 
    -----------
    article: Scholarly article type.
    """
    publications = get_author_publications(full_author_name)
    for publication_title in publications:
        if title in publication_title:
            article = get_article(publication_title)
            break
    if article is None:
        print("No publication by this title was found for this author")
    return article


def scihub_download_pdf(paper: dict, output_folder: str = "/input/pdf") -> None:
    """
    Downloads a PDF of an article using the provided 'title' and 'link' fields from a dictionary, and saves it to the specified output folder with title as name.

    Parameters:
    -----------
    paper (dict): Dictionary containing the paper 'title' and 'link'.
    output_folder: The folder to store the pdf files in. Defaults to /input/pdf.

    Return:
    -----------
    None
    """
    title = paper['title']
    link = paper['link']
    destination = f"./{output_folder}/{title}.pdf"

    # Download the PDF from the article link using the scihub library
    try:
        scihub_download(keyword=link, out=destination)
    except Exception as e:
        print(f"Error: Unable to download PDF for article '{title}': {e}")
        return

    return


def main(author) -> None:
    """Downloads the first article of a given author."""
    first_article = {}
    publications = get_author_publications(author)
    print(f"downloading: {publications[1]}")
    first_article = get_article(publications[1])
    print(first_article)
    scihub_download_pdf(first_article)
    return


if __name__ == "__main__":
    main("Sam Philip Boerlijst")
