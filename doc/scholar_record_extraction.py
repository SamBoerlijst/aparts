from scholarly import ProxyGenerator, scholarly
import csv, os.path

# setup proxy to stop ip from being blocked
pg = ProxyGenerator()
pg.FreeProxies()
pg.rotate = True
scholarly.use_proxy(pg)

def download_articles(search_query: str, number: int = 100, year_range: list = [2000, 2020]) -> None:
    """
    Download article metadata to a csv based on a query. The function allows for intterruption at any time so that the number of articles may be extended later on.

    Parameters:
    -----------
    search_query (str): Query representative of field of research to search articles for.
    number (int): Number of articles to query and store.
    year_range (list): List stating the time period of articles to include.

    Returns:
    -----------
    None 
    """
    year_low = year_range[0]
    year_high = year_range[1]
    
    if os.path.exists('articles.csv'):
        with open('articles.csv', mode='r', encoding='utf-8') as file:
            rows = csv.reader(file)
            start_index = sum(1 for row in rows) - 1  # subtract 1 to exclude the header row


    search_results = scholarly.search_pubs(query=search_query, year_low=year_low, year_high=year_high, citations=False, start_index=start_index)

    headers = ['title', 'authors', 'year', 'abstract', 'journal', 'url']
    file_exists = os.path.isfile('articles.csv')

    with open('articles.csv', mode='a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer.writerow(headers)

        for i, pub in enumerate(search_results):
            if i < (number+start_index) :
                pub = scholarly.search_single_pub(pub['bib']['title'], filled=True)
                pub = scholarly.fill(pub)
                print(pub)
                title = pub['bib']['title']
                authors = pub['bib']['author']
                pub_year = pub['bib']['pub_year']
                journal = pub['bib']['journal']
                abstract = pub['bib']['abstract'].replace('\n', ' ')
                pub_url = pub['pub_url']
                writer.writerow([title, authors, pub_year, abstract, journal, pub_url])
                print(f'Article {i+start_index}/{number} downloaded and exported to CSV.')
            else:
                break

    print('Articles downloaded and exported to CSV.')
    return

if __name__ == "__main__":
    download_articles(search_query = "Culex AND population dynamics", number = 100)