from scholarly import ProxyGenerator, scholarly
import csv, os.path

# setup proxy to stop ip from being blocked
pg = ProxyGenerator()
pg.FreeProxies()
scholarly.use_proxy(pg)

def download_articles(search_query: str, number: int = 100, year_range: list = [2000, 2020], start_index: int = 0):
    year_low = year_range[0]
    year_high = year_range[1]
    search_results = scholarly.search_pubs(query=search_query, year_low=year_low, year_high=year_high, citations=False, start_index=start_index)

    headers = ['title', 'authors', 'year', 'abstract', 'url']
    file_exists = os.path.isfile('articles.csv')

    with open('articles.csv', mode='a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer.writerow(headers)

        for i, pub in enumerate(search_results):
            if i < number:
                pub = scholarly.fill(pub)
                title = pub['bib']['title']
                authors = pub['bib']['author']
                pub_year = pub['bib']['pub_year']
                abstract = pub['bib']['abstract'].replace('\n', ' ')
                pub_url = pub['pub_url']
                writer.writerow([title, authors, pub_year, abstract, pub_url])
                print(f'Article {i+1+start_index}/{number} downloaded and exported to CSV.')
            else:
                break

    print('Articles downloaded and exported to CSV.')
    return

if __name__ == "__main__":
    download_articles(search_query = "Culex AND population dynamics", number = 10)