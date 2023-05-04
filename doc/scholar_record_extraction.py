from scholarly import ProxyGenerator, scholarly
import pandas as pd

# setup proxy to stop ip from being blocked
pg = ProxyGenerator()
pg.FreeProxies()
scholarly.use_proxy(pg)

def download_articles(search_query:str, number:int = 100, year_range:list=[2000,2020]):
    refs = {}
    sections = number/20
    year_low = year_range[0]
    year_high = year_range[1]
    search_query = scholarly.search_pubs(query=search_query, year_low=year_low, year_high=year_high, citations = False)

    for item in search_query:
        result = item['bib']
        refs[result['title']] = result
    
    refs.to_csv('articles.csv', index=False)
    print(refs)
    print('Articles downloaded and exported to CSV.')

if __name__ == "__main__":
    download_articles("Culex AND population dynamics")