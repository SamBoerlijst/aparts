import requests


class Taxon:
    def __init__(self, taxon_data: dict):
        self.usageKey = taxon_data.get('usageKey', {})
        self.scientificName = taxon_data.get('scientificName', {})
        self.canonicalName = taxon_data.get('canonicalName', {})
        self.rank = taxon_data.get('rank', {})
        self.status = taxon_data.get('status', {})
        self.confidence = taxon_data.get('confidence', {})
        self.matchType = taxon_data.get('matchType', {})
        self.kingdom = taxon_data.get('kingdom', {})
        self.phylum = taxon_data.get('phylum', {})
        self.order = taxon_data.get('order', {})
        self.family = taxon_data.get('family', {})
        self.genus = taxon_data.get('genus', {})
        self.species = taxon_data.get('species', {})
        self.kingdomKey = taxon_data.get('kingdomKey', {})
        self.phylumKey = taxon_data.get('phylumKey', {})
        self.classKey = taxon_data.get('classKey', {})
        self.orderKey = taxon_data.get('orderKey', {})
        self.familyKey = taxon_data.get('familyKey', {})
        self.genusKey = taxon_data.get('genusKey', {})
        self.speciesKey = taxon_data.get('speciesKey', {})
        self.synonym = taxon_data.get('synonym', [])
        self.taxonclass = taxon_data.get('class', {})
        self.description = taxon_data.get('description', {})

    def asstr(self) -> str:
        total_name = f"{self.kingdom}/{self.phylum}/{self.order}/{self.family}/{self.genus}/{self.species}"
        total_name = total_name.replace(" ", "_")
        return total_name

    def asdict(self) -> dict:
        return {'usageKey': self.usageKey, 'scientificName': self.scientificName, 'canonicalName': self.canonicalName, 'rank': self.rank,
                'status': self.status, 'confidence': self.confidence, 'matchType': self.matchType, 'total_name': self.asstr(), 'kingdom': self.kingdom, 'phylum': self.phylum,
                'order': self.order, 'family': self.family, 'genus': self.genus, 'species': self.species, 'kingdomKey': self.kingdomKey,
                'phylumKey': self.phylumKey, 'classKey': self.classKey, 'orderKey': self.orderKey, 'familyKey': self.familyKey,
                'genusKey': self.genusKey, 'speciesKey': self.speciesKey, 'synonym': self.synonym, 'class': self.taxonclass, 'description': self.description}


def fetch_taxon_gbif(query: str):
    """
    Fetches taxon metadata on a taxon in json formal from the gbif api.

    Parameters:
    -----------
    query (str, optional): The query string used to search for papers. If not provided, the function will prompt the user for input.

    Returns:
    -----------
    results (dict): A json dictionary on the taxon its phylogeny.
    """
    if query == "":
        query = input('Find taxon data for: ')
        if not query:
            return {}

    description = ''
    rsp = requests.get('https://api.gbif.org/v1/species/match',
                       params={'name': query})
    rsp.raise_for_status()
    results = rsp.json()
    species_name, description = fetch_near_taxon_gbif(query)
    if results['matchType'] == 'NONE':
        if species_name == '':
            print(f'No match found for \"{query}\"')
            return {}
        else:
            print(
                f'No exact match found for \"{query}\". Trying nearest match: \"{species_name}\".')
            results = fetch_taxon_gbif(species_name)
    results['description'] = description
    return results


def fetch_near_taxon_gbif(query: str, rank: str = 'species'):
    """
    Fetches scientific name and description of the first taxon match in json formal from the gbif api.

    Parameters:
    -----------
    query (str, optional): The query string used to search for papers. If not provided, the function will prompt the user for input.

    rank (str, optional):"
    Returns:
    -----------
    results (dict): A json dictionary on the taxon its phylogeny.
    """
    def largest_in_dict(dict) -> tuple[str, str]:
        if not dict:
            return "", ""
        item = max(dict, key=lambda k: len(dict[k]))
        return dict[item], item

    def nested_list_to_dict(results, range_i: int = 3) -> dict:
        description_dict = {}
        if range_i > len(results):
            range_i = len(results)
        for i in range(range_i):
            item = results[i]['descriptions']
            for j in range(len(item)):
                description_species = results[i]['species']
                description = results[i]['descriptions'][j]['description']
                description_dict[f'{description_species}_{j}'] = description
        return description_dict

    if query == "":
        query = input('scientific name and description for: ')
        if not query:
            return {}

    rsp1 = requests.get('https://api.gbif.org/v1/species/search',
                        params={'q': query, 'rank': rank, 'strict': 'true'})
    rsp1.raise_for_status()
    results_total = rsp1.json()

    if not results_total['results']:
        print('no matches found')
        return "", ""
    else:
        species = results_total['results'][0]['species'] if results_total['results'] else ''
        description_dict = nested_list_to_dict(results_total['results'])
        selected_description, selected_species = largest_in_dict(
            description_dict)
        selected_species = selected_species[:-2]
        # if selected_species != query: print(f"using description for {selected_species}")
        return species, selected_description


def collect_taxon_metadata(query, taxon_dict: dict) -> dict:
    """
    Fetches taxon metadata from the Semantic Scholar API based on a query and specified fields.

    Parameters:
    -----------
    query (str, optional): The query string used to search for papers. If not provided, the function will prompt the user for input.

    taxon_dict (dict): The dictionary to store the results in.

    Returns:
    -----------
    taxon_data (dict): A dictionary where keys are the species name, and values are dictionaries containing taxon metadata.
    """
    if isinstance(query, list):
        for item in query:
            collect_taxon_metadata(item, taxon_dict)
    else:
        json_data = fetch_taxon_gbif(query)

        if json_data['matchType'] != 'EXACT':
            print('Match might be wrong. Please check the collected record')

        taxon = Taxon(json_data).asdict()
        current_name = taxon['canonicalName']

        if current_name not in taxon_dict:
            taxon_dict[current_name] = taxon
        else:
            if not taxon_dict[current_name]['usageKey'] == taxon['usageKey']:
                current_name = query
                taxon_dict[current_name] = taxon
            else:
                for key in taxon_dict[current_name]:
                    if taxon_dict[current_name][key] == '' and taxon[key] != '':
                        taxon_dict[current_name][key] = taxon[key]

                if len(taxon_dict[current_name]['description']) < len(taxon['description']):
                    taxon_dict[current_name]['description'] = taxon['description']

        taxon_data = taxon_dict[current_name]
        return taxon_data


if __name__ == '__main__':
    taxon_dict = {}
    query_list = ['culex pipiens', 'Aedes aegypti', 'Ae albopictus']
    collect_taxon_metadata(query_list, taxon_dict)
    print(taxon_dict)
    #for item in taxon_dict:
    #    print(taxon_dict[item]['total_name'])
