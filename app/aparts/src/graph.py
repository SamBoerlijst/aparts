import os

import pandas as pd
from pyvis.network import Network

from aparts.src.extract_references import (extract_references_from_file,
                                           extract_references_from_folder)

color_scheme = ["#87DE3C", "#7029A6", "#FF1288",
                "#FFEB0A", "#2B95FF", "#894fc0", "#FFFFFF"]


def find_value_and_delete_upper_level_entry(dictionary, value) -> dict:
    """
    Finds the index of a value and removes its (upper level) entry from a (nested) dictionary

    Parameters:
    -----------
    dictionary (dict): (Nested) dictionary.

    value (str): Name of the value from which the parent item should be removed.

    Returns:
    -----------
    new_dict (dict): Corrected dictionary. 
    """
    new_dict = {}

    for key, val in dictionary.items():
        if isinstance(val, dict):
            new_val = find_value_and_delete_upper_level_entry(val, value)
            if new_val:
                new_dict[key] = new_val
        else:
            if val != value:
                new_dict[key] = val

    return new_dict


def flatten_nested_dict_value_to_list(dictionary: dict, value: str) -> list:
    """
    Returns all entries within the provided value of a dictionary as an unordered list.

    Parameters:
    -----------
    dictionary (dict): (Nested) dictionary.

    value (str): Name of the value containing the data of interest.

    Returns:
    -----------
    item_list (list): Unordered list of all found values. 

    """
    item_list = []
    for item in dictionary.values():
        entry = f"{item[value]}"
        item_list.append(entry)
    return item_list


def remove_dead_links_from_reference_dict(check_dict: dict, check_key: str, compare_dict: dict, compare_key: str, feedback: bool) -> dict:
    """
    Compare items within a dictionary of source-destination relationships with a reference dictionary and remove 1) self-citations or 2) items without a parent.

    Parameters:
    -----------
    check_dict (dict): Dictionary containing items to correct.

    check_key: (str): Key within the compare dictionary containing the item identifyer.

    compare_dict: (dict): Dictionary containing items to use as reference.

    compare_key: (str): Key within the compare dictionary containing the item identifyer.

    feedback: Whether to look for positive or negative feedback. Positive removes values that occur in both lists (for instance in two columns of the same dictionary, resulting in a feedback loop). Negative removes values that occur only in the check_key.

    Returns:
    -----------
    new_dict (dict): Dictionary of corrected source-destination relationships.
    """
    new_dict = check_dict.copy()
    items_with_data = flatten_nested_dict_value_to_list(
        compare_dict, compare_key)
    items_to_check = flatten_nested_dict_value_to_list(check_dict, check_key)
    for item in items_to_check:
        if feedback == True:
            if item in items_with_data:
                new_dict = find_value_and_delete_upper_level_entry(
                    new_dict, item)
        if feedback == False:
            if item not in items_with_data:
                new_dict = find_value_and_delete_upper_level_entry(
                    new_dict, item)
    return new_dict


def file_name_to_title(filename: str, CSV: str) -> str:
    """
    Return the title for an item by lookup of their filename within the given Excel file.

    Parameters:
    -----------
    filename (str): Filename for a dataframe item.

    CSV (str): Path to the reference dataframe.

    Returns:
    -----------
    title (str): Corresponding reference title.
    """
    source_frame = pd.read_csv(CSV)
    row = source_frame.loc[source_frame['file'] == filename]
    title = row['title']
    return title


def replace_filenames_by_title(column_name: str, dictionary: dict, CSV: str) -> dict:
    """
    Substitute filenames by titles for each entry in the given column of a dictionary, using metadata from a CSV.

    Parameters:
    -----------
    column_name (str): Name of the column containing the filenames.

    dictionary (dict): Source item containing the filenames that should be replaced.

    CSV (str): Absolute path to the reference dataframe.

    Returns:
    -----------
    dictionary_fixed (dict): Item containing the (substituted) titles and associated data.
    """
    dictionary_fixed = {}
    try:
        for key, item in dictionary.items():
            filename = item[column_name]
            title = file_name_to_title(filename, CSV)
            current_item = item
            current_item[column_name] = title
            dictionary_fixed.update(current_item)
    except:
        None
    return dictionary_fixed


def collect_data_from_csv(CSV: str) -> dict:
    """
    Return relevant article metadata from a csv file as dictionary.

    Parameters:
    -----------
    CSV (str): Absolute path to the excel file containing references and their respective metadata.

    Returns:
    --------

    dataframe (dict): Dictionary containing each item and their respective metadata as index[title:str, year:str, authorlist:list, journal:str, filename:str, keywords:list]
    """
    source_frame = pd.read_csv(CSV)
    dataframe = {}
    for index, row in source_frame.iterrows():
        if str(row["title"]) != "nan":
            authorlist = str(row['author']).split(" and ")
            title = row['title']
            keyword_string = row['keywords']
            keyword_string = f"{keyword_string}"
            keyword_string.replace(",", "").replace(
                "nan", "").replace("/", "_")
            keywords = keyword_string.split(" ")
            keywords = [x for x in keywords if len(x) >= 3]
            keywords = [x for x in keywords if x != 'nan']
            year = str(row['date']).split("-")[0]
            year = year.replace("nan", "")
            journal = str(row['journaltitle'])
            filename = row["file"]
            index = index
            current_item = {index: {"title": title, "year": year, "authorlist": authorlist,
                                    "journal": journal, "filename": filename, "keywords": keywords}}
            dataframe.update(current_item)
    return dataframe


def parse_data_from_csv(CSV) -> tuple[dict, dict, dict, dict]:
    """
    Extracts relationships between reference and respective tag(s), year, author(s) or journal and returns these per group as dictionary.

    Parameters:
    -----------
    CSV (str): Absolute path to the excel file containing references and their respective metadata.

    Returns:
    -----------
    reference_data_tags (dict): Dictionary containing the linked data for parent to tags as reference - tag - color.

    reference_data_year (dict): Dictionary containing the linked data for parent to year as reference - year - color.

    reference_data_authors (dict): Dictionary containing the linked data for parent to authors as reference - author - color.

    reference_data_journal (dict):Dictionary containing the linked data for parent to journal as reference - journal - color.
    """
    dataframe = collect_data_from_csv(CSV)
    reference_data_tags = {}
    reference_data_year = {}
    reference_data_authors = {}
    reference_data_journal = {}
    for key, value in dataframe.items():
        source = value['title']
        for item in value['keywords']:
            index = len(reference_data_tags.items())+1
            current = {index: {"source": source,
                               "destination": item, "color": 1}}
            reference_data_tags.update(current)
        index = len(reference_data_year.items())+1
        current = {index: {"source": source,
                           "destination": value['year'], "color": 3}}
        reference_data_year.update(current)
        for item in value['authorlist']:
            index = len(reference_data_authors.items())+1
            current = {index: {"source": source,
                               "destination": item, "color": 4}}
            reference_data_authors.update(current)
        index = len(reference_data_journal.items())+1
        current = {index: {"source": source,
                           "destination": value['journal'], "color": 3}}
        reference_data_journal.update(current)
    return reference_data_tags, reference_data_year, reference_data_authors, reference_data_journal


def link_items_to_source(item_dict: dict, column_name: str, source: str, depth: int) -> tuple[dict, dict, dict]:
    """
    Create a dictionary linking items from a given dictionary column to the given source file.

    Parameters:
    -----------
    item_dict (dict): Dictionary containing a column of data to link.

    column_name (str): Name of the column containing the destination identifyer 'Title'.

    source (str): Name of the source file

    depth (bool): Include author and year data for each reference

    Returns:
    --------
    inter_reference_dict (dict): Dictionary containing the linked data for parent to citation as reference - citation_title - color.

    year_dict (dict): Dictionary containing the linked data for citation to year as citation_title - year - color relationships.

    author_dict (dict): Dictionary containing the linked data for citation to author as citation_title -  author - color relationships.
    """
    inter_reference_dict = {}
    year_dict = {}
    author_dict = {}
    for reference_value in item_dict:
        destination = reference_value[column_name]
        index = len(inter_reference_dict.items())+1
        color = 2
        current_item = {index: {"source": source,
                                "destination": destination, "color": color}}
        inter_reference_dict.update(current_item)
        if depth >= 2:
            index = len(year_dict.items())+1
            color = 3
            metadata_source = reference_value[column_name]
            destination = reference_value["Year"]
            current_item = {index: {"source": metadata_source,
                                    "destination": destination, "color": color}}
            year_dict.update(current_item)
            for item in reference_value["Authors"]:
                index = len(author_dict.items())+1
                color = 4
                destination = item
                current_item = {index: {"source": metadata_source,
                                        "destination": destination, "color": color}}
                author_dict.update(current_item)
    author_dict = remove_dead_links_from_reference_dict(
        author_dict, "source", inter_reference_dict, "destination", False)
    year_dict = remove_dead_links_from_reference_dict(
        year_dict, "source", author_dict, "source", False)
    inter_reference_dict = remove_dead_links_from_reference_dict(
        inter_reference_dict, "source", inter_reference_dict, "destination", True)
    return inter_reference_dict, year_dict, author_dict


def link_from_file(filepath: str, depth: int) -> dict:
    """
    Loop through the references section of a txt file and establish source-destination links for network analysis.

    Parameters:
    -----------
    filepath (str): Absolute path to the file from which citations should be indexed. 

    depth (int): Indicates whether citation-year and citation-author relationships should be generated also. True if > 2.

    Returns:
    --------
    inter_reference_dict (dict): Dictionary containing the linked data for parent to citation as reference - citation_title - color.

    year_dict (dict): Dictionary containing the linked data for citation to year as citation_title - year - color relationships.

    author_dict (dict): Dictionary containing the linked data for citation to author as citation_title -  author - color relationships.
    """
    item_dict = extract_references_from_file(filepath)
    column_name = "Title"
    source = os.path.basename(filepath)
    inter_reference_dict, year_dict, author_dict = link_items_to_source(
        item_dict, column_name, source, depth)
    return inter_reference_dict, year_dict, author_dict


def link_from_folder(folderpath: str, depth: int) -> tuple[dict, dict, dict]:
    """
    Loops over all txt files within a document to generate reference_title-citation_title, citation_title-year and citation_title-author relationships from APA6 formatted citations within the reference section of each document. 

    Parameters:
    -----------
    folderpath (str): Absolute path to the folder containing text files from which citations should be indexed. 

    depth (int): Indicates whether citation-year and citation-author relationships should be generated also. True if > 2.

    Returns:
    --------
    inter_reference_dict (dict): Dictionary containing the linked data for parent to citation as reference - citation_title - color.

    year_dict (dict): Dictionary containing the linked data for citation to year as citation_title - year - color relationships.

    author_dict (dict): Dictionary containing the linked data for citation to author as citation_title -  author - color relationships.
    """
    inter_reference_dict = {}
    year_dict = {}
    author_dict = {}
    column_name = "Title"
    ref_dict = extract_references_from_folder(folderpath)
    for item_key, item_value in ref_dict.items():
        source = item_value["item"]
        item_dict = item_value["reference"]
        inter_reference_dict_b, year_dict_b, author_dict_b = link_items_to_source(
            item_dict, column_name, source, depth)
        inter_reference_dict.update(inter_reference_dict_b)
        year_dict.update(year_dict_b)
        author_dict.update(author_dict_b)
    return inter_reference_dict, year_dict, author_dict


def create_network_lists(reference_dict: dict[int, dict]) -> list:
    """
    Generate source and destination lists from a source-destination dataframe. 

    Parameters:
    -----------
    reference_dict: Nested datagrame containing source, destination and color per index.

    Returns:
    --------
    source_list (list): List containing source nodes.

    destination_list (list): List containing destination nodes.

    color_list (list): List containing colours for the respective relationship.

    """
    color_list = []
    source_list = []
    destination_list = []
    for item_key, item_value in reference_dict.items():
        try:
            if len(item_value.keys()) == 3:
                source = item_value["source"]
                destination = item_value["destination"]
                color = item_value["color"]
                color_list.append(color)
                source_list.append(source)
                destination_list.append(destination)
        except:
            None
    return source_list, destination_list, color_list


def graph_view(totalCSV: str, path: str, height: str, width: str, depth: int, color_scheme: list, graph_name: str) -> None:
    """
    Generate a 2D node-network visualization of articles, relevant metadata (authors, journals, tags) and optionally their citations and respective authors. Saves the graph as html.

    Parameters:
    -----------
    totalCSV (str): Excelfile containing article metadata to include.

    path (str): Absolute path to the folder containing text files to scan for citations (see depth).

    height (str): Indicates how high the graph should be rendered, either in px or %, e.g. '1080xp'. 

    width (str): Indicates how wide the graph should be rendered, either in px or %, e.g. '1920px'. 

    depth (int): Indicates what level of data should be included with inreasing level of complexity: References in bib/csv file (1), citations within each reference (2), authors and year for each citation(3).

    color_scheme (list): Hexcode list of colours to render the graph with. 

    graph_name (str):

    Returns:
    --------
    None
    """
    def populate_graph(dataset: zip, size_source: int, size_destination: int, group_source: str, group_destination: str, level_source: int, level_destination: int) -> None:
        """
        Generate nodes and erdges thereof from the given source-destination zip.

        Parameters:
        -----------
        dataset (zip): Sourcefile containing the source-destination relationships and respective colors to use.

        size_source (int): Node size for the source node.

        size_destination (int): Node size for the source node.

        group_source (str): Node group for the source node.

        group_destination (str): Node group for the source node.

        level_source (int): Node level for the source node.

        level_destination (int): Node level for the source node.

        Returns:
        --------
        None

        """
        for e in dataset:
            source = e[0]
            destination = e[1]
            color = e[2]
            net.add_node(source, source, level=level_source,
                         color=color_scheme[color], size=size_source, borderWidthSelected=18, group=group_source, title=source)
            net.add_node(destination, destination, level=level_destination,
                         color=color_scheme[color], size=size_destination, borderWidthSelected=18, group=group_destination, title=destination)
            net.add_edge(source, destination, value=5, color=color_scheme[5])

    if os.path.isfile(path):
        inter_reference_dict, year_dict, author_dict = link_from_file(
            path, depth)
    elif os.path.isdir(path):
        inter_reference_dict, year_dict, author_dict = link_from_folder(
            path, depth)

    net = Network(height=height, width=width,
                  bgcolor="#222222", font_color="white", directed=True, filter_menu=True)
    net.barnes_hut()
    # net.force_atlas_2based(central_gravity=0.02, spring_length=60, spring_strength=0.2, overlap=1, gravity=-100)

    # substitute filenames by titlenames
    inter_reference_dict = replace_filenames_by_title(
        "source", inter_reference_dict, totalCSV)
    year_dict = replace_filenames_by_title("source", year_dict, totalCSV)
    author_dict = replace_filenames_by_title("source", author_dict, totalCSV)

    # inter_citation
    if depth >= 1:
        inter_reference_dict = replace_filenames_by_title(
            "destination", inter_reference_dict, totalCSV)
        sources, targets, color = create_network_lists(inter_reference_dict)
        inter_citation_data = zip(sources, targets, color)
        populate_graph(inter_citation_data, 6, 4,
                       "source_document", "citation", 2, 3)

    # intra_citation
    if depth >= 2:
        sources, targets, color = create_network_lists(year_dict)
        intra_citation_data_year = zip(sources, targets, color)
        populate_graph(intra_citation_data_year, 4, 2,
                       "citation", "citation_year", 3, 4)
        sources, targets, color = create_network_lists(author_dict)
        intra_citation_data_authors = zip(sources, targets, color)
        populate_graph(intra_citation_data_authors, 4,
                       2, "citation", "author", 3, 4)

    # reference
    reference_dict_tags, reference_dict_year, reference_dict_authors, reference_dict_journal = parse_data_from_csv(
        totalCSV)

    sources, targets, color = create_network_lists(reference_dict_tags)
    reference_data_tags = zip(sources, targets, color)
    populate_graph(reference_data_tags, 8, 6, "source_document", "tags", 1, 2)
    sources, targets, color = create_network_lists(reference_dict_year)
    reference_data_year = zip(sources, targets, color)
    populate_graph(reference_data_year, 8, 6, "source_document", "year", 1, 2)
    sources, targets, color = create_network_lists(reference_dict_authors)
    reference_data_authors = zip(sources, targets, color)
    populate_graph(reference_data_authors, 8, 6,
                   "source_document", "author", 1, 2)
    sources, targets, color = create_network_lists(reference_dict_journal)
    reference_data_journal = zip(sources, targets, color)
    populate_graph(reference_data_journal, 8, 6,
                   "source_document", "journal", 1, 2)

    neighbor_map = net.get_adj_list()
    # add neighbor data to node hover data
    for node in net.nodes:
        node["value"] = len(neighbor_map[node["id"]])
        node["title"] += " Neighbors:<br>" + \
            "<br>".join(neighbor_map[node["id"]])
    net.set_edge_smooth("dynamic")
    net.show_buttons('physics')
    net.show(f"{graph_name}.html", notebook=False)
    return


if __name__ == "__main__":
    graph_view("C:/NLPvenv/NLP/output/CSV/total.csv",
               "C:/NLPvenv/NLP/input/pdf/docs/corrected/test", "1080px", "100%", 3, color_scheme, 'network')
