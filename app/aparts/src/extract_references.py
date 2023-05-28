from aparts.src.weighted_tagging import split_text_to_sections, print_nested_dict
from aparts.src.APT import list_filenames
import re

def extract_references(text):
    authorname_pattern = r"(?<!\()\b([A-Z][a-zA-Z]+,\s[A-Z](?:\.[A-Z])*(?:,\s[A-Z](?:\.[A-Z])*)*)"
    year_pattern = r"(?:1[0-9]|20|21)\d{2}"
    title_pattern = r"\.\s(.+?)\."

    data_list = []

    while True:
        reference_data = {'Authors': [], 'Year': '', 'Title': ''}

        # Find and temporarily store any number of authors before the next match of year_pattern
        author_match = re.search(authorname_pattern, text)
        if author_match:
            authors = author_match.group(1)
            reference_data['Authors'].append(authors)
            text = text[author_match.end():]

            while True:
                # Check if the position of the next author is before the position of the next year_pattern
                next_author_match = re.search(authorname_pattern, text)
                next_year_match = re.search(year_pattern, text)
                if next_author_match and next_year_match and next_author_match.start() < next_year_match.start():
                    next_authors = next_author_match.group(1)
                    reference_data['Authors'].append(next_authors)
                    text = text[next_author_match.end():]
                else:
                    break
        else:
            break

        # Store the match of year_pattern
        year_match = re.search(year_pattern, text)
        if year_match:
            reference_data['Year'] = year_match.group()
            text = text[year_match.end():]

        # Find the first match to title_pattern and temporarily store it
        title_match = re.search(title_pattern, text)
        if title_match:
            reference_data['Title'] = title_match.group(1)
        if len(reference_data['Title']) >= 14 and re.match(r'^[A-Z][a-z]', reference_data['Title']):
            data_list.append(reference_data)

    return data_list


def extract_references_from_file(filepath: str) -> dict:
    # Loop through each reference within a file and extract the metadata
    sectiondict = split_text_to_sections(filepath)
    references = sectiondict["references"]
    ref_dict = extract_references(references)
    return ref_dict


def extract_references_from_folder(filepath: str) -> dict:
    # Loop through each txt file within a folder, get references within each file and extract the metadata
    ref_dict = {}
    filenames = list_filenames(filepath, "*.txt")
    for i in range(len(filenames)):
        item_name = filenames[i]
        item_path = f"{filepath}/{item_name}.txt"
        item_references = extract_references_from_file(item_path)
        ref_dict[i] = {"item": item_name, "reference": item_references}
    return ref_dict

if __name__ == "__main__":
    dict = extract_references_from_folder("C:/NLPvenv/NLP/input/pdf/docs/corrected/test")
    print_nested_dict(dict)
