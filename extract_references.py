from weighted_tagging import split_text_to_sections, print_nested_dict
import re

def extract_references(text: str) -> dict:
    # Create a dictionary to store the metadata for each reference
    metadata = {}
    if text.lower().startswith("references"):
        reference_string = text[len('references'):].lstrip()
    def split_references(text):
        # Add a space after the doi link to separate it from the next reference
        text = re.sub(r'(https://doi.org/\S|10+)', r'\1 ', text)
        # Split the text into a list of references
        references = re.split(r'(?<!https://doi\.)\d+\.', text)
        # Remove leading and trailing whitespace from each reference
        references = [ref.strip() for ref in references if ref.strip()]
        return references

    references = split_references(reference_string)
    print(references)
    # Loop through each reference and extract the metadata
    for i in range(len(references)):
        refnum = i+1
        split_by_period = references[i].split('.')
        split_by_bracket = references[i].split('(')
        if len(split_by_bracket) > 1:
            length = len(split_by_period)
            # Extract the reference number and remove the period
            title = references[i].split('.')[length-2] 
            author_section = "".join(split_by_period[0:length-2])
            author = author_section.split('(')[0]
            year_section = split_by_bracket[1].replace(")", "")
            year = year_section.split('.')[0]
            metadata[refnum] = {'Authors': author, 'Year': year, 'Title': title}
    return metadata


def extract_references_from_file(filepath:str)->dict:
    sectiondict = split_text_to_sections(filepath)
    references = sectiondict["references"]
    ref_dict = extract_references(references)
    print_nested_dict(ref_dict)
    return ref_dict


if __name__ == "__main__":
    extract_references_from_file("input/pdf/docs/Alberdi et al. - 2018 - Scrutinizing key steps for reliable metabarcoding .txt")