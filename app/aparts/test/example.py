from src.construct_keylist import generate_folder_structure, generate_keylist
from src.APT import automated_pdf_tagging

#generate folder structure for output
generate_folder_structure()

#generate keylist using a csv containing article records, the "Author Keywords" column therein and .bib file. 
#generate_keylist(records = "input/records.csv", bibfile = "input/library.bib", author_given_keywords="Author Keywords")

## It is advised to review the keylist document for artifacts before proceeding to the next step.

#Tag articles using weighted tagging using a pdf folder and bib file.
#automated_pdf_tagging(source_folder="C:/Users/sboer/Zotero/storage", bibfile="input/library.bib", alternate_lists="all", weighted = True, treshold = 5, summaries = True)
