import aparts

#generate folder structure for output
aparts.generate_folder_structure()

#generate keylist using a csv containing article records, the "Author Keywords" column therein and .bib file. 
aparts.generate_keylist(records = "input/records.csv", bibfile = "input/library.bib", author_given_keywords="Author Keywords")

## It is advised to review the keylist document for artifacts before proceeding to the next step.

#Tag articles using weighted tagging using a pdf folder and bib file.
aparts.automated_pdf_tagging(source_folder="C:/Users/sboer/Zotero/storage", bibfile="input/library.bib", alternate_lists="all", weighted = True, treshold = 5, summaries = True)
