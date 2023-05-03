# APART: Academic PDF Automated Reference Tagging. 

Automated workflow to index academic articles with a personalized set of keywords. Designed to be used in combination with reference editors, and markdown-based personal knowledge management systems like obsidian and notion.

#### Functionality
- build a set of keywords by a query representative of the field of research
- download missing pdf files using sci-hub
- tag all pdf files within a folder irrespective of folder structure using 7 NLP algorithms 
- return the keywords to a .bib file for use in reference managers
- returns .md summaries per author, article and journal for use in markdown knowledge bases.

#### Use cases
- Optimizing queries for scientific reviews
- Article selection for scientific review
- Indexing bilbiography
- Node-network analysis

#### input
- a query representative of the field of research
- .bib file containing the references to be indexed
- folderpath containing article pdf files

#### output
- .bib file for indexed citations
- .csv file containing the metadata for each citation
- .md files per article, author and journal giving a dynnamic and interlinked overview of metadata and associated tags and (co-)authors



## building a keyword list
Collect keywords from (web of science or google scholar) csv list of titles and abstracts using 7 common NLP algorithms.
bigram, keybert, RAKE, textrank, topicrank, TF-IDF and YAKE
2-4 of the algorithms
exclude blacklist
add from .bib
optional keyword lists for statistical tests, countries, genomics, phylogenies and ecology

The list may be combined those with author given tags and tags present in bib file and export as csv


## download (missing) pdf files
by title and author name


## tag pdf files
collect pdf files
conversion to txt
tagging weighted by section


## .bib output
Adds tags to supplied .bib file


## Markdown summaries
text based summaries using javascript code blocks so that the database stays dynamically updated

#### Article summary
Metadata: tags, metadata present in apa6 formatted citation, abstract
interlinked to relevant journal and authors

#### Author summary
co-authors by frequency
tags by frequency
associated papers
interlinked to relevant papers and journals

#### Journal summary
authors by frequency
tags by frequency
associated papers
interlinked to relevant papers and authors


## csv file
metadata acquired from the .bib file and indexed tags per article
