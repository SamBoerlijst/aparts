Workflow
========

The following describes a general step-by-step workflow to generate and use keywords, and subsequently tag a corpus. All snippets are presented as python code. To run the code, either store it in a .py file or run from the terminal by running:

.. code-block:: bash

  python

In doing so the prompt before the command line should have changed from e.g. 'C:\\user\\current_user>' to '>>>', indicating that python code may now be run directly. To exit this mode press Ctrl+z followed by ENTER. 



Generate Keywords
-----------------
To generate a list of keywords, run the following command indicating the locations of the sourcefiles:

- Input folder: the absolute path which contains the records csv and bib file, and is also the location to store the keyword list in. In this case this is the folder: C:/aparts/input

- records: the name of the records csv file without the extendion. In this case this is the filename: records

- bibfile: the name of the exported reference list, or bib file without the extendion. In this case this is the filename: Library

- output_name: The filename that the output should be given. In this case the generated keywords will be stored in the file: keylist

- libtex_csv: Filename for the outputfile containing the contents of the bib file


.. note::

  If this is the first time generating a keylist, be sure to read "Function overview/Construct keyword list" and install the mentioned dependencies.


.. code-block:: python

  from aparts import generate_keylist

  generate_keylist(input_folder = "C:/aparts/input", records = "records", bibfile = "Library", output_name = "keylist", libtex_csv = "corpus_metadata")



Validate the keyword list
-------------------------
Before actually tagging the files, it may be wise to have a look at the generated keywords. Namely, some chemical names, or other jargon including symbols may have broken up into multiple parts. Additionally, some non-informative words may have accidentally be selected. At this stage this can be simply remedied by deleting the irrelevant keywords.



Collect pdf files
-----------------
The entries will be scanned for keywords by use of their corresponding pdf file. Thus, if any PDF files are missing, they will not be indexed. Optionally, you may add missing files by uding sci-hub using the following code. For this you need the article title, and the folderpath where the pdf should be stored.

.. code-block:: python

  from aparts.src.download_pdf import scihub_download_pdf

  scihub_download_pdf(paper=get_article(title="Biting the hand that feeds: Anthropogenic drivers interactively make mosquitoes thrive"), output_folder = "C:/aparts/input/pdf")



Tag articles
------------
To tag the articles, run the following command. Tagging can be done by tag presence in the source text or by weighted occurrence. The latter takes into account in which section a tag is found (e.g. occurrence in the abstract is higher rated than occurrence in the introduction) to limit false positives, and may be tweaked using the treshold. If summaries are set to true, summaries per article, author and journal are generated as additional output. Optionally this may be done lateron by use of the create_summaries function.

- source_folder: the reference manager path containing all pdf files. In this case: C:/.../Zotero/storage

- bibfile: path to the .bib file to which to add the tags. IN this case: C:/.../input/Library.bib

- alternate_lists: String defining the alternative lists to use. Options include: ‘all’, ‘statistics’, ‘countries’, ‘genomics’, ‘phylogenies’, ‘ecology’, ‘culicid_genera’ or any combinations thereof e.g. “statistics and countries”.

.. code-block:: python

  from aparts.src.APT import automated_pdf_tagging

  automated_pdf_tagging(source_folder="C:/.../Zotero/storage", bibfile="C:/.../input/Library.bib", alternate_lists="all", weighted = True, treshold = 5, summaries = True)



Select articles by dissimilarity
--------------------------------
To select a section of articles by tag dissimilarity, use the following command. Simply provide the path to the csv containing all (tagged) records and the amount of articles to be selected, and a list of titles will be returned. 

.. code-block:: python

  from aparts.src.subsampling import subsample_from_csv

  subsample_from_csv(CSV_path="C:/.../output/csv/total.csv", n=30)
