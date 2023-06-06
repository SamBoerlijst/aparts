Construct keyword list
======================

First time use
++++++++++++++
Download the required models 

spacy:
-------

.. code-block:: sh
   :caption: Terminal
    
   C:/../aparts> python -m spacy download en_core_web_sm

nltk
----
.. code-block:: sh
   :caption: Terminal
 
   C:/../aparts> python
   >>> import nltk
   >>> nltk.download('punkt')

Run the code
++++++++++++
Open up VScode, anaconda or whatever interface you will be using.
Open a terminal (within the program) and move to the directory you want to store the input and output in.

Setup the folder
----------------
.. code-block:: python
   :caption: python
    
    from aparts import generate_folder_structure 
    generate_folder_structure()

Move the records.csv and library.bib to the input folder

Generate the tags
-----------------
.. code-block:: python
   :caption: python
    
    from aparts import generate_keylist 
    generate_keylist(input_folder = "C:/aparts/input", records = "records", bibfile = "Library")

The process may take a while - up to an hour - depending on the amount of records used for keyword generation.
files are created by each algorithm at each step of the proces (first title extraction, then abstract), so that the process may be followed up more easily in case of any interruptions.