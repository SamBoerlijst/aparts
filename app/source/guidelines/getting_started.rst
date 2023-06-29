Getting started
+++++++++++++++

Install the library
----------------------

.. code-block:: bash
    
    pip install git+https://github.com/SamBoerlijst/aparts.git


Test the package
----------------

To test whether the library works you may use the function, which will prepare the input and output directory relative to the curren folder.

.. code-block:: python
    
    from aparts import generate_folder_structure

    generate_folder_structure()
