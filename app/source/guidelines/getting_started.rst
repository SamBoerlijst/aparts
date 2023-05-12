Getting started
+++++++++++++++

Install the library
----------------------

.. code-block:: bash
    
    pip install aparts


Confirm installation of direct dependencies
-------------------------------------------
This package relies on the textrank and topicrank models from https://github.com/boudinfl/pke/tree/master. As pypi does not allow direct dependencies, this package has to be installed manually:

.. code-block:: bash
    
    pip install git+https://github.com/boudinfl/pke.git


Test the package
----------------

To test whether the library works you may use the function, which will prepare the input and output directory relative to the curren folder.

.. code-block:: python
    
    from aparts import generate_folder_structure

    generate_folder_structure()