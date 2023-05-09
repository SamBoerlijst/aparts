from setuptools import find_packages
from distutils.core import setup

with open("Readme.md", "r") as f:
    long_description = f.read()

setup(
    name="aparts",
    version="0.0.1",
    description="Tags pdf files using a tailored list of keywords and stores the results in a bib file and markdown summaries to be used in reference managers and personal knowledge management systems. ",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SamBoerlijst/aparts",
    author="Sam Boerlijst",
    author_email="samboerlijst@yahoo.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    install_requires=["Levenshtein >= 0.20.9","pandas >= 1.5.2", "anyascii >= 0.3.1", "pybtex >= 0.24.0", "PyPDF2 >= 3.0.0", "gensim >= 4.3.0", "numpy >= 1.24.0", 
                      "cleantext >= 1.1.4", "fuzzywuzzy >= 0.18.0", "keybert >= 0.7.0", "nlp_rake >= 0.0.2", "nltk >= 3.8", "pke >= 2.0.0", "six >= 1.16.0", "spacy>= 3.4.4", "yake>= 0.4.8", "scholarly>= 1.7.11", "scidownl>= 1.0.2"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.9.0",
)