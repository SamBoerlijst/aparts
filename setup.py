from setuptools import find_packages, setup

with open("Readme.md", "r") as f:
    long_description = f.read()

setup(
    name="aparts",
    version="0.0.15",
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
    install_requires=["alabaster==0.7.13", "anyascii==0.3.2", "anyio==3.6.2", "arrow==1.2.3", "async-generator==1.10", "attrs==23.1.0", "Babel==2.12.1", "beautifulsoup4==4.12.2", "bibtexparser==1.4.0", "blis==0.7.8", "catalogue==2.0.8", "certifi==2023.5.7", "cffi==1.15.1", "charset-normalizer==3.1.0", "cleantext==1.1.4", "click==8.1.3", "colorama==0.4.6", "confection==0.0.4", "cymem==2.0.7", "Deprecated>=1.2.13", "docutils>=0.18.1", "exceptiongroup>=1.1.1", "fake-useragent>=1.1.3", "filelock>=3.12.0", "free-proxy>=1.1.1", "fsspec>=2023.5.0", "future>=0.18.3", "fuzzywuzzy>=0.18.0", "gensim>=4.3.1", "greenlet>=2.0.2", "h11>=0.14.0", "httpcore>=0.17.0", "httpx>=0.24.0", "huggingface-hub>=0.14.1", "idna>=3.4", "imagesize>=1.4.1", "importlib-metadata>=6.6.0", "importlib-resources>=5.12.0", "jellyfish>=0.11.2", "Jinja2>=3.1.2", "joblib>=1.2.0", "keybert>=0.7.0", "langcodes>=3.3.0", "latexcodec>=2.0.1", "Levenshtein>=0.21.0", "loguru>=0.7.0", "lxml>=4.9.2", "markdown-it-py>=2.2.0", "MarkupSafe>=2.1.2", "mdurl>=0.1.2", "mpmath>=1.3.0", "murmurhash>=1.0.9", "networkx>=3.1", "nlp-rake>=0.0.2", "nltk>=3.8.1", "numpy>=1.24.3", "outcome>=1.2.0", "packaging>=23.1", "pandas>=2.0.1", "pathy>=0.10.1", "Pillow>=9.5.0", "preshed>=3.0.8", "pybtex>=0.24.0", "pycparser>=2.21", "pydantic>=1.10.7", "Pygments>=2.15.1",
                      "pyparsing>=3.0.9", "PyPDF2>=3.0.1", "PySocks>=1.7.1", "python-dateutil>=2.8.2", "python-dotenv>=1.0.0", "pytz>=2023.3", "PyYAML>=6.0", "rapidfuzz>=3.0.0", "regex>=2023.5.5", "requests>=2.30.0", "rich>=13.3.5", "scholarly>=1.7.11", "scidownl>=1.0.2", "scikit-learn>=1.2.2", "scipy>=1.10.1", "segtok>=1.5.11", "selenium>=4.9.1", "sentence-transformers>=2.2.2", "sentencepiece>=0.1.99", "six>=1.16.0", "smart-open>=6.3.0", "sniffio>=1.3.0", "snowballstemmer>=2.2.0", "sortedcontainers>=2.4.0", "soupsieve>=2.4.1", "spacy>=3.5.2", "spacy-legacy>=3.0.12", "spacy-loggers>=1.0.4", "Sphinx>=6.2.1", "sphinx-rtd-theme>=1.2.0", "sphinxcontrib-applehelp>=1.0.4", "sphinxcontrib-devhelp>=1.0.2", "sphinxcontrib-htmlhelp>=2.0.1", "sphinxcontrib-jquery>=4.1", "sphinxcontrib-jsmath>=1.0.1", "sphinxcontrib-qthelp>=1.0.3", "sphinxcontrib-serializinghtml>=1.1.5", "SQLAlchemy>=2.0.12", "srsly>=2.4.6", "sympy>=1.11.1", "tablib>=3.4.0", "tabulate>=0.9.0", "thinc>=8.1.10", "threadpoolctl>=3.1.0", "tokenizers>=0.13.3", "torch>=2.0.1", "torchvision>=0.15.2", "tqdm>=4.65.0", "transformers>=4.28.1", "trio>=0.22.0", "trio-websocket>=0.10.2", "typer>=0.7.0", "typing_extensions>=4.5.0", "tzdata>=2023.3", "Unidecode>=1.3.6", "urllib3>=2.0.2", "wasabi>=1.1.1", "wget>=3.2", "win32-setctime>=1.1.0", "wrapt>=1.15.0", "wsproto>=1.2.0", "yake>=0.4.8", "zipp>=3.15.0"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.9.0",
)
