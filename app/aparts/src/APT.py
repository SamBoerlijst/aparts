import io
import os
import re
from fnmatch import fnmatch
from shutil import copy2

import Levenshtein
import pandas as pd
from anyascii import anyascii
from pybtex.database import parse_file
from PyPDF2 import PdfReader

from aparts.src.weighted_tagging import split_text_to_sections, count_keyword_occurrences, filter_values, weigh_keywords, denest_and_order_dict, print_nested_dict

""" aparts
- Academic Pdf - Automated Reference Tagging System: extract pdf from refmanager folder, convert to lowercase utf-8 txt, index keywords from keylist and store those in csv, bib and md format
- Author: Sam Boerlijst
- Date: 9/5/2023
"""

# optional taglists
altlist = ['statistics', 'countries', 'genomics', 'phylogenies', 'ecology', 'culicid_genera']
statistics = [
            "anova",
            "ancova",
            "anderson darling test",
            "anderson-darling test",
            "anosim",
            "binomial",
            "binomial distribution",
            "bootstrap",
            "categorical",
            "chi-test",
            "chi-squared",
            "chi-squared test",
            "cramer's coefficient",
            "dunn-test",
            "dunn's test",
            "exponential",
            "friedman",
            "friedman test",
            "friedman-test",
            "g-test",
            "glm",
            "glmm",
            "hypergeometric",
            "interaction",
            "interactive",
            "kolmogorov smirnov test",
            "kolmogorov-smirnov test",
            "kruskall test",
            "kruskall-wallis",
            "kruskall-wallis test",
            "linear model",
            "linear regression",
            "logistic regression",
            "mancova",
            "manova",
            "mann-whitney",
            "mean-squared error",
            "mixed-effects",
            "multifactorial",
            "multiple regression",
            "multi-way anova",
            "nested-design",
            "nested factor",
            "non-parametric",
            "nonparametric",
            "normal distribution",
            "normally distributed",
            "one-way anova",
            "parametric",
            "pearsons",
            "permanova",
            "pca",
            "phi coefficient",
            "poisson",
            "poisson distribution",
            "polynomial",
            "probability",
            "quantitative",
            "random forest",
            "repeated measures",
            "scheier-ray-hare test",
            "sign test",
            "simple regression",
            "simpr",
            "simpsons index",
            "spearman's rank",
            "t-test",
            "t test",
            "three-way anova",
            "two-way anova",
            "wilcoxon",
        ]
countries = [
            "Côte d'Ivoire",
            "Lao People's Democratic Republic",
            "Aland Islands",
            "Albania",
            "Algeria",
            "American Samoa",
            "Andorra",
            "Angola",
            "Anguilla",
            "Antarctica",
            "Antigua and Barbuda",
            "Argentina",
            "Armenia",
            "Aruba",
            "Australia",
            "Austria",
            "Azerbaijan",
            "Bahamas",
            "Bahrain",
            "Bangladesh",
            "Barbados",
            "Belarus",
            "Belgium",
            "Belize",
            "Benin",
            "Bermuda",
            "Bhutan",
            "Bolivia",
            "Boluvaria",
            "Bonaire",
            "Bosnia",
            "Botswana",
            "Bouvet Island",
            "Brazil",
            "British Indian Ocean Territory",
            "British Virgin Islands",
            "Brunei Darussalam",
            "Bulgaria",
            "Burkina Faso",
            "Burundi",
            "Cambodia",
            "Cameroon",
            "Canada",
            "Cape Verde",
            "Cayman Islands",
            "Central African Republic",
            "Chad",
            "Chile",
            "China",
            "Christmas Island",
            "Cocos (Keeling) Islands",
            "Cocos Islands",
            "Colombia",
            "Comoros",
            "Congo",
            "Cook Islands",
            "Costa Rica",
            "Croatia",
            "Cuba",
            "Curaçao",
            "Cyprus",
            "Czech Republic",
            "Denmark",
            "Djibouti",
            "Dominica",
            "Dominican Republic",
            "Ecuador",
            "Egypt",
            "El Salvador",
            "Equatorial Guinea",
            "Eritrea",
            "Estonia",
            "Ethiopia",
            "Falkland Islands",
            "Faroe Islands",
            "Fiji",
            "Finland",
            "France",
            "French Guiana",
            "French Polynesia",
            "French Southern Territories",
            "Gabon",
            "Gambia",
            "Georgia",
            "Germany",
            "Ghana",
            "Gibraltar",
            "Greece",
            "Greenland",
            "Grenada",
            "Guadeloupe",
            "Guam",
            "Guatemala",
            "Guernsey",
            "Guinea",
            "Guinea-Bissau",
            "Guyana",
            "Haiti",
            "Heard Island",
            "Herzegovina",
            "Holy See",
            "Honduras",
            "Hong Kong",
            "Hungary",
            "Iceland",
            "India",
            "Indonesia",
            "Iran",
            "Iraq",
            "Ireland",
            "Islamic Republic of Iran",
            "Isle of Man",
            "Israel",
            "Italy",
            "Jamaica",
            "Jan Mayen",
            "Japan",
            "Jersey",
            "Jordan",
            "Kazakhstan",
            "Kenya",
            "Kiribati",
            "Korea",
            "Kuwait",
            "Kyrgyzstan",
            "Latvia",
            "Lebanon",
            "Lesotho",
            "Liberia",
            "Libya",
            "Liechtenstein",
            "Lithuania",
            "Luxembourg",
            "Macao",
            "Macedonia",
            "Madagascar",
            "Malawi",
            "Malaysia",
            "Maldives",
            "Mali",
            "Malta",
            "Malvinas",
            "Marshall Islands",
            "Martinique",
            "Mauritania",
            "Mauritius",
            "Mayotte",
            "McDonald Islands",
            "Mexico",
            "Micronesia",
            "Moldova",
            "Monaco",
            "Mongolia",
            "Montenegro",
            "Montserrat",
            "Morocco",
            "Mozambique",
            "Myanmar",
            "Namibia",
            "Nauru",
            "Nepal",
            "Netherlands",
            "New Caledonia",
            "New Zealand",
            "Nicaragua",
            "Niger",
            "Nigeria",
            "Niue",
            "Norfolk Island",
            "Northern Mariana Islands",
            "Norway",
            "Oman",
            "Pakistan",
            "Palau",
            "Palestinian Territory",
            "Panama",
            "Papua New Guinea",
            "Paraguay",
            "Peru",
            "Philippines",
            "Pitcairn",
            "Plurinational State of Bolivia",
            "Poland",
            "Portugal",
            "Puerto Rico",
            "Qatar",
            "Republic of Bolivaria",
            "Republic of Korea",
            "Republic of Moldova",
            "Romania",
            "Russia",
            "Russian Federation",
            "Rwanda",
            "Réunion",
            "Saba",
            "Saint Barthélemy",
            "Saint Helena",
            "Saint Kitts",
            "Saint Lucia",
            "Saint Martin",
            "Saint Miquelon",
            "Saint Nevis",
            "Saint Pierre",
            "Saint Vincent",
            "Samoa",
            "San Marino",
            "Sao Tome and Principe",
            "Saudi Arabia",
            "Senegal",
            "Serbia",
            "Seychelles",
            "Sierra Leone",
            "Singapore",
            "Sint Eustatius",
            "Sint Maarten",
            "Slovakia",
            "Slovenia",
            "Solomon Islands",
            "Somalia",
            "South Africa",
            "South Georgia",
            "South Sandwich Islands",
            "South Sudan",
            "Spain",
            "Sri Lanka",
            "Sudan",
            "Suriname",
            "Svalbard",
            "Swaziland",
            "Sweden",
            "Switzerland",
            "Syria",
            "Syrian Arab Republic",
            "Taiwan",
            "Tajikistan",
            "Tanzania",
            "Thailand",
            "The Democratic Republic of congo",
            "Timor-Leste",
            "Tobago",
            "Togo",
            "Tokelau",
            "Tonga",
            "Trinidad",
            "Tunisia",
            "Turkey",
            "Turkmenistan",
            "Turks and Caicos Islands",
            "Tuvalu",
            "U.S. Virgin Islands",
            "Uganda",
            "Ukraine",
            "United Arab Emirates",
            "United Kingdom",
            "United States Minor Outlying Islands",
            "United States",
            "Uruguay",
            "Uzbekistan",
            "Vanuatu",
            "Vatican City",
            "Venezuela",
            "Viet Nam",
            "Wallis and Futuna",
            "Yemen",
            "Zambia",
            "Zimbabwe",
            "the Grenadines",
            "Ascension and Tristan da Cunha",
            "Province of China",
            "Republic of",
            "United Republic of Afghanistan",
            "Afghanistan",
        ]
genomics = [
            "accumulation curve",
            "annealing",
            "adapters",
            "alignment",
            "amplification",
            "amplicon",
            "ancient dna",
            "ancient edna",
            "barcode",
            "bioindicator",
            "blast",
            "bold",
            "buffer",
            "cdna",
            "chimera",
            "chimera detection",
            "clustering",
            "co1",
            "degradation",
            "dna",
            "dsdna",
            "ecoprimer",
            "edna",
            "genbank",
            "genome",
            "genomics",
            "hiseq",
            "inhibitor",
            "illumina",
            "iontorrent",
            "locus",
            "loci",
            "matk",
            "metabarcoding",
            "metafast",
            "methylation",
            "minion",
            "miseq",
            "mitochondrial",
            "mothur",
            "motu",
            "multiplex",
            "nanopore",
            "ngs",
            "nuclear",
            "obitools",
            "otu",
            "pcr",
            "primer",
            "primer3",
            "protein",
            "proteomics",
            "pyrosequencing",
            "qiime",
            "qpcr",
            "rarefaction curve",
            "rdna",
            "rna",
            "rrna",
            "sanger",
            "sequencing",
            "shotgun-pcr",
            "ssdna",
            "sumaclust",
            "tag",
        ]
phylogenies = [
            "biotype",
            "clade",
            "identification",
            "identification key",
            "macrofauna",
            "morphological",
            "morphology",
            "species",
            "species complex",
            "taxa",
            "taxon",
            "taxonomy",
        ]
ecology = [
            "abundance",
            "adult",
            "alpha-diversity",
            "beta-diversity",
            "biodiversity",
            "development rate",
            "diet analysis",
            "ecology",
            "faecal",
            "feeding",
            "female",
            "field study",
            "field-like",
            "host",
            "host-range",
            "juvenile",
            "lab study",
            "longevity",
            "macrocosm",
            "malaise",
            "male",
            "mesocosm",
            "microcosm",
            "monitoring",
            "mortality",
            "pupa",
            "sex-ratio",
            "sex ratio",
            "shannon",
            "species richness",
            "subadult",
            "survival",
        ]
culicid_genera = [
            "Abraedes",
            "Acartomyia",
            "Aedeomyia",
            "Aedes",
            "Aedimorphus",
            "Alanstonea",
            "Albuginosus",
            "Anopheles",
            "Armigeres",
            "Ayurakitia",
            "Aztecaedes",
            "Belkinius",
            "Bifidistylus",
            "Bironella",
            "Borichinda",
            "Bothaella",
            "Bruceharrisonius",
            "Cancraedes",
            "Catageiomyia",
            "Catatassomyia",
            "Chagasia",
            "Christophersiomyia",
            "Collessius",
            "Coquillettidia",
            "Cornetius",
            "Culex",
            "Culiseta",
            "Dahliana",
            "Danielsia",
            "Deinocerites",
            "Dendroskusea",
            "Diceromyia",
            "Dobrotworskyius",
            "Downsiomyia",
            "Edwardsaedes",
            "Elpeytonius",
            "Eretmapodites",
            "Ficalbia",
            "Finlaya",
            "Fredwardsius",
            "Galindomyia",
            "Georgecraigius",
            "Geoskusea",
            "Gilesius",
            "Gymnometopa",
            "Haemagogus",
            "Halaedes",
            "Heizmannia",
            "Himalaius",
            "Hodgesia",
            "Hopkinsius",
            "Howardina",
            "Huaedes",
            "Hulecoeteomyia",
            "Indusius",
            "Isoaedes",
            "Isostomyia",
            "Jarnellius",
            "Jihlienius",
            "Johnbelkinia",
            "Kenknightia",
            "Kimia",
            "Kompia",
            "Leptosomatomyia",
            "Levua",
            "Lewnielsenius",
            "Limatus",
            "Lorrainea",
            "Luius",
            "Lutzia",
            "Macleaya",
            "Malaya",
            "Mansonia",
            "Maorigoeldia",
            "Mimomyia",
            "Molpemyia",
            "Mucidus",
            "Neomelaniconion",
            "Nyctomyia",
            "Ochlerotatus",
            "Onirion",
            "Opifex",
            "Orthopodomyia",
            "Paraedes",
            "Patmarksia",
            "Paulianius",
            "Petermattinglyius",
            "Phagomyia",
            "Polyleptiomyia",
            "Pseudarmigeres",
            "Psorophora",
            "Rampamyia",
            "Rhinoskusea",
            "Runchomyia",
            "Sabethes",
            "Sallumia",
            "Scutomyia",
            "Shannoniana",
            "Skusea",
            "Stegomyia",
            "Tanakaius",
            "Tewarius",
            "Topomyia",
            "Toxorhynchites",
            "Trichoprosopon",
            "Tripteroides",
            "Udaya",
            "Uranotaenia",
            "Vansomerenis",
            "Verrallina",
            "Wyeomyia",
            "Zavortinkius",
            "Zeugnomyia",
        ]


### subroutines

## folder manipulation
def merge_sourcefolder_to_distfolder(type:str, files:list, src_dir:str, root_dst_dir:str, overwrite:bool) -> None:
    """
    Collect files with given extension from folder including subfolders and copy to destination folder.

    Parameters:
    -----------
    type (str): Extension of the filetype to copy.

    files (list):  Collection of files to search within.

    src_dir (str): Path to the folder where the original is located.

    root_dst_dir (str): Path to the folder in which the copy should be stored.

    overwrite (bool): Overwrite any existing files in the destination folder.

    Returns:
    --------
    None
    """
    for item in files:
        if fnmatch(item, type):
            src_file = os.path.join(src_dir, item)
            dst_file = os.path.join(root_dst_dir, item)
            if os.path.exists(dst_file) & overwrite == 1:
                os.remove(dst_file)
            if not os.path.exists(dst_file):
                copy2(src_file, root_dst_dir)
    return

def collect_PDF_files(root_src_dir: str, root_dst_dir: str) -> None:
    """
    Collect all PDF files from folder including subfolders and copy to destination folder.

    Parameters:
    -----------
    root_src_dir (str): Path to the folder where the original files are located.
    
    root_dst_dir (str): Path to the folder in which the copies should be stored.

    Returns:
    --------
    None
    """
    print("collecting pdf files")
    type = "*.pdf"
    for src_dir, dirs, files in os.walk(root_src_dir):
        name, ext = os.path.splitext(root_src_dir)
        end = name.split("/").pop()
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        # create destination folder if not yet present
        if not root_dst_dir:
            os.makedirs(dst_dir)
        merge_sourcefolder_to_distfolder(type, files, src_dir, root_dst_dir, overwrite=1)
    return

def guarantee_folder_exists(folder:str) -> None:
    """Create folder if not yet present."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    return

def guarantee_md_output_folders_exist(folder:str)->None:
    guarantee_folder_exists(f"{folder}/papers")
    guarantee_folder_exists(folder)
    guarantee_folder_exists(f"{folder}/authors")
    guarantee_folder_exists(f"{folder}/journals")

## file manipulation
def guarantee_csv_exists(filename:str, df:pd.DataFrame) -> None:
    """Create csv if not yet present populated by the given dataframe."""
    if not os.path.isfile(filename):
        df.to_csv(filename, index = False)
    return

## indexing
def list_filenames(folder:str, type:str) -> list:
    """
    List all filenames of the supplied type within a given folder.

    Parameters:
    -----------
    folder (str): Path to the folder where the files are located. 
    
    type (str): Extension of the files to filter e.g. asterisk followed by .txt.

    Returns:
    --------
    itemlist (list): List of files of given extension.
    """
    itemlist = []
    for path, folders, files in os.walk(folder):
        for file in files:
            if fnmatch(file, type):
                name, ext = os.path.splitext(file)
                end = name.split("/").pop()
                itemlist.append(end)
    return itemlist

## content manipulation
def reset_eof_of_pdf_return_stream(pdf_stream_in: list) -> list:
    """
    Fix EOF errors in files by finding EOF position.

    Parameters:
    -----------
    pdf_stream_in (list): List of bytes for which the EOF position should be determined.

    Returns:
    --------
    pdf_stream_in[:actual_line] (list): collection of EOF positions.
    """
    for i, x in enumerate(pdf_stream_in[::-1]):
        if b"%%EOF" in x:
            actual_line = len(pdf_stream_in) - i
            pdf_stream_in[:actual_line]
            # print(f'EOF found at line position {-i} = actual {actual_line}, with value {x}')
            break
    # return the list up to that point
    return 

def convert_unicode_from_string(string:str)->str:
    """Convert an unicode string to utf-8."""
    string = string.encode("unicode-escape").decode()
    string = string.replace("\\n", " ")
    string = repr(string)
    string = anyascii(string)
    string = string.strip("'")
    return string

def remove_trailing_backslashes(string:str)->str:
    """Remove trailing backslashes from a string caused by decoding errors."""
    string = (
        string.replace("\\\\", "\\")
        .replace("\\\\", "\\")
    )
    return string

def pdf2txtfolder(PDFfolder: str, TXTfolder: str) -> None:
    """
    Convert all PDF files in a given folder to txt files.

    Parameters:
    -----------
    PDFfolder (str): Path to the folder where the original files are located.
    
    TXTfolder (str): Path to the folder in which the extracted text should be stored.

    Returns:
    --------
    None
    """
    print("converting pdf files to txt")
    guarantee_folder_exists(folder=TXTfolder)
    input_file_list = list_filenames(folder=PDFfolder, type="*.pdf")
    output_file_list = list_filenames(folder=TXTfolder, type="*.txt")
    # add txt files for pdfs that haven't been processed while correcting eof
    for file_ in input_file_list:
        if str(file_) not in output_file_list:
            if os.path.isfile(f"{TXTfolder}/{file_}.txt") == False:
                with open(f"{PDFfolder}/{file_}.pdf", "rb") as p:
                    contents = p.readlines()
                    contents_eof_corrected = reset_eof_of_pdf_return_stream(contents)
                with open(f"{PDFfolder}/{file_}.pdf", "wb") as d:
                    d.writelines(contents_eof_corrected)

                # create txt with the same name in the output folder
                with open(f"{TXTfolder}/{file_}.txt", "w+") as t:
                    with open(f"{PDFfolder}/{file_}.pdf", "rb") as f:
                        read_pdf = PdfReader(f)
                        number_of_pages = len(read_pdf.pages)
                        # write page content to files
                        for i in range(number_of_pages):
                            page = read_pdf.pages[i]
                            page_content = page.extract_text()
                            page_content_utf8 = convert_unicode_from_string(page_content)
                            page_content_utf8 = remove_trailing_backslashes(page_content_utf8)
                            # save the extracted data from pdf to a txt file with one line per page
                            t.write(str(page_content_utf8.encode()))
                            t.write("\n")
                            # print(i+1,"/",number_of_pages)
                print("converted ", str(file_)[0:30], "... to txt")
    return

def unicodecleanup_folder(TXTfolder: str, TXTCorfolder: str):
    """
    Corrects unicode decode errors for each text file in the given folder.

    Parameters:
    -----------
    TXTfolder (str): Path to the folder where the input files are stored.
    
    TXTCorfolder (str):Path to the folder where the output files should be stored.

    Returns:
    --------
    None
    """
    print("correcting unicode decode errors in txt files")
    guarantee_folder_exists(TXTCorfolder)
    itemlist = list_filenames(TXTfolder, "*.txt")
    itemlist1 = list_filenames(TXTCorfolder, "*.txt")
    # add txt files for txt that havent been processed
    for item in itemlist:
        if str(item) not in itemlist1:
            with open(f"{TXTfolder}/{item}.txt", "r") as t:
                file1 = open(f"{TXTCorfolder}/{item}.txt", "w+")
                for line in t:
                    string_utf8 = convert_unicode_from_string(line)
                    string_utf8 = remove_trailing_backslashes(string_utf8)
                    file1.write(str(string_utf8.encode()))
                    file1.write("\n")
                print(f"corrected encoding for {item}")
    return

def sort_joined_list(input_list:list) -> list:
    """
    Returns a sorted list with only the unique values of the input list.

    Parameters:
    input_list: a list of values.

    Returns:
    A sorted list of unique values in input_list.
    """
    return list(set(input_list))

def filter_uniques_from_list(List:list)->list:
    """Filter uniques in list of strings."""
    List = sorted(list(set(List)))
    return List

def set_additional_keywords(alternate_lists:str) -> list:
    """
    Construct list of additional keywords by matching contents of string to all - or a selection of - optional taglists.

    Parameters:
    -----------
    alternate_lists (str): String containing the names of the additional lists of keywoords to include.
    
    Returns:
    --------
    taglist (list): List of chosen optional keywords.
    """
    #empty list for storing optional taglists
    taglist = []
    if(alternate_lists.__contains__("all")):
        for item in altlist:
            taglist = (taglist + eval(item))
    else:
        for item in altlist:
            if(alternate_lists.__contains__(str(item))):
                taglist = (taglist + eval(item))
    return taglist

def find_keywords(keylist: list, text: str) -> list:
    """
    Scan a string for keywords from the given list.

    Parameters:
    -----------
    keylist (list): List of strings containing the keywords of interest.
    text (str): String of text to search for keywords.

    Returns:
    --------
    text_keylist (list): List of strings containing the found keywords.
    """
    text_keylist = []
    for keyword in keylist:
        patterns = [
            f"{keyword} ",
            f" {keyword}",
            f"{keyword}-",
            f"-{keyword}",
            f"{keyword}.",
            f".{keyword}"
        ]
        if any(text.find(pattern) != -1 for pattern in patterns):
            text_keylist.append(keyword)
    return text_keylist

def get_filename(file:io.BufferedReader)->str:
    """Extract filename from path."""
    name, ext = os.path.splitext(file.name)
    end = name.split("/").pop()
    return end

def fix_broken_words(text:str)->str:
    """Fix words broken up by typesetting."""
    return text.replace("- ", "")

def remove_special_characters(text: str) -> str:
    """Delete illegal characters for filenames from a string."""
    illegal_chars = r'[\\\/_\{\}<>?%*:\|\"\'\s]'  # Regular expression pattern
    cleaned_text = re.sub(illegal_chars, '', text)
    return cleaned_text

def author_to_firstname_lastname(row: pd.Series) -> list:
    """
    Changes all author names within a pandas row from "lastname, firstname" to "firstname lastname"

    Parameters:
    -----------
    row (pd.Series): Row from pandas dataset containing authornames.

    Return:
    -----------
    authorlist (list): List of all authornames as "firstname lastname".
    """
    authorlist = []
    author = row["author"]
    if pd.isna(author):
        authorlist = ["nan"]
    else:
        authors = str(author)
        if "and " in authors:
            names = authors.split("and ")
            for name in names:
                name = name.replace("{","").replace("}","").replace(".","")
                if ", " in name:
                    lastname, firstname = name.split(", ")
                    if firstname.endswith(" "):
                        authorlist.append("[[" + firstname + lastname + "]]")
                    else:
                        authorlist.append("[[" + firstname + " " + lastname + "]]")
                else:
                    authorlist.append("[[" + name + "]]")
        elif ", " in authors:
            lastname, firstname = authors.split(", ")
            if firstname.endswith(" "):
                authorlist.append("[[" + firstname + lastname + "]]")
            else:
                authorlist.append("[[" + firstname + " " + lastname + "]]")
        else:
            authorlist = [authors]
    return authorlist

def correct_authornames(filepath: str)->None:
    """
    Scans a dataset of articles for authornames, changes them from lastname, firstname to firstname lastname format, and saves the names to an "author_corrected" column.

    Parameters:
    -----------
    filepath (str): Filepath to dataset containing authornames to be changed.
    """
    path = str(filepath)
    dataframe = pd.read_csv(path)
    data = {}
    for index, row in dataframe.iterrows():
        if not pd.isna(row["title"]):
            name = []
            authorlist_corrected = []
            authorlist = author_to_firstname_lastname(row)
            for ele in authorlist:
                if str(ele) != "empty":
                    name = (
                        str(ele)
                        .replace("{", "")
                        .replace("}", "")
                        .replace(".", "")
                    )
                    authorlist_corrected.append(name)
            authorlist_corrected = ", ".join(authorlist_corrected)
            pd_row = pd.DataFrame({"author_corrected": [authorlist_corrected]}, index=[0])
            data[index] = pd_row
    # repopulate author_corrected column with name surname
    for index, pd_row in data.items():
        dataframe.at[index, "author_corrected"] = pd_row["author_corrected"][0]
    dataframe.to_csv(filepath, index=False)
    return

def collapse_authors(names:list)->list:
    """
    Collapses a list of names, where a person may occur multiple times with different variations of initials and names,
    into a single representation for each person while keeping the longest name.

    Parameters:
    -----------
    names (list): A list of strings representing names.

    Returns:
    -----------
    list: A list of collapsed names, where each name is the longest representation for a person.
    """
    collapsed_names = {}
    
    # Iterate over each name in the list
    for name in names:
        name = name.replace("[","").replace("]","")
        name_parts = name.split()
        initials = [part[0] for part in name_parts[:-1]]  # Get initials or first letters
        last_name = name_parts[-1]  # Get last name
        
        # Create a key using initials and last name
        key = ''.join(initials) + ' ' + last_name
        
        # Check if the key already exists in the collapsed_names dictionary
        if key in collapsed_names:
            # Compare the current name with the stored name using Levenshtein distance
            current_distance = Levenshtein.distance(name, collapsed_names[key])
            stored_distance = Levenshtein.distance(collapsed_names[key], name)
            
            # If the current name is longer or has a smaller distance, update the dictionary value
            if len(name) > len(collapsed_names[key]) or current_distance < stored_distance:
                collapsed_names[key] = name
        else:
            # If the key doesn't exist, add the name to the dictionary
            collapsed_names[key] = name
    
    # Return the values of the collapsed_names dictionary
    return list(collapsed_names.values())

def populate_with_template(title:str, template_text: str) -> None:
    """Create file and fill contents with template text."""
    with open(title, "wb+") as file:
        # populate with template
        for line in template_text:
            file.write(str(line).encode("utf-8"))
            file.close
    return

def check_record_type(row:pd.Series) -> str:
    """Determine wheter a record refers to a book or article by isbn"""
    if not "10." in str(row["doi"]):
        if row["note"] != "nan" or row["isbn"] != "nan":
            value = "book"
    else:
            value = "article"
    return value

def populate_placeholders(placeholderlist:list, valuelist:list, record:pd.Series, data:str):
    """Replace the placeholders in a file with recorddata for a given file"""
    for placeholder, value_key in zip(placeholderlist, valuelist):
        placeholder = str(placeholder)
        value = str(record[value_key])  
        if placeholder == "VALUE:journal":
            if value != "nan":
                value = value.replace("{", "").replace("}", "").replace(".", "").replace("&", "and").replace("\\\\", "").replace("\\", "").replace(":", "").replace(";", "")
            data = data.replace(placeholder, value)
        elif placeholder == "VALUE:keywords":
            if value != "nan":
                value = value.replace(" & ", "_&_").replace(": ", "/").replace(" ", "/").replace(",/", ", ")
            data = data.replace(placeholder, value)
        else:
            data = data.replace(placeholder, value)
    return data

## subroutines
def preprocess_text(text:str)->str:
    """Fix common problems generated by typesetting."""
    text = repr(text)
    text = remove_trailing_backslashes(text)
    text = convert_unicode_from_string(text)
    text = text.lower()
    text = fix_broken_words(text)
    return text

def construct_keylist(keylist_path:str, alternate_lists:str)->list:
    """
    Merge list of keywords from csv with chosen optional keywords.
    
    Parameters:
    -----------
    keylist_path (str): Path to the csv containg the keywords to be used. Generated by construct_keylist().
    
    alternate_lists (str): String defining the alternative lists to use. Options include: 'all', 'statistics', 'countries', 'genomics', 'phylogenies', 'ecology', 'culicid_genera' or any combinations thereof e.g. "statistics and countries".

    Return:
    -----------
    keylist (list): List of the combination of unique keywords.
    """
    keylist_csv = pd.read_csv(keylist_path)["ID"].tolist()
    additional_keywords = set_additional_keywords(alternate_lists)
    keylist = (keylist_csv + additional_keywords)
    keylist = sort_joined_list(keylist)
    return keylist

def calculate_tag_counts(taglists, separator:str = ", ") -> pd.DataFrame:
    """
    Calculates the counts of tags across the entire corpus.

    Parameters:
    -----------
    taglists (list, string or pd.Series): List of strings or list of lists containing tags from each document

    separator (str): In case a Series is used as input, this indicates the separator between tags within an item 
    
    Returns:
    -----------
    tagcounts (pd.DataFrame): DataFrame containing the calculated counts per keyword over the entire corpus, ordered by occurrence
    """

    tag_counts = {}
    def count_items(string_list: list):
        for item in string_list:
            if item in tag_counts:
                tag_counts[item] += 1
            else:
                tag_counts[item] = 1
    
    if isinstance(taglists, list) and all(isinstance(sublist, list) for sublist in taglists):
        for item in taglists:
            count_items(item)
    elif isinstance(taglists, list):
        count_items(taglists)
    elif isinstance(taglists, pd.Series):
        split_list = [str(item).split(separator) for item in taglists]
        taglist = [item for sublist in split_list for item in sublist]
        count_items(taglist)
    else:
        print("please use a supported type: list, str or pd.Series")

    tag_counts = pd.DataFrame({"keyword": list(tag_counts.keys()), "count": list(tag_counts.values())})

    tag_counts = tag_counts.sort_values(by="count", ascending=False)

    return tag_counts


## scan the txt files for strings contained in the keylist
def tag_folder(TXTCorfolder: str = "input/pdf/docs/corrected", keylist_path: str = "input/keylist.csv", outputCSV: str = "output/csv/keywords.csv", alternate_lists: str = "none", print_to_console: bool = False)-> None:
    """
    Scans all txt files in a given folder for keywords from the supplied csv file and any alternate lists specified. The output is stored as csv with the filename as index.
    
    Parameters:
    -----------
    TXTCorfolder (str): Path to the folder in which to store the unicode corrected txt files.

    keylist_path (str): Path to the csv containg the keywords to be used. Generated by construct_keylist().

    outputCSV (str): Path to the file in which to store the keywords indexed by document name. 

    alternate_lists (str): String defining the alternative lists to use. Options include: 'all', 'statistics', 'countries', 'genomics', 'phylogenies', 'ecology', 'culicid_genera' or any combinations thereof e.g. "statistics and countries".

    print_to_console (bool): Whether a summary of the output should be shown in console log

    Return:
    -----------
    None
    """
    print("scanning txt files for tags")
    
    keylist = construct_keylist(keylist_path, alternate_lists)
    
    df = pd.DataFrame(columns=["file", "keywords"])
    guarantee_csv_exists(outputCSV, df)

    outputfile = pd.read_csv(outputCSV)["file"].tolist()
    itemlist = list_filenames(TXTCorfolder, "*.txt")
    taglists = []
    
    for item in itemlist:
        if str(item) not in outputfile:
            with open(f"{TXTCorfolder}/{item}.txt", "rb") as file:
                for line in file:
                    text = preprocess_text(line)
                    keywords = find_keywords(keylist, text)
                unique_keywords = filter_uniques_from_list(keywords)

                taglists.append(unique_keywords)  # Add tags from current document to taglists

                # create pandas row
                dfcolumn = pd.DataFrame(
                    {"file": str(item), "keywords": str(unique_keywords)}, index=[0]
                )
                dfcolumn.to_csv(outputCSV, mode="a", index=False, header=False)
            print(f"tagged {item}")
    
    print("finished writing tags to keywords.csv")
    tagcounts = calculate_tag_counts(taglists)

    if print_to_console:
        print(tagcounts)
    return 

def tag_csv(inputCSV: str, outputCSV: str = "", outputfolder: str = "C:/NLPvenv/NLP/output/csv", titlecolumn: str = "Title", abstractcolumn: str = "Abstract", keylist_path: str = "input/keylist.csv", alternate_lists: str = "none", print_to_console: bool = False) -> None:
    """
    Scans all items in a csv file for keywords and any alternate lists specified.
    
    Parameters:
    -----------
    inputCSV (str): Path to the file containing the items (title and abstract) to scan. 

    outputCSV (str): Filename which to store the keywords indexed by document name. 

    outputfolder (str): Path to the folder in which the output should be stored.

    keylist_path (str): Path to the csv containg the keywords to be used. Generated by construct_keylist().

    alternate_lists (str): String defining the alternative lists to use. Options include: 'all', 'statistics', 'countries', 'genomics', 'phylogenies', 'ecology', 'culicid_genera' or any combinations thereof e.g. "statistics and countries".

    print_to_console (bool): Whether a summary of the output should be shown in console log

    Return:
    -----------
    None
    """
    print("scanning csv for tags")

    if len(outputCSV) == 0:
        outputname = os.path.basename(inputCSV)
        outputCSV = outputfolder + "/" + outputname
        print(f"Output file set to: {outputCSV}")
    keylist = construct_keylist(keylist_path, alternate_lists)

    df = pd.read_csv(inputCSV)
    df['Keywords'] = None
    guarantee_csv_exists(outputCSV, df)
    
    taglists = []

    for i in range(len(df)):
        if df['Keywords'][i] == None:
            item = str(df[titlecolumn][i]) + " " + str(df[abstractcolumn][i])
            title = df[titlecolumn][i]
            text = preprocess_text(item)
            keywords = find_keywords(keylist, text)
            unique_keywords = filter_uniques_from_list(keywords)
            
            taglists.append(unique_keywords)  # Add tags from current document to taglists
            
            df.loc[df[titlecolumn] == title, 'Keywords'] = ', '.join(unique_keywords)
    
    df.to_csv(outputCSV, index=False)
    print("finished writing tags to keywords.csv")
    tagcounts = calculate_tag_counts(taglists)

    if print_to_console:
        pd.set_option('display.max_rows', None) 
        print(tagcounts)
    return

def tag_file_weighted(filename: str, keylist_path: str = "input/keylist.csv", alternate_lists: str ="all", treshold: int = 2, print_to_console: bool = False) -> dict[str, int]:
    """
    Scans a file for tags and weighs them based on the counts and locations of each occurrence within the document.
    """
    section_dictionary = split_text_to_sections(filename)
    keylist = construct_keylist(
        keylist_path, alternate_lists)
    word_counts = count_keyword_occurrences(section_dictionary, keylist)
    word_counts_filtered = filter_values(word_counts, 1)
    word_counts_weighed = weigh_keywords(word_counts_filtered)
    word_counts_end = filter_values(word_counts_weighed, treshold)
    denested = denest_and_order_dict(word_counts_end)
    if print_to_console == True:
        print_nested_dict(denested)
    return denested

def tag_folder_weighted(input_path: str, outputCSV="output/csv/keywords.csv", keylist_path:str = "input/keylist.csv", alternate_lists: str = "all", treshold: int = 2,  print_to_console: bool = False) -> None:
    """
    Scans each file within a folder for tags and weighs them based on the counts and locations of each occurrence within the document. Saves the data to a csv file in the provided folder and prints the data if output is set to True

    Parameters:
    -----------
    input_path (str): Path of the folder containing the files to scan

    outputCSV (str): Path to the file in which to store the keywords indexed by document name. 

    print_to_console (bool): whether a summary of the output should be shown in console log

    Return:
    -----------
    tagcounts (csv): dataframe containing the calculated counts per file per keyword

    """
    print("scanning txt files for tags")
    
    df = pd.DataFrame(columns=["file", "keywords"])
    guarantee_csv_exists(outputCSV, df)

    outputfile = pd.read_csv(outputCSV)["file"].tolist()
    itemlist = list_filenames(input_path, "*.txt")
    taglists = []

    for item in itemlist:
        file_path = f"{input_path}/{item}.txt"
        if str(item) not in outputfile:
            text_keylist = list(tag_file_weighted(file_path, keylist_path, alternate_lists, treshold).keys())
            
            taglists.append(text_keylist) 

            # create pandas row
            dfcolumn = pd.DataFrame(
            {"file": str(item), "keywords": str(text_keylist)}, index=[0])
            dfcolumn.to_csv(outputCSV, mode="a", index=False, header=False)
            print("tagged", str(item))
    
    print("finished writing tags to keywords.csv")
    tagcounts = calculate_tag_counts(taglists)

    if print_to_console == True:
        print(tagcounts)
    return 

## write the tags back into the .bib file
def write_bib(output_csv_file="output/csv/keywords.csv", libtex_csv="input/savedrecs.csv", bibfile="", bibfolder ="output/bib", CSVtotal="output/csv/total.csv") -> None:
    """
    Imports the tags from csvtotal and combines them with the information given in the bibfile after which it stores the information as csv and bib.
    
    Parameters:
    -----------
    - outputCSV (str): A filepath to a CSV file with the 'keywords' column containing tags to be merged with tags generated by the program.
    
    - libtex_csv (str): A filepath to a CSV file with bibliographic information to be updated with merged tags.
    
    - bibfile (str): A filepath to a .bib file with bibliographic entries to be updated with tags.
    
    - bibfolder (str): A filepath to the directory where the updated .bib file should be saved.
    
    - CSVtotal (str): A filepath to the directory where the merged CSV file should be saved.

    Returns:
    -----------
    None
    """
    keyframe = pd.read_csv(output_csv_file).rename(columns={"keywords": "generated"})
    libframe = pd.read_csv(libtex_csv)
    
    # join dataframes and extract tags
    totalframe = libframe.merge(keyframe, how="outer").fillna("")
    totalframe["keywords"] = totalframe.apply(
        lambda row: ", ".join(set(str(row["keywords"]).lower().split(", ") + str(row["generated"]).lower().split(", "))).replace("'", "").replace("[", "").replace("]", ""),
        axis=1
    )
    
    # write all libtex data to csv
    totalframe.drop(columns=["generated"]).to_csv(CSVtotal, index=False)
    
    bib_data = parse_file(bibfile)
    end = os.path.splitext(os.path.basename(bibfile))[0]
    # add tags to bibtex library
    for entry in bib_data.entries.values():
        entryname = str(entry.key)
        keywordlookup = totalframe.loc[totalframe["entry"] == entryname, "keywords"].iloc[0]
        entry.fields["keywords"] = keywordlookup
    bib_data.to_file(f"{bibfolder}/{end}_tagged.bib")
    return

## create summary per article
# generate markdown files per article. The files are dynamically updates using js-code.
def write_article_summaries(CSVtotal: str="output/csv/total.csv", Article_template: str="input/templates/Paper.md", mdFolder: str="output/md",) -> None:
    """
    The write_article_summaries() function creates article summaries from a given CSV file of article information and a markdown template file. The summaries are saved as markdown files in a specified folder.

    Inputs:
    -----------
    CSVtotal (str): The path to a CSV file containing the article information. The file must contain the following columns: keywords, title, abstract, author, year, containerTitle, page, doi, id, URL, file, and note.

    Article_template (str): The path to a markdown file that contains a template for the article summaries. The template should include placeholders that correspond to the columns in the CSV file.

    mdFolder (str): The path to a folder where the markdown files will be saved.

    Return:
    -----------
    The function does not return any output, but saves the article summaries as markdown files in the specified folder.

    Note: The function uses several string manipulation functions to clean and format text data.
    """
    print("creating article summaries")
    dataframe = pd.read_csv(CSVtotal)
    with open(Article_template, "r") as f:
        template_text = f.read()
    placeholderlist = [
        "VALUE:keywords",
        "VALUE:title",
        "VALUE:abstract",
        "VALUE:cite",
        "VALUE:author",
        "VALUE:year",
        "VALUE:containerTitle",
        "VALUE:page",
        "VALUE:doi",
        "VALUE:id",
        "VALUE:URL",
        "VALUE:file",
        "VALUE:note",
    ]
    valuelist = [
        "keywords",
        "title",
        "abstract",
        "author",
        "author_corrected",
        "date",
        "journaltitle",
        "pages",
        "doi",
        "entry",
        "url",
        "file",
        "note",
    ]
    # create md file per article and repopulate other placeholders
    for index, row in dataframe.iterrows():
        if str(row["title"]) != "nan":
            # Create file with title as name
            title = str(row["title"])
            illegal_chars = [":", ";", "=", ".", "{", "}", "?", ",", "\\", "/", "*", '"', "'", "textgreater", "textbackslash", "textlessI", "textlessspan"]
            for char in illegal_chars:
                title = title.replace(char, "_")
            name = title
            fullname = mdFolder + "/papers/Note " + name + ".md"
            value = check_record_type(row)
            data = template_text
            data = data.replace("VALUE:type", value)
            # populate placeholders
            data = populate_placeholders(placeholderlist, valuelist, row, data)
            populate_with_template(fullname, data)
            print(f"generated atricle summary for {row['title']}")
        else:
            print("skipped ", str(row["title"]))
    return


## create javascript based dynamic summary per author
def write_author_summaries(CSVtotal: str = "output/csv/total.csv", Author_template: str = "input/templates/Author.md", mdFolder: str = "output/md") -> None:
    """
    Creates author summaries per author per article from a given CSV file of article information and a markdown template file. The summaries are saved as markdown files in a specified folder.

    Inputs:
    -----------
    CSVtotal (str): The path to a CSV file containing the article information. The file must contain the following columns: keywords, title, abstract, author, year, containerTitle, page, doi, id, URL, file, and note.

    Author_template (str): The path to a markdown file that contains a template for the author summaries. The template should include placeholders that correspond to the columns in the CSV file.

    mdFolder (str): The path to a folder where the markdown files will be saved.

    Returns:
    -----------
    None
    """
    print("Creating author summaries")
    
    dataframe = pd.read_csv(CSVtotal)
    template_text = open(Author_template, "r").read()
    
    authorlist = []
    
    for index, row in dataframe.iterrows():
        if row["title"] != "nan" and str(row["author"]) != "nan":
            authorlist.extend(author_to_firstname_lastname(row))
    
    unique_authors = collapse_authors(authorlist)
    
    for author in unique_authors:
        if str(author) != "empty":  
            filename = f"{mdFolder}/authors/{author}.md"    
            with open(filename, "wb") as file:
                for line in template_text:
                    file.write(str(line).encode("utf-8"))
                file.close()
                
            print(f"Generated author summary for {author}")
    
    print("Generated author summaries")
    return


## create javascript based dynamic summary per journal
def write_journal_summaries(CSVtotal: str = "output/csv/total.csv", Journal_template: str = "input/templates/Journal.md", mdFolder: str = "output/md") -> None:
    """
    Reads a CSV file containing information about journals and creates summary files for each journal using a template.
    The summary files are saved in a specified directory. The function iterates over the rows in the CSV file and for each row,
    it checks if the "journaltitle" column is not null. If it is not null, it removes special characters from the journal title
    and creates a file with the modified title as its name.

    Parameters:
    -----------
    CSVtotal (str): The path to the CSV file containing journal information.
    Journal_template (str): The path to the template file to be used for creating journal summary files.
    mdFolder (str): The path to the directory where the journal summary files will be saved.

    Returns:
    -----------
    None
    """
    print("Creating journal summaries")
    
    dataframe = pd.read_csv(CSVtotal)
    template_text = open(Journal_template, "r").read()

    for index, row in dataframe.iterrows():
        if not pd.isnull(row["journaltitle"]):
            journal_title = str(row["journaltitle"])
            cleaned_title = remove_special_characters(journal_title)
            file_name = mdFolder + "/journals/" + cleaned_title + ".md"

            with open(file_name, "w", encoding="utf-8") as file:
                file.write(template_text)
            print(f"generated journal summary for {cleaned_title}")
            
    print("Generated journal summaries")
    return


### main workflows
def prepare_input(source_folder="", PDFfolder="input/pdf", TXTfolder="input/pdf/docs", TXTCorfolder="input/pdf/docs/corrected") -> None:
    """
    Collects pdf files from given folder and subfolders, converts them to txt and cleans unicode decoding errors.

    Parameters:
    -----------
    source_folder (str): the reference manager path containing all pdf files.

    PDFfolder (str): Path to the folder in which to store all pdf files.

    TXTCorfolder (str): Path to the folder in which to store the txt files.
    
    TXTCorfolder (str): Path to the folder in which to store the unicode corrected txt files.

    Return:
    -----------
    None
    """
    collect_PDF_files(source_folder, PDFfolder)
    pdf2txtfolder(PDFfolder, TXTfolder)
    unicodecleanup_folder(TXTfolder, TXTCorfolder)
    return


def create_summaries(mdFolder="output/md", Article_template="input/templates/Paper.md", Author_template="input/templates/Author.md", Journal_template="input/templates/Journal.md", CSVtotal="output/csv/total.csv") -> None:
    """
    Creates a summary per article and js based dynamic summaries per author and journal.
    
    Parameters:
    -----------
    mdFolder (str): Path to the folder in which to store the .md output.

    Article_template (str): Path to the template for the article .md files.

    Author_template (str): Path to the template for the author .md files.

    Journal_template (str): Path to the template for the journal .md files.
    
    CSVtotal (str): Path to the file in which to store the .csv output.

    Return:
    -----------
    None

    """
    guarantee_md_output_folders_exist(mdFolder)
    correct_authornames(CSVtotal)
    write_article_summaries(CSVtotal, Article_template, mdFolder)
    write_author_summaries(CSVtotal, Author_template, mdFolder)
    write_journal_summaries(CSVtotal, Journal_template, mdFolder)


### complete tagging routine
def automated_pdf_tagging(source_folder:str="", PDFfolder:str="input/pdf", TXTfolder:str="input/pdf/docs", TXTCorfolder:str="input/pdf/docs/corrected", keylist_path:str="input/keylist.csv", outputCSV:str="output/csv/keywords.csv", libtex_csv:str="input/savedrecs.csv", bibfile:str="", bibfolder:str="output/bib", CSVtotal:str="output/csv/total.csv", mdFolder:str="output/md", Article_template:str="input/templates/Paper.md", Author_template:str="input/templates/Author.md", Journal_template:str="input/templates/Journal.md", alternate_lists:str="none", weighted:bool= False, treshold:int = 2, summaries:bool = False) -> None:
    """
    Complete workflow for pdf tagging. Define 1) the reference manager path containing all pdf files and 2) the path to the .bib file, 3) the alternative taglist to include (defaults to "none").

    Parameters:
    -----------
    source_folder (str): the reference manager path containing all pdf files.

    PDFfolder (str): Path to the folder in which to store all pdf files.

    TXTCorfolder (str): Path to the folder in which to store the txt files.
    
    TXTCorfolder (str): Path to the folder in which to store the unicode corrected txt files.

    keylist_path (str): Path to the csv containg the keywords to be used. Generated by construct_keylist().

    outputCSV (str): Path to the file in which to store the keywords indexed by document name. 

    libtex_csv (str): Path the csv file containing all data exported from the csv fille.

    bibfile (str): Path to the .bib file to which to add the tags. 

    bibfolder (str): Path to the folder in which to store the .bib ouput.

    CSVtotal (str): Path to the file in which to store the .csv output.

    mdFolder (str): Path to the folder in which to store the .md output.

    Article_template (str): Path to the template for the article .md files.

    Author_template (str): Path to the template for the author .md files.

    Journal_template (str): Path to the template for the journal .md files.

    alternate_lists (str): String defining the alternative lists to use. Options include: 'all', 'statistics', 'countries', 'genomics', 'phylogenies', 'ecology', 'culicid_genera' or any combinations thereof e.g. "statistics and countries".
    
    weighted (bool): Indicates whether keywords should be weighted by section of occurrence.

    treshold (int): Minimum value used to either store or dispose a keywords after weighing. 

    summaries (bool): Indicates whether markdown summaries per article, author and journal should be generated.

    Return:
    -----------
    None
    """    
    prepare_input(source_folder, PDFfolder, TXTfolder, TXTCorfolder)
    guarantee_folder_exists("output/bib")
    guarantee_folder_exists("output/csv")
    guarantee_folder_exists("output/md")
    if weighted == True:
        tag_folder_weighted(input_path = TXTCorfolder, keylist_path = keylist_path, alternate_lists = alternate_lists, treshold = 2)
    else:
        tag_folder(TXTCorfolder, keylist_path, outputCSV, alternate_lists)
    write_bib(outputCSV, libtex_csv, bibfile, bibfolder, CSVtotal)
    if summaries == True:
        create_summaries(mdFolder, Article_template, Author_template, Journal_template, CSVtotal)
    return


if __name__ == "__main__":
    #automated_pdf_tagging(source_folder="C:/Users/sboer/Zotero/storage", bibfile="input/library.bib", alternate_lists="all", weighted = True, treshold = 5, summaries = True)
    #tag_csv(inputCSV="C:/NLPvenv/NLP/input/savedrecs_lianas.csv", titlecolumn="Article Title", keylist_path="C:/NLPvenv/NLP/input/keylist_lianas.csv", print_to_console=True)
    df = pd.read_csv("C:/NLPvenv/NLP/output/csv/savedrecs_lianas.csv")['Keywords']
    summary = calculate_tag_counts(df)
    pd.set_option('display.max_rows', None) 
    print(summary)