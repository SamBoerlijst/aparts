import io
import os
from fnmatch import fnmatch
from gc import collect
from shutil import copy2

import bibtexparser as bibtex
import pandas as pd
from anyascii import anyascii
from bibtexparser.bparser import BibTexParser
from pybtex.database.input import bibtex
from PyPDF2 import PdfReader

""" APART
- 2023/4/30 Sam Boerlijst
- Academic Pdf - Automated Reference Tagging: extract pdf from refmanager folder, convert to lowercase utf-8 txt, index keywords from keylist and store those in csv, bib and md format
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

# build class to store tags and number of occurences
class scorelist:
    def __init__(self, tag:str, score:int):
        self.tag = tag
        self.score = score

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
    for file_ in files:
        if fnmatch(file_, type):
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(root_dst_dir, file_)
            if os.path.exists(dst_file) & overwrite == 1:
                os.remove(dst_file)
            if not os.path.exists(dst_file):
                copy2(src_file, root_dst_dir)

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
    del(src_dir, dirs, files, name, ext, end, dst_dir, src_file, dst_file)
    collect()

def guarantee_folder_exists(folder:str) -> None:
    """Create folder if not yet present."""
    if not os.path.exists(folder):
        os.makedirs(folder)

## file manipulation
def guarantee_csv_exists(filename:str, df:pd.DataFrame) -> None:
    """Create csv if not yet present populated by the given dataframe."""
    if os.path.isfile(filename) == False:
        df.to_csv(filename, index = False)

## indexing
def list_filenames(folder:str, type:str) -> list:
    """
    List all filenames of the supplied type within a given folder.

    Parameters:
    -----------
    folder (str) : Path to the folder where the files are located.
    
    type (str): Extension of the files to filter.

    Returns:
    --------
    file_list (list): List of files of given extension.
    """
    file_list = []
    for path, folders, files in os.walk(folder):
        for file in files:
            if fnmatch(file, type):
                name, ext = os.path.splitext(file)
                end = name.split("/").pop()
                file_list.append(end)
    return file_list

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
            # print(f'EOF found at line position {-i} = actual {actual_line}, with value {x}')
            break
    # return the list up to that point
    return pdf_stream_in[:actual_line]

def convert_unicode_from_string(string:str)->str:
    """Convert an unicode string to utf-8."""
    string = string.encode("unicode-escape").decode()
    string = string.replace("\\n", " ")
    string = repr(string)
    string = anyascii(string)
    string = string.strip("'").encode().decode("unicode-escape")
    return string

def remove_trailing_backslashes(string:str)->str:
    """Remove trailing backslashes from a string caused by decoding errors."""
    string = (
        string.replace("\\\\", "\\")
        .replace("\\\\", "\\")
    )
    return string

## convert pdf to txt, transform to lowercase and store in input folder
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
    # add txt files for pdfs that havent been processed while correcting eof
    for file_ in input_file_list:
        if str(file_) not in output_file_list & os.path.isfile(f"{TXTfolder}/{file_}.txt") == False:
            with open(f"{PDFfolder}/{file_}.pdf", "rb") as p:
                contents = p.readlines()
                contents_eof_corrected = reset_eof_of_pdf_return_stream(contents)
            with open(f"{PDFfolder}/{file_}.pdf", "wb") as d:
                d.writelines(contents_eof_corrected)
            with open(f"{PDFfolder}/{file_}.pdf", "rb") as f:
                read_pdf = PdfReader(f)
                number_of_pages = len(read_pdf.pages)
            # create txt with same name in output folder
            with open(f"{TXTfolder}/{file_}.txt", "w+") as t:
                # write pagecontent to files
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
    del(input_file_list, output_file_list, name, end, page_content, page_content_utf8)
    collect()
    # add way to deal with exceptions

## get rid of decode errors and store in subfolder
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
    file_list = [list_filenames(TXTfolder)]
    file_list1 = [list_filenames(TXTCorfolder)]
    # add txt files for txt that havent been processed
    for file_ in file_list:
        if str(file_) not in file_list1:
            with open(f"{TXTfolder}/{file_}.txt", "r") as t:
                file1 = open(f"{TXTCorfolder}/{file_}.txt", "w+")
                for line in t:
                    string_utf8 = convert_unicode_from_string(line)
                    string_utf8 = remove_trailing_backslashes(string_utf8)
                    file1.write(str(string_utf8.encode()))
                    file1.write("\n")
                file1.close()
                print(f"corrected encoding for {file_}")
    del(file_list, file_list1, name, end, ext)
    collect()

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

def find_keywords(keylist:list, text:str) ->list:
    """
    Scan a string for keywords from the given list.

    Parameters:
    -----------
    keylist (list): List of stings containing the keywords of interest.

    text (str): String of text to search for keywords.

    Returns:
    --------
    text_keylist (list): List of strings containing the found keywords.
    """
    text_keylist = []
    for i in range(len(keylist)):
        index = text.find(f"{keylist[i]} ")
        index1 = text.find(f" {keylist[i]}")
        index2 = text.find(f"{keylist[i]}-")
        index3 = text.find(f"-{keylist[i]}")
        index4 = text.find(f"{keylist[i]}.")
        index5 = text.find(f".{keylist[i]}")
        if (
            index != -1
            or index1 != -1
            or index2 != -1
            or index3 != -1
            or index4 != -1
            or index5 != -1
        ):
            text_keylist.append(keylist[i])
    return text_keylist

def get_filename(file:io.BufferedReader)->str:
    """Extract filename from path."""
    name, ext = os.path.splitext(file.name)
    end = name.split("/").pop()
    return end

def fix_broken_words(text:str)->str:
    """Fix words broken up by typesetting."""
    text = text.replace("- ", "")
    return text

def remove_special_characters(text:str)->str:
    """Delete illegal characters for filenames from string."""
    text = (
            text
            .replace("\\\\", "")
            .replace("\\", "")
            .replace("/", " ")
            .replace("_", " ")
            .replace("{", "")
            .replace("}", "")
            .replace("<", "")
            .replace(">", "")
            .replace("?", "")
            .replace("%", "")
            .replace("*", "")
            .replace(":", "")
            .replace("|", "")
            .replace("\"", "")
            .replace("'", "")
        )
    return text

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

## scan the txt files for strings contained in the keylist
def keylist_search(TXTCorfolder="input/pdf/docs/corrected", keylist_path="input/keylist_total.csv", outputCSV="output/csv/keywords.csv", alternate_lists:str="none"):
    """
    Scans all txt files in a given folder for keywords from the supplied csv file and any alternate lists specified. The output is stored as csv with the filename as index.
    
    Parameters:
    -----------
    TXTCorfolder (str): Path to the folder in which to store the unicode corrected txt files.

    keylist_path (str): Path to the csv containg the keywords to be used. Generated by construct_keylist().

    outputCSV (str): Path to the file in which to store the keywords indexed by document name. 

    alternate_lists (str): String defining the alternative lists to use. Options include: 'all', 'statistics', 'countries', 'genomics', 'phylogenies', 'ecology', 'culicid_genera' or any combinations thereof e.g. "statistics and countries".

    Return:
    -----------
    None
    """
    print("scanning txt files for tags")
    
    keylist = construct_keylist(keylist_path, alternate_lists)
    
    df = pd.DataFrame(columns=["file", "keywords"])
    guarantee_csv_exists(outputCSV, df)

    outputfile = pd.read_csv(outputCSV)["file"].tolist()
    file_list = [list_filenames(TXTCorfolder)]

    for file_ in file_list:
        if str(file_) not in outputfile:
            with open(f"{TXTCorfolder}/{file_}.txt", "rb") as file:
                for line in file:
                    text = preprocess_text(line)
                    text_keylist = find_keywords(keylist, text, )
                text_keylist = filter_uniques_from_list(text_keylist)
                # create pandas row
                dfcolumn = pd.DataFrame(
                    {"file": str(file_), "keywords": str(text_keylist)}, index=[0]
                )
                dfcolumn.to_csv(outputCSV, mode="a", index=False, header=False)
            print("tagged", str(file_))
    
    print("finished writing tags to keywords.csv")
    
    del(Keylist, file_list, templist, df, outputfile)
    collect()


## write the tags back into the .bib file
def write_bib(output_csv_file="output/csv/keywords.csv", libtex_csv="input/libtex.csv", bibfile="", bibfolder ="output/bibtex", CSVtotal="output/csv/total.csv") -> None:
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
    
    # add tags to bibtex library
    parser = BibTexParser(common_strings=True)
    with open(bibfile) as bib:
        bib_data = bibtex.load(bib, parser=parser)
        end = os.path.splitext(os.path.basename(bibfile))[0]
        with open(os.path.join(bibfolder, f"{end}_tagged.bib"), "w") as output:
            for entry in bib_data.entries.values():
                entryname = str(entry.key)
                keywordlookup = totalframe.loc[totalframe["entry"] == entryname, "keywords"].iloc[0]
                entry.fields["keywords"] = keywordlookup
            output.write(bibtex.dumps(bib_data))

## create summary per article
# generate markdown files per article. The files are dynamically updates using js-code.
def write_article_summaries(CSVtotal: str="output/csv/total.csv", Article_template: str="input/templates/Paper.md", mdFolder: str="output/md",):
    """
    The write_article_summaries() function creates article summaries from a given CSV file of article information and a markdown template file. The summaries are saved as markdown files in a specified folder.

    Inputs:
    -----------
    CSVtotal (str): The path to a CSV file containing the article information. The file must contain the following columns: keywords, title, abstract, author, year, containerTitle, page, doi, id, URL, file, and note.

    Article_template (str): The path to a markdown file that contains a template for the article summaries. The template should include placeholders that correspond to the columns in the CSV file.

    mdFolder (str): The path to a folder where the markdown files will be saved.

    Output:
    -----------
    The function does not return any output, but saves the article summaries as markdown files in the specified folder.

    Note: The function uses several string manipulation functions to clean and format text data.
    """
    print("creating article summaries")
    dataframe = pd.read_csv(CSVtotal)
    #create md folder if not present
    if not os.path.exists(mdFolder):
        os.makedirs(mdFolder)
        papers = mdFolder + "/papers"
        os.makedirs(papers)
        authors = mdFolder + "/authors"
        os.makedirs(authors)
        journals = mdFolder + "/journals"
        os.makedirs(journals)
    #open the template text
    with open(Article_template, "r") as f:
        template_text = f.read()
    #define placeholders and respective values
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
    pd_row = pd.DataFrame(columns=["author"])
    # change authors to firstname surname
    for index, row in dataframe.iterrows():
        if row["title"] != "nan":
            name = []
            authorlist = []
            authorlist_corrected = []
            if str(row["author"]) != "nan":
                authors = "".join(row["author"])
                # multiple authors
                if "and " in authors:
                    name = authors.split("and ")
                    for item in name:
                        if ", " in item:
                            lastname, firstname = str(item).split(", ")
                            if firstname.endswith(" "):
                                authorlist.append(
                                    str("[[" + firstname + lastname + "]]")
                                )
                            else:
                                authorlist.append(
                                    str("[[" + firstname + " " + lastname + "]]")
                                )
                        else:
                            authorlist.append(str("[[" + item + "]]"))
                # single author
                elif ", " in authors:
                    name = authors
                    lastname, firstname = str(name).split(", ")
                    if firstname.endswith(" "):
                        authorlist.append(str("[[" + firstname + lastname + "]]"))
                    else:
                        authorlist.append(str("[[" + firstname + " " + lastname + "]]"))
                else:
                    authorlist = authors
            else:
                authorlist = "nan"
            for ele in authorlist:
                if str(ele) != "empty":
                    # create file with title as name
                    name = (
                        str(ele)
                        .replace("{", "")
                        .replace("}", "")
                        .replace(".", "")
                    )
                    authorlist_corrected.append(name)
            authorlist_corrected = str(", ".join(authorlist_corrected))
            pd_row = pd.concat(
                [pd_row, pd.DataFrame({"author": authorlist_corrected}, index=[0])],
                ignore_index=True,
            )
    # repopulate authorcolumn with name surname
    dataframe["author_corrected"] = pd_row["author"]
    # create md file per article and repopulate other placeholders
    for index, row in dataframe.iterrows():
        if str(row["title"]) != "nan":
            # create file with title as name
            name = (
                str(row["title"])
                .replace(":", "_")
                .replace(";", "")
                .replace("=", "")
                .replace(".", "_")
                .replace("{", "")
                .replace("}", "")
                .replace("?", "")
                .replace(",", "")
                .replace("\\\\", "")
                .replace("\\", "")
                .replace("/", "")
                .replace("*", "")
                .replace('"', "")
                .replace("'", "")
                .replace("textgreater", "")
                .replace("textbackslash", "")
                .replace("textlessI", "")
                .replace("textlessspan", "")
            )
            fullname = mdFolder + "/papers/Note " + name + ".md"
            with open(fullname, "wb+") as file:
                # populate with template
                for line in template_text:
                    file.write(str(line).encode("utf-8"))
                    file.close
            with open(fullname, "r") as file:
                data = file.read()
                # determine type
                placeholder = "VALUE:type"
                if row["note"] != "nan" or row["isbn"] != "nan":
                    if "isbn" in str(row["note"]):
                        value = "book"
                    else:
                        value = "article"
                else:
                    value = "article"
                data = data.replace(placeholder, value)
                file.close
                # print('replacing ', placeholder, ' with ', value)
            with open(fullname, "wb") as file:
                file.write(str(data).encode("utf-8"))
                file.close
            # populate placeholders
            for variable in range(len(placeholderlist)):
                placeholder = str(placeholderlist[variable])
                value = row[str(valuelist[variable])]
                with open(fullname, "r") as file:
                    if placeholder == "VALUE:journal":
                        if str(value) != "nan":
                            value = (
                                value.replace("{", "")
                                .replace("}", "")
                                .replace(".", "")
                                .replace("&", "and")
                                .replace("\\\\", "")
                                .replace("\\", "")
                                .replace(":", "")
                                .replace(";", "")
                            )
                            data = data.replace(str(placeholder), str(value))
                        else:
                            data = data.replace(str(placeholder), str(value))
                    if placeholder == "VALUE:keywords":
                        if str(value) != "nan":
                            value = (
                                value.replace(" & ", "_&_")
                                .replace(": ", "/")
                                .replace(" ", "/")
                                .replace(",/", ", ")
                            )
                        data = data.replace(str(placeholder), str(value))
                    else:
                        data = data.replace(str(placeholder), str(value))
                    # print('replacing ', placeholder, ' with ', value)
                    file.close
                with open(fullname, "wb") as file:
                    file.write(str(data).encode("utf-8"))
                    file.close
            print("processed: ", str(row["title"]))
        else:
            print("skipped ", str(row["title"]))
    del(dataframe, template_text, placeholderlist, valuelist, pd_row, authorlist, authorlist_corrected, value, placeholder, data, file)
    collect()


## create javascript based dynamic summary per author
def write_author_summary(CSVtotal: str="output/csv/total.csv", Author_template: str="input/templates/Author.md", mdFolder: str="output/md",):
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
    print("creating author summaries")
    dataframe = pd.read_csv(CSVtotal)
    template_text = open(Author_template, "r").read()
    # change authors to firstname surname
    for index, row in dataframe.iterrows():
        if row["title"] != "nan":
            authordf = pd.DataFrame()
            name = []
            authorlist = []
            if str(row["author"]) != "nan":
                authors = "".join(row["author"])
                # multiple authors
                if "and " in authors:
                    name = authors.split(" and ")
                    for item in name:
                        if ", " in item:
                            lastname, firstname = str(item).split(", ")
                            if firstname.endswith(" "):
                                authorlist.append(str(firstname + lastname))
                            else:
                                authorlist.append(str(firstname + " " + lastname))
                        else:
                            authorlist.append(str(item))
                # single author
                elif ", " in authors:
                    name = authors
                    lastname, firstname = str(name).split(", ")
                    if firstname.endswith(" "):
                        authorlist.append(str(firstname + lastname))
                    else:
                        authorlist.append(str(firstname + " " + lastname))
                else:
                    authorlist = [authors]
                for ele in authorlist:
                    if str(ele) != "empty":
                        # create file with title as name
                        name = (
                            str(ele)
                            .replace("{", "")
                            .replace("}", "")
                            .replace(".", "")
                        )
                        fullname = mdFolder + "/authors/" + str(name) + ".md"
                        with open(fullname, "wb") as file:
                            # populate with template
                            for line in template_text:
                                file.write(str(line).encode("utf-8"))
                                file.close
                #print("created ", authorlist)
            else:
                authorlist = "nan"
    print("generated author summaries")
    del(dataframe, template_text, name, authorlist, authors, lastname, firstname, fullname, file)
    collect()


## create javascript based dynamic summary per journal
def write_journal_summary(CSVtotal: str="output/csv/total.csv", Journal_template: str="input/templates/Journal.md", mdFolder: str="output/md"):
    """
    Reads a CSV file containing information about journals and creates summary files for each journal using a template. The summary files are saved in a specified directory. The function iterates over the rows in the CSV file and for each row, it checks if the "journaltitle" column is not null. If it is not null, it removes special characters from the journal title and creates a file with the modified title as its name.

    Parameters:
    -----------
    CSVtotal (str): The path to the CSV file containing journal information.
    
    Journal_template (str): The path to the template file to be used for creating journal summary files.
    
    mdFolder (str): The path to the directory where the journal summary files will be saved.

    Returns:
    -----------
    None
    """
    print("creating journal summaries")
    dataframe = pd.read_csv(CSVtotal)
    template_text = open(Journal_template, "r").read()
    for index, row in dataframe.iterrows():
        if row["title"] != "nan":
            journal = ""
            name = []
            if str(row["journaltitle"]) != "nan":
                journal = "".join(row["journaltitle"])
                name = remove_special_characters(journal)
                fullname = mdFolder + "/journals/" + str(name) + ".md"
                with open(fullname, "wb") as file:
                    # populate with template
                    for line in template_text:
                        file.write(str(line).encode("utf-8"))
                        file.close
                #print("created ", name)
    print("generated journal summaries")
    del(dataframe, template_text, journal, name, fullname, file)
    collect()


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

def tag_input(TXTCorfolder="input/pdf/docs/corrected", keylist_path="input/keylist_total.csv", outputCSV="output/csv/keywords.csv", libtex_csv="input/libtex.csv", bibfile="", bibfolder ="output/bibtex", CSVtotal="output/csv/total.csv", alternate_lists="none") -> None:
    """
    Scans txt files for tags and exports results as csv and .bib.

    Parameters:
    -----------
    TXTCorfolder (str): Path to the folder in which to store the unicode corrected txt files.

    keylist_path (str): Path to the csv containg the keywords to be used. Generated by construct_keylist().

    outputCSV (str): Path to the file in which to store the keywords indexed by document name. 

    libtex_csv (str): Path the csv file containing all data exported from the csv fille.

    bibfile (str): Path to the .bib file to which to add the tags. 

    bibfolder (str): Path to the folder in which to store the .bib ouput.

    CSVtotal (str): Path to the file in which to store the .csv output.

    alternate_lists (str): String defining the alternative lists to use. Options include: 'all', 'statistics', 'countries', 'genomics', 'phylogenies', 'ecology', 'culicid_genera' or any combinations thereof e.g. "statistics and countries".

    Return:
    -----------
    None
    """
    keylist_search(TXTCorfolder, keylist_path, outputCSV, alternate_lists, )
    write_bib(outputCSV, libtex_csv, bibfile, bibfolder, CSVtotal)

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
    write_article_summaries(CSVtotal, Article_template, mdFolder)
    write_author_summary(CSVtotal, Author_template, mdFolder)
    write_journal_summary(CSVtotal, Journal_template, mdFolder)


### complete tagging routine
def automated_pdf_tagging(source_folder="", PDFfolder="input/pdf", TXTfolder="input/pdf/docs", TXTCorfolder="input/pdf/docs/corrected", keylist_path="input/keylist_total.csv", outputCSV="output/csv/keywords.csv", libtex_csv="input/libtex.csv", bibfile="", bibfolder ="output/bibtex", CSVtotal="output/csv/total.csv", mdFolder="output/md", Article_template="input/templates/Paper.md", Author_template="input/templates/Author.md", Journal_template="input/templates/Journal.md", alternate_lists="none") -> None:
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
    

    Return:
    -----------
    None
    """    
    prepare_input(source_folder, PDFfolder, TXTfolder, TXTCorfolder)
    tag_input(TXTCorfolder, keylist_path, outputCSV, libtex_csv, bibfile, bibfolder, CSVtotal, alternate_lists, )
    create_summaries(mdFolder, Article_template, Author_template, Journal_template, CSVtotal)


if __name__ == "__main__":
    automated_pdf_tagging(source_folder="C:/Users/sboer/Zotero/storage", bibfile="input/pc_Library_1-5-2023.bib", alternate_lists="all")