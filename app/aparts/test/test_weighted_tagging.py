import unittest

from aparts.src.weighted_tagging import (clean_end_section,
                                         count_keyword_occurrences,
                                         extract_sections,
                                         prepare_bytes_for_pattern,
                                         remove_typographic_line_breaks)


class TestTaggingFunctions(unittest.TestCase):

    def test_prepare_bytes_for_pattern(self):
        input_text = 'b abstract: This is an abstract. keywords: important, keywords introduction: Intro text'
        result = prepare_bytes_for_pattern(input_text)
        self.assertEqual(result, 'This is an abstract. keywords: important, keywords introduction: Intro text')

    def test_remove_typographic_line_breaks(self):
        input_text = "This is a te- st with a hyphen."
        result = remove_typographic_line_breaks(input_text)
        self.assertEqual(result, "This is a test with a hyphen.")

    def test_extract_sections(self):
        input_text = "abstract: This is an abstract. introduction: Intro text"
        result = extract_sections(input_text)
        expected_result = {
            "abstract": "This is an abstract.",
            "keywords": "",
            "introduction": "Intro text",
            "methods": "",
            "results": "",
            "discussion": "",
            "conclusion": "",
            "references": ""
        }
        self.assertDictEqual(result, expected_result)

    def test_clean_end_section(self):
        sections = {
            "abstract": "This is an abstract. keywords",
            "introduction": "Intro text. methods",
            "methods": "",
            "results": "",
            "discussion": "",
            "conclusion": "",
            "references": ""
        }
        result = clean_end_section(sections)
        print(result)
        expected_result = {
            "abstract": "",  # Update this line
            "introduction": "",  # Update this line
            "methods": "",
            "results": "",
            "discussion": "",
            "conclusion": "",
            "references": ""
        }
        self.assertDictEqual(result, expected_result)
        self.assertDictEqual(result, expected_result)

    # Add more test cases for other functions


if __name__ == '__main__':
    unittest.main()
