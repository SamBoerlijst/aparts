import os
import unittest

from aparts.src.summarization import (generate_sentence_tokens,
                                      remove_repeating_sentences,
                                      summarize_csv, summarize_file,
                                      summarize_text, summarize_tokens)


class TestSummarizerFunctions(unittest.TestCase):

    def setUp(self):
        self.sample_text = """
            Anthropogenic stressors on the environment are increasing at unprecedented rates and include urbanization, nutrient pollution, water management, 
            altered land use and climate change. Their effects on disease vectors are poorly understood.A series of full factorial experiments investigated 
            how key human induced abiotic pressures, and interactions between these, affect population parameters of the cosmopolitan disease vector, 
            Culex pipiens  s.l. Selected pressures include eutrophication, salinity, mean temperature, and temperature fluctuation.Data were collected for 
            each individual pressure and for potential interactions between eutrophication, salinization and temperature. All experiments assessed survival, 
            time to pupation, time to emergence, sex-ratio and ovipositioning behavior.The results show that stressors affect vector survival, may speed up 
            development and alter female to male ratio, although large differences between stressors exist to quite different extents. While positive effects 
            of increasing levels of eutrophication on survival were consistent, negative effects of salinity on survival were only apparent at higher 
            temperatures, thus indicating a strong interaction effect between salinization and temperature. Temperature had no independent effect on larval 
            survival. Overall, increasing eutrophication and temperatures, and the fluctuations thereof, lowered development rate, time to pupation and time 
            to emergence while increasing levels of salinity increased development time. Higher levels of eutrophication positively impacted egg-laying 
            behavior; the reverse was found for salinity while no effects of temperature on egg-laying behavior were observed.Results suggest large and 
            positive impacts of anthropogenically induced habitat alterations on mosquito population dynamics. Many of these effects are exacerbated by 
            increasing temperatures and fluctuations therein. In a world where eutrophication and salinization are increasingly abundant, mosquitoes are 
            likelyimportant benefactors. Ultimately, this study illustrates the importance of including multiple and combined stressors in predictive models 
            as well as in prevention and mitigation strategies, particularly because they resonate with possible, but yet underdeveloped action plans. 
        """

        self.sections = {
            'section_1': ["This is the first section of the text."],
            'section_2': ["This is the second section of the text."],
        }

    def test_remove_repeating_sentences(self):
        sentence_tokens, _ = generate_sentence_tokens(self.sample_text)
        sentence_tokens_filtered = remove_repeating_sentences(
            sentence_tokens, self.sections)
        self.assertTrue(isinstance(sentence_tokens_filtered, list))

    def test_generate_sentence_tokens(self):
        sentence_tokens, word_frequencies = generate_sentence_tokens(
            self.sample_text)
        self.assertTrue(isinstance(sentence_tokens, list))
        self.assertTrue(isinstance(word_frequencies, dict))
        self.assertTrue(len(sentence_tokens) > 0)
        self.assertTrue(len(word_frequencies) > 0)

    def test_summarize_tokens(self):
        sentence_tokens, word_frequencies = generate_sentence_tokens(
            self.sample_text)
        summary = summarize_tokens(
            sentence_tokens, word_frequencies, amount=2, offset=0)
        self.assertTrue(isinstance(summary, str))
        self.assertTrue(len(summary) > 0)
        self.assertTrue(len(summary) < len(self.sample_text))

    def test_summarize_text(self):
        summary2 = summarize_text(
            self.sample_text, self.sections, amount=2, offset=0)
        summary3 = summarize_text(
            self.sample_text, self.sections, amount=3, offset=0)
        self.assertTrue(isinstance(summary2, str))
        self.assertTrue(len(summary2) > 0)
        self.assertTrue(len(summary2) < len(summary3))

    def test_summarize_file(self):
        filepath = 'C:/NLPvenv/aparts/app/aparts/test/test.txt'
        summary = summarize_file(filepath, ['abstract'], amount=2, offset=0)
        self.assertTrue(isinstance(summary, str))
        self.assertTrue(len(summary) > 0)

    def test_summarize_csv(self):
        outputCSV = "output_summary.csv"  # Update with the actual path
        os.makedirs(self.txtfolder, exist_ok=True)
        with open(os.path.join(self.txtfolder, "sample_file.txt"), 'w') as f:
            f.write(self.sample_text)
        summary = summarize_csv(outputCSV, self.txtfolder, list(self.sections.keys()), amount=2, offset=0)
        self.assertTrue(summary is None)  # Since it's a void function

if __name__ == '__main__':
    unittest.main()
