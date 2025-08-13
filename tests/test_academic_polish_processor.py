"""
Unit tests for Academic Polish Processor

Tests the academic polish enhancement functionality added in Story 2.6
"""

import unittest
from src.post_processors.academic_polish_processor import AcademicPolishProcessor, PolishIssue


class TestAcademicPolishProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = AcademicPolishProcessor()
    
    def test_filler_word_removal(self):
        """Test removal of filler words and phrases"""
        test_cases = [
            ("Um, today we will discuss dharma.", "Today we will discuss dharma."),
            ("Uh, this verse speaks about karma.", "This verse speaks about karma."),
            ("We study, you know, the Gita today.", "We study the Gita today."),
            ("I mean, Krishna teaches wisdom.", "Krishna teaches wisdom."),
            ("The the soul is eternal.", "The soul is eternal."),
            ("And and dharma guides us.", "And dharma guides us."),
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                result, issues = self.processor.polish_srt_content(f"1\n00:00:01,000 --> 00:00:05,000\n{original}\n")
                lines = result.strip().split('\n')
                polished_text = lines[2] if len(lines) > 2 else ""
                self.assertEqual(polished_text, expected, 
                                f"Failed to remove filler words from: '{original}'")
    
    def test_capitalization_fixes(self):
        """Test sentence capitalization fixes"""
        test_cases = [
            ("today we study dharma.", "Today we study dharma."),
            ("Krishna teaches. then we learn.", "Krishna teaches. Then we learn."),
            ("What is dharma? it guides us.", "What is dharma? It guides us."),
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                result, issues = self.processor.polish_srt_content(f"1\n00:00:01,000 --> 00:00:05,000\n{original}\n")
                lines = result.strip().split('\n')
                polished_text = lines[2] if len(lines) > 2 else ""
                self.assertEqual(polished_text, expected)
    
    def test_sanskrit_term_standardization(self):
        """Test Sanskrit/Hindi term capitalization"""
        test_cases = [
            ("Today we study krishna.", "Today we study Krishna."),
            ("The bhagavad gita teaches us.", "The Bhagavad Gita teaches us."),
            ("rama is divine.", "Rama is divine."),
            ("We learn about shiva.", "We learn about Shiva."),
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                result, issues = self.processor.polish_srt_content(f"1\n00:00:01,000 --> 00:00:05,000\n{original}\n")
                lines = result.strip().split('\n')
                polished_text = lines[2] if len(lines) > 2 else ""
                self.assertEqual(polished_text, expected)
    
    def test_format_consistency(self):
        """Test format consistency fixes"""
        test_cases = [
            ("Today we study dharma  and karma.", "Today we study dharma and karma."),
            ("The soul cannot be cut , burned.", "The soul cannot be cut, burned."),
            ("We learn -- Krishna teaches.", "We learn—Krishna teaches."),
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                result, issues = self.processor.polish_srt_content(f"1\n00:00:01,000 --> 00:00:05,000\n{original}\n")
                lines = result.strip().split('\n')
                polished_text = lines[2] if len(lines) > 2 else ""
                self.assertEqual(polished_text, expected)
    
    def test_srt_format_preservation(self):
        """Test that SRT format is preserved during polishing"""
        srt_content = """1
00:00:01,000 --> 00:00:05,000
um, today we study krishna.

2
00:00:06,000 --> 00:00:10,000
the bhagavad gita teaches us."""

        result, issues = self.processor.polish_srt_content(srt_content)
        lines = result.strip().split('\n')
        
        # Check subtitle numbers preserved
        self.assertEqual(lines[0], "1")
        self.assertEqual(lines[4], "2")
        
        # Check timestamps preserved
        self.assertEqual(lines[1], "00:00:01,000 --> 00:00:05,000")
        self.assertEqual(lines[5], "00:00:06,000 --> 00:00:10,000")
        
        # Check content was polished
        self.assertEqual(lines[2], "Today we study Krishna.")
        self.assertEqual(lines[6], "The Bhagavad Gita teaches us.")
    
    def test_comprehensive_polish_workflow(self):
        """Test the complete academic polish workflow"""
        test_content = """1
00:00:01,000 --> 00:00:05,000
um, today we will discuss, uh, krishna and dharma.

2
00:00:06,000 --> 00:00:10,000
the bhagavad gita teaches. then we learn about rama."""

        polished_content, polish_issues = self.processor.polish_srt_content(test_content)
        
        # Should have multiple types of issues fixed
        issue_types = {issue.issue_type for issue in polish_issues}
        expected_types = {'filler_word_removal', 'capitalization', 'sanskrit_standardization'}
        self.assertTrue(expected_types.issubset(issue_types), 
                       f"Missing issue types. Expected: {expected_types}, Got: {issue_types}")
        
        # Verify final content is academically polished
        lines = polished_content.strip().split('\n')
        self.assertEqual(lines[2], "Today we will discuss Krishna and dharma.")
        self.assertEqual(lines[6], "The Bhagavad Gita teaches. Then we learn about Rama.")


if __name__ == '__main__':
    unittest.main()