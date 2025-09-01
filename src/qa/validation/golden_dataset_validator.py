"""
Epic 4 - Story 4.3: Benchmarking & Continuous Improvement
Golden Dataset Validation System for accuracy measurement and benchmarking

This module provides:
- Automated accuracy validation against expert-verified content
- Word Error Rate (WER) calculation for Sanskrit/Hindi content
- IAST transliteration compliance checking
- Scripture verse identification accuracy
- Comprehensive quality metrics and reporting
"""

import os
import json
import logging
import re
import difflib
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import pysrt

# Text processing imports
import unicodedata
from fuzzywuzzy import fuzz
import Levenshtein

# Visualization and reporting
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ValidationMetrics:
    """Metrics from golden dataset validation."""
    overall_accuracy: float
    word_error_rate: float
    sanskrit_accuracy: float
    hindi_accuracy: float
    iast_compliance: float
    verse_accuracy: float
    character_accuracy: float
    
    # Detailed metrics
    total_segments: int
    processed_segments: int
    failed_segments: int
    
    # Error analysis
    substitution_errors: int
    insertion_errors: int
    deletion_errors: int
    
    # Performance metrics
    processing_time: float
    segments_per_second: float
    
    # Quality distribution
    high_quality_segments: int  # >95% accuracy
    medium_quality_segments: int  # 85-95% accuracy
    low_quality_segments: int  # <85% accuracy


@dataclass
class SegmentValidationResult:
    """Validation result for a single segment."""
    segment_id: str
    original_text: str
    processed_text: str
    golden_text: str
    
    accuracy_score: float
    word_error_rate: float
    character_error_rate: float
    
    sanskrit_terms_correct: int
    sanskrit_terms_total: int
    hindi_terms_correct: int
    hindi_terms_total: int
    
    iast_compliant: bool
    verse_identified: bool
    
    errors: List[Dict[str, Any]]
    suggestions: List[str]


class TextComparator:
    """Utility class for comparing and analyzing text differences."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize punctuation
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        return text
    
    def calculate_word_error_rate(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate (WER) between reference and hypothesis text.
        
        WER = (S + D + I) / N
        where S = substitutions, D = deletions, I = insertions, N = words in reference
        """
        ref_words = self.normalize_text(reference).split()
        hyp_words = self.normalize_text(hypothesis).split()
        
        # Use edit distance on word level
        operations = self._get_edit_operations(ref_words, hyp_words)
        
        substitutions = operations['substitutions']
        deletions = operations['deletions']
        insertions = operations['insertions']
        
        if len(ref_words) == 0:
            return 1.0 if len(hyp_words) > 0 else 0.0
        
        wer = (substitutions + deletions + insertions) / len(ref_words)
        return min(wer, 1.0)  # Cap at 1.0
    
    def calculate_character_error_rate(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate (CER)."""
        ref_chars = list(self.normalize_text(reference))
        hyp_chars = list(self.normalize_text(hypothesis))
        
        operations = self._get_edit_operations(ref_chars, hyp_chars)
        
        if len(ref_chars) == 0:
            return 1.0 if len(hyp_chars) > 0 else 0.0
        
        cer = (operations['substitutions'] + operations['deletions'] + operations['insertions']) / len(ref_chars)
        return min(cer, 1.0)
    
    def _get_edit_operations(self, reference: List[str], hypothesis: List[str]) -> Dict[str, int]:
        """Get detailed edit operations using Levenshtein distance."""
        # Use Levenshtein distance with operation details
        operations = Levenshtein.editops(reference, hypothesis)
        
        substitutions = sum(1 for op, _, _ in operations if op == 'replace')
        deletions = sum(1 for op, _, _ in operations if op == 'delete')
        insertions = sum(1 for op, _, _ in operations if op == 'insert')
        
        return {
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions
        }
    
    def get_detailed_diff(self, reference: str, hypothesis: str) -> List[Dict[str, Any]]:
        """Get detailed character-level differences."""
        ref_normalized = self.normalize_text(reference)
        hyp_normalized = self.normalize_text(hypothesis)
        
        differ = difflib.SequenceMatcher(None, ref_normalized, hyp_normalized)
        differences = []
        
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag != 'equal':
                differences.append({
                    'operation': tag,
                    'reference_text': ref_normalized[i1:i2],
                    'hypothesis_text': hyp_normalized[j1:j2],
                    'position': i1
                })
        
        return differences


class TermAccuracyAnalyzer:
    """Analyzer for Sanskrit/Hindi term accuracy."""
    
    def __init__(self, lexicon_paths: Dict[str, str]):
        self.lexicon_paths = lexicon_paths
        self.sanskrit_terms = self._load_sanskrit_terms()
        self.hindi_terms = self._load_hindi_terms()
        self.iast_patterns = self._compile_iast_patterns()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _load_sanskrit_terms(self) -> set:
        """Load Sanskrit terms from lexicons."""
        terms = set()
        
        try:
            if 'sanskrit' in self.lexicon_paths:
                # Load from YAML/JSON lexicon
                import yaml
                with open(self.lexicon_paths['sanskrit'], 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                # Extract terms from different sections
                for section in ['corrections', 'proper_nouns', 'verses']:
                    if section in data:
                        for term, variations in data[section].items():
                            terms.add(term)
                            if isinstance(variations, list):
                                terms.update(variations)
        
        except Exception as e:
            self.logger.warning(f"Could not load Sanskrit terms: {e}")
        
        return terms
    
    def _load_hindi_terms(self) -> set:
        """Load Hindi terms from lexicons."""
        terms = set()
        
        try:
            if 'hindi' in self.lexicon_paths:
                import yaml
                with open(self.lexicon_paths['hindi'], 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                for section in ['corrections', 'proper_nouns']:
                    if section in data:
                        for term, variations in data[section].items():
                            terms.add(term)
                            if isinstance(variations, list):
                                terms.update(variations)
        
        except Exception as e:
            self.logger.warning(f"Could not load Hindi terms: {e}")
        
        return terms
    
    def _compile_iast_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for IAST compliance checking."""
        iast_patterns = [
            # IAST diacritical marks
            re.compile(r'[āīūṛṝḷṃḥṅñṭḍṇśṣ]'),
            # Proper IAST combinations
            re.compile(r'(kṣ|jñ|tr|śr)'),
            # Avoid common non-IAST patterns
            re.compile(r'[^a-zA-Zāīūṛṝḷṁṃḥṅñṭḍṇśṣ\s\-\']', re.UNICODE)
        ]
        
        return iast_patterns
    
    def analyze_term_accuracy(
        self, 
        reference_text: str, 
        processed_text: str
    ) -> Dict[str, Any]:
        """Analyze accuracy of Sanskrit/Hindi terms."""
        ref_words = reference_text.split()
        proc_words = processed_text.split()
        
        # Identify Sanskrit and Hindi terms in reference
        ref_sanskrit_terms = [w for w in ref_words if self._is_sanskrit_term(w)]
        ref_hindi_terms = [w for w in ref_words if self._is_hindi_term(w)]
        
        # Check accuracy
        sanskrit_correct = 0
        hindi_correct = 0
        
        # Simple word-level matching (could be enhanced with alignment)
        for term in ref_sanskrit_terms:
            if term in proc_words:
                sanskrit_correct += 1
        
        for term in ref_hindi_terms:
            if term in proc_words:
                hindi_correct += 1
        
        # IAST compliance check
        iast_compliant = self._check_iast_compliance(processed_text)
        
        return {
            'sanskrit_terms_correct': sanskrit_correct,
            'sanskrit_terms_total': len(ref_sanskrit_terms),
            'hindi_terms_correct': hindi_correct,
            'hindi_terms_total': len(ref_hindi_terms),
            'iast_compliant': iast_compliant,
            'sanskrit_accuracy': sanskrit_correct / len(ref_sanskrit_terms) if ref_sanskrit_terms else 1.0,
            'hindi_accuracy': hindi_correct / len(ref_hindi_terms) if ref_hindi_terms else 1.0
        }
    
    def _is_sanskrit_term(self, word: str) -> bool:
        """Check if word is a Sanskrit term."""
        # Remove punctuation for checking
        clean_word = re.sub(r'[^\w\-\'āīūṛṝḷṁṃḥṅñṭḍṇśṣ]', '', word)
        return clean_word.lower() in self.sanskrit_terms
    
    def _is_hindi_term(self, word: str) -> bool:
        """Check if word is a Hindi term."""
        clean_word = re.sub(r'[^\w\-\']', '', word)
        return clean_word.lower() in self.hindi_terms
    
    def _check_iast_compliance(self, text: str) -> bool:
        """Check if text follows IAST transliteration standards."""
        # Check for presence of IAST diacritical marks where expected
        has_iast_marks = bool(self.iast_patterns[0].search(text))
        
        # Check for proper IAST combinations
        has_proper_combinations = bool(self.iast_patterns[1].search(text))
        
        # Check for non-IAST characters in Sanskrit context
        has_invalid_chars = bool(self.iast_patterns[2].search(text))
        
        # Basic IAST compliance: has marks, no invalid chars
        return has_iast_marks and not has_invalid_chars


class VerseIdentificationAnalyzer:
    """Analyzer for scripture verse identification accuracy."""
    
    def __init__(self, verses_lexicon_path: str):
        self.verses_lexicon_path = verses_lexicon_path
        self.canonical_verses = self._load_canonical_verses()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _load_canonical_verses(self) -> Dict[str, str]:
        """Load canonical verse texts."""
        verses = {}
        
        try:
            import yaml
            with open(self.verses_lexicon_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if 'verses' in data:
                verses = data['verses']
            
        except Exception as e:
            self.logger.warning(f"Could not load canonical verses: {e}")
        
        return verses
    
    def analyze_verse_identification(
        self, 
        reference_text: str, 
        processed_text: str
    ) -> Dict[str, Any]:
        """Analyze verse identification accuracy."""
        ref_verses = self._identify_verses(reference_text)
        proc_verses = self._identify_verses(processed_text)
        
        # Check if verses match
        verse_identified = False
        if ref_verses and proc_verses:
            # Check for overlap in identified verses
            ref_set = set(ref_verses)
            proc_set = set(proc_verses)
            verse_identified = len(ref_set.intersection(proc_set)) > 0
        elif not ref_verses and not proc_verses:
            # Both correctly identified no verses
            verse_identified = True
        
        return {
            'verse_identified': verse_identified,
            'reference_verses': ref_verses,
            'processed_verses': proc_verses,
            'verse_accuracy': 1.0 if verse_identified else 0.0
        }
    
    def _identify_verses(self, text: str) -> List[str]:
        """Identify verses in text by matching against canonical verses."""
        identified_verses = []
        
        # Simple substring matching (could be enhanced)
        for verse_id, canonical_text in self.canonical_verses.items():
            # Check if canonical text appears in the input text
            similarity = fuzz.partial_ratio(text.lower(), canonical_text.lower())
            
            if similarity > 80:  # Threshold for verse identification
                identified_verses.append(verse_id)
        
        return identified_verses


class GoldenDatasetValidator:
    """
    Main validator for measuring accuracy against golden dataset.
    
    Features:
    - Automated accuracy measurement against expert-verified content
    - Comprehensive quality metrics calculation
    - Detailed error analysis and reporting
    - Performance benchmarking
    - Quality trend tracking
    """
    
    def __init__(
        self, 
        golden_dataset_path: str,
        lexicon_paths: Dict[str, str] = None
    ):
        self.golden_dataset_path = Path(golden_dataset_path)
        self.lexicon_paths = lexicon_paths or {
            'sanskrit': 'data/lexicons/corrections.yaml',
            'hindi': 'data/lexicons/corrections.yaml',
            'verses': 'data/lexicons/verses.yaml'
        }
        
        # Initialize analyzers
        self.text_comparator = TextComparator()
        self.term_analyzer = TermAccuracyAnalyzer(self.lexicon_paths)
        self.verse_analyzer = VerseIdentificationAnalyzer(self.lexicon_paths['verses'])
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_processing_accuracy(
        self,
        processed_output_path: str,
        output_report_path: str = None
    ) -> ValidationMetrics:
        """
        Validate processing accuracy against golden dataset.
        
        Args:
            processed_output_path: Path to processed SRT files
            output_report_path: Optional path to save detailed report
            
        Returns:
            ValidationMetrics with comprehensive accuracy measurements
        """
        processed_path = Path(processed_output_path)
        start_time = datetime.utcnow()
        
        self.logger.info("Starting golden dataset validation...")
        
        # Find matching file pairs
        file_pairs = self._find_file_pairs(processed_path)
        
        if not file_pairs:
            raise ValueError("No matching file pairs found between golden dataset and processed output")
        
        # Validate each file pair
        segment_results = []
        for golden_file, processed_file in file_pairs:
            results = self._validate_file_pair(golden_file, processed_file)
            segment_results.extend(results)
        
        # Calculate aggregate metrics
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        metrics = self._calculate_aggregate_metrics(segment_results, processing_time)
        
        # Generate report if requested
        if output_report_path:
            self._generate_validation_report(metrics, segment_results, output_report_path)
        
        self.logger.info(
            f"Validation completed: {metrics.overall_accuracy:.1%} accuracy, "
            f"{metrics.word_error_rate:.1%} WER"
        )
        
        return metrics
    
    def _find_file_pairs(self, processed_path: Path) -> List[Tuple[Path, Path]]:
        """Find pairs of golden dataset and processed files."""
        file_pairs = []
        
        # Get all golden dataset files
        golden_files = list(self.golden_dataset_path.glob("**/*.srt"))
        
        for golden_file in golden_files:
            # Find corresponding processed file
            relative_path = golden_file.relative_to(self.golden_dataset_path)
            processed_file = processed_path / relative_path
            
            if processed_file.exists():
                file_pairs.append((golden_file, processed_file))
            else:
                self.logger.warning(f"No processed file found for {golden_file}")
        
        return file_pairs
    
    def _validate_file_pair(
        self, 
        golden_file: Path, 
        processed_file: Path
    ) -> List[SegmentValidationResult]:
        """Validate a pair of golden and processed SRT files."""
        try:
            # Load SRT files
            golden_srt = pysrt.open(str(golden_file), encoding='utf-8')
            processed_srt = pysrt.open(str(processed_file), encoding='utf-8')
            
            results = []
            
            # Compare segments
            for i, golden_segment in enumerate(golden_srt):
                if i < len(processed_srt):
                    processed_segment = processed_srt[i]
                    
                    result = self._validate_segment_pair(
                        f"{golden_file.stem}_{i}",
                        golden_segment.text,
                        processed_segment.text
                    )
                    results.append(result)
                else:
                    # Missing segment in processed file
                    result = SegmentValidationResult(
                        segment_id=f"{golden_file.stem}_{i}",
                        original_text="",
                        processed_text="",
                        golden_text=golden_segment.text,
                        accuracy_score=0.0,
                        word_error_rate=1.0,
                        character_error_rate=1.0,
                        sanskrit_terms_correct=0,
                        sanskrit_terms_total=0,
                        hindi_terms_correct=0,
                        hindi_terms_total=0,
                        iast_compliant=False,
                        verse_identified=False,
                        errors=[{"type": "missing_segment", "description": "Segment missing in processed file"}],
                        suggestions=["Check processing completeness"]
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error validating file pair {golden_file} vs {processed_file}: {e}")
            return []
    
    def _validate_segment_pair(
        self, 
        segment_id: str, 
        golden_text: str, 
        processed_text: str
    ) -> SegmentValidationResult:
        """Validate a pair of golden and processed segments."""
        
        # Calculate basic accuracy metrics
        wer = self.text_comparator.calculate_word_error_rate(golden_text, processed_text)
        cer = self.text_comparator.calculate_character_error_rate(golden_text, processed_text)
        accuracy_score = 1.0 - wer  # Simple accuracy based on WER
        
        # Analyze term accuracy
        term_analysis = self.term_analyzer.analyze_term_accuracy(golden_text, processed_text)
        
        # Analyze verse identification
        verse_analysis = self.verse_analyzer.analyze_verse_identification(golden_text, processed_text)
        
        # Get detailed differences for error analysis
        differences = self.text_comparator.get_detailed_diff(golden_text, processed_text)
        
        # Create errors list
        errors = []
        for diff in differences:
            errors.append({
                "type": diff['operation'],
                "position": diff['position'],
                "expected": diff['reference_text'],
                "actual": diff['hypothesis_text']
            })
        
        # Generate suggestions
        suggestions = self._generate_suggestions(wer, term_analysis, verse_analysis, errors)
        
        return SegmentValidationResult(
            segment_id=segment_id,
            original_text=processed_text,  # What was actually processed
            processed_text=processed_text,
            golden_text=golden_text,
            accuracy_score=accuracy_score,
            word_error_rate=wer,
            character_error_rate=cer,
            sanskrit_terms_correct=term_analysis['sanskrit_terms_correct'],
            sanskrit_terms_total=term_analysis['sanskrit_terms_total'],
            hindi_terms_correct=term_analysis['hindi_terms_correct'],
            hindi_terms_total=term_analysis['hindi_terms_total'],
            iast_compliant=term_analysis['iast_compliant'],
            verse_identified=verse_analysis['verse_identified'],
            errors=errors,
            suggestions=suggestions
        )
    
    def _generate_suggestions(
        self, 
        wer: float, 
        term_analysis: Dict[str, Any], 
        verse_analysis: Dict[str, Any],
        errors: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []
        
        if wer > 0.3:
            suggestions.append("High word error rate - review overall processing accuracy")
        
        if term_analysis['sanskrit_accuracy'] < 0.8:
            suggestions.append("Low Sanskrit term accuracy - update Sanskrit lexicon")
        
        if term_analysis['hindi_accuracy'] < 0.8:
            suggestions.append("Low Hindi term accuracy - update Hindi lexicon")
        
        if not term_analysis['iast_compliant']:
            suggestions.append("IAST compliance issues - review transliteration rules")
        
        if not verse_analysis['verse_identified']:
            suggestions.append("Verse identification failed - update verse database")
        
        # Error-specific suggestions
        substitution_errors = sum(1 for e in errors if e['type'] == 'replace')
        if substitution_errors > 3:
            suggestions.append("Many substitution errors - check lexicon coverage")
        
        return suggestions
    
    def _calculate_aggregate_metrics(
        self, 
        segment_results: List[SegmentValidationResult],
        processing_time: float
    ) -> ValidationMetrics:
        """Calculate aggregate metrics from segment results."""
        
        if not segment_results:
            raise ValueError("No segment results to calculate metrics from")
        
        # Basic counts
        total_segments = len(segment_results)
        processed_segments = sum(1 for r in segment_results if r.accuracy_score > 0)
        failed_segments = total_segments - processed_segments
        
        # Accuracy metrics
        accuracy_scores = [r.accuracy_score for r in segment_results]
        word_error_rates = [r.word_error_rate for r in segment_results]
        
        overall_accuracy = np.mean(accuracy_scores)
        word_error_rate = np.mean(word_error_rates)
        character_accuracy = 1.0 - np.mean([r.character_error_rate for r in segment_results])
        
        # Sanskrit/Hindi accuracy
        sanskrit_correct = sum(r.sanskrit_terms_correct for r in segment_results)
        sanskrit_total = sum(r.sanskrit_terms_total for r in segment_results)
        hindi_correct = sum(r.hindi_terms_correct for r in segment_results)
        hindi_total = sum(r.hindi_terms_total for r in segment_results)
        
        sanskrit_accuracy = sanskrit_correct / sanskrit_total if sanskrit_total > 0 else 1.0
        hindi_accuracy = hindi_correct / hindi_total if hindi_total > 0 else 1.0
        
        # IAST compliance
        iast_compliant_segments = sum(1 for r in segment_results if r.iast_compliant)
        iast_compliance = iast_compliant_segments / total_segments
        
        # Verse identification accuracy
        verse_identified_segments = sum(1 for r in segment_results if r.verse_identified)
        verse_accuracy = verse_identified_segments / total_segments
        
        # Error analysis
        substitution_errors = sum(len([e for e in r.errors if e['type'] == 'replace']) for r in segment_results)
        insertion_errors = sum(len([e for e in r.errors if e['type'] == 'insert']) for r in segment_results)
        deletion_errors = sum(len([e for e in r.errors if e['type'] == 'delete']) for r in segment_results)
        
        # Quality distribution
        high_quality = sum(1 for r in segment_results if r.accuracy_score >= 0.95)
        medium_quality = sum(1 for r in segment_results if 0.85 <= r.accuracy_score < 0.95)
        low_quality = sum(1 for r in segment_results if r.accuracy_score < 0.85)
        
        # Performance metrics
        segments_per_second = total_segments / processing_time if processing_time > 0 else 0
        
        return ValidationMetrics(
            overall_accuracy=overall_accuracy,
            word_error_rate=word_error_rate,
            sanskrit_accuracy=sanskrit_accuracy,
            hindi_accuracy=hindi_accuracy,
            iast_compliance=iast_compliance,
            verse_accuracy=verse_accuracy,
            character_accuracy=character_accuracy,
            total_segments=total_segments,
            processed_segments=processed_segments,
            failed_segments=failed_segments,
            substitution_errors=substitution_errors,
            insertion_errors=insertion_errors,
            deletion_errors=deletion_errors,
            processing_time=processing_time,
            segments_per_second=segments_per_second,
            high_quality_segments=high_quality,
            medium_quality_segments=medium_quality,
            low_quality_segments=low_quality
        )
    
    def _generate_validation_report(
        self,
        metrics: ValidationMetrics,
        segment_results: List[SegmentValidationResult],
        output_path: str
    ):
        """Generate detailed validation report."""
        report_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'summary': asdict(metrics),
            'detailed_results': [asdict(result) for result in segment_results[:100]]  # Limit for size
        }
        
        # Save JSON report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate visualizations if possible
        try:
            self._create_validation_visualizations(metrics, segment_results, output_path)
        except Exception as e:
            self.logger.warning(f"Could not create visualizations: {e}")
    
    def _create_validation_visualizations(
        self,
        metrics: ValidationMetrics,
        segment_results: List[SegmentValidationResult],
        base_path: str
    ):
        """Create validation result visualizations."""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Golden Dataset Validation Results', fontsize=16)
        
        # 1. Accuracy Distribution
        accuracy_scores = [r.accuracy_score for r in segment_results]
        axes[0, 0].hist(accuracy_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(metrics.overall_accuracy, color='red', linestyle='--', 
                          label=f'Mean: {metrics.overall_accuracy:.1%}')
        axes[0, 0].set_xlabel('Accuracy Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Accuracy Score Distribution')
        axes[0, 0].legend()
        
        # 2. Quality Categories
        categories = ['High Quality\n(≥95%)', 'Medium Quality\n(85-95%)', 'Low Quality\n(<85%)']
        counts = [metrics.high_quality_segments, metrics.medium_quality_segments, metrics.low_quality_segments]
        colors = ['green', 'orange', 'red']
        
        axes[0, 1].bar(categories, counts, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Number of Segments')
        axes[0, 1].set_title('Quality Distribution')
        
        # Add percentage labels
        total = sum(counts)
        for i, count in enumerate(counts):
            percentage = count / total * 100 if total > 0 else 0
            axes[0, 1].text(i, count + 0.5, f'{percentage:.1f}%', ha='center', va='bottom')
        
        # 3. Error Types
        error_types = ['Substitutions', 'Insertions', 'Deletions']
        error_counts = [metrics.substitution_errors, metrics.insertion_errors, metrics.deletion_errors]
        
        axes[1, 0].bar(error_types, error_counts, color='lightcoral', alpha=0.7)
        axes[1, 0].set_ylabel('Error Count')
        axes[1, 0].set_title('Error Type Distribution')
        
        # 4. Accuracy Metrics Comparison
        metric_names = ['Overall', 'Sanskrit', 'Hindi', 'IAST', 'Verse', 'Character']
        metric_values = [
            metrics.overall_accuracy,
            metrics.sanskrit_accuracy,
            metrics.hindi_accuracy,
            metrics.iast_compliance,
            metrics.verse_accuracy,
            metrics.character_accuracy
        ]
        
        bars = axes[1, 1].bar(metric_names, metric_values, color='lightgreen', alpha=0.7)
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Accuracy Metrics Comparison')
        axes[1, 1].set_ylim(0, 1.0)
        
        # Add percentage labels
        for bar, value in zip(bars, metric_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.1%}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_path = base_path.replace('.json', '_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()


# Convenience functions for easy usage

def validate_batch_processing(
    golden_dataset_path: str,
    processed_output_path: str,
    lexicon_paths: Dict[str, str] = None,
    report_path: str = None
) -> ValidationMetrics:
    """
    Convenience function to validate batch processing results.
    
    Args:
        golden_dataset_path: Path to golden dataset directory
        processed_output_path: Path to processed SRT files
        lexicon_paths: Optional paths to lexicon files
        report_path: Optional path to save detailed report
        
    Returns:
        ValidationMetrics with accuracy measurements
    """
    validator = GoldenDatasetValidator(golden_dataset_path, lexicon_paths)
    return validator.validate_processing_accuracy(processed_output_path, report_path)


def continuous_validation_monitoring(
    golden_dataset_path: str,
    processed_output_path: str,
    baseline_metrics_path: str = None,
    alert_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Continuous monitoring of processing quality with alerting.
    
    Args:
        golden_dataset_path: Path to golden dataset
        processed_output_path: Path to processed files
        baseline_metrics_path: Path to baseline metrics file
        alert_threshold: Threshold for quality degradation alerts
        
    Returns:
        Monitoring results with alerts if quality degraded
    """
    # Run validation
    current_metrics = validate_batch_processing(golden_dataset_path, processed_output_path)
    
    alerts = []
    
    # Compare with baseline if available
    if baseline_metrics_path and os.path.exists(baseline_metrics_path):
        try:
            with open(baseline_metrics_path, 'r') as f:
                baseline_data = json.load(f)
                baseline_accuracy = baseline_data.get('overall_accuracy', 0)
                
                # Check for degradation
                accuracy_drop = baseline_accuracy - current_metrics.overall_accuracy
                if accuracy_drop > alert_threshold:
                    alerts.append({
                        'type': 'quality_degradation',
                        'severity': 'high',
                        'message': f'Accuracy dropped by {accuracy_drop:.1%} from baseline',
                        'current_accuracy': current_metrics.overall_accuracy,
                        'baseline_accuracy': baseline_accuracy
                    })
        
        except Exception as e:
            alerts.append({
                'type': 'baseline_error',
                'severity': 'low',
                'message': f'Could not load baseline metrics: {e}'
            })
    
    return {
        'current_metrics': asdict(current_metrics),
        'alerts': alerts,
        'timestamp': datetime.utcnow().isoformat()
    }