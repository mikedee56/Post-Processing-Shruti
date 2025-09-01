"""
Lexicon Validation Tools

This module provides comprehensive validation tools for lexicon files,
ensuring data integrity, consistency, and adherence to IAST standards.
"""

import yaml
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from utils.logger_config import get_logger


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Represents a validation result."""
    level: ValidationLevel
    category: str
    message: str
    file_name: str = ""
    entry_term: str = ""
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class LexiconValidator:
    """Comprehensive lexicon validation toolkit."""
    
    def __init__(self, lexicon_dir: Path = None):
        """
        Initialize the lexicon validator.
        
        Args:
            lexicon_dir: Directory containing lexicon files
        """
        self.logger = get_logger(__name__)
        self.lexicon_dir = lexicon_dir or Path("data/lexicons")
        
        # Valid categories for entries
        self.valid_categories = {
            'deity', 'scripture', 'concept', 'practice', 'philosophy', 
            'character', 'teacher', 'reference', 'temporal', 
            'verse_reference', 'chapter_reference'
        }
        
        # IAST diacritical marks for validation
        self.iast_diacriticals = set('āīūṛṝḷḹēōṁṃḥṅñṭḍṇśṣ')
        
        # Common ASR error patterns
        self.common_asr_patterns = {
            r'\bkrsna\b': 'krishna',
            r'\bgyan\b': 'jnana', 
            r'\bdhyan\b': 'dhyana',
            r'\bsiv\b': 'shiv',
            r'\bvisnu\b': 'vishnu'
        }
    
    def validate_all_lexicons(self) -> List[ValidationResult]:
        """
        Validate all lexicon files comprehensively.
        
        Returns:
            List of validation results
        """
        results = []
        
        lexicon_files = [
            "corrections.yaml",
            "proper_nouns.yaml", 
            "phrases.yaml",
            "verses.yaml"
        ]
        
        self.logger.info(f"Starting validation of {len(lexicon_files)} lexicon files")
        
        for file_name in lexicon_files:
            file_path = self.lexicon_dir / file_name
            
            if not file_path.exists():
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    category="file_missing",
                    message=f"Lexicon file not found: {file_name}",
                    file_name=file_name
                ))
                continue
            
            # Validate individual file
            file_results = self.validate_lexicon_file(file_path)
            results.extend(file_results)
        
        # Cross-file validation
        cross_validation_results = self.validate_cross_file_consistency()
        results.extend(cross_validation_results)
        
        self.logger.info(f"Validation completed. Found {len(results)} issues")
        return results
    
    def validate_lexicon_file(self, file_path: Path) -> List[ValidationResult]:
        """
        Validate a single lexicon file.
        
        Args:
            file_path: Path to lexicon file
            
        Returns:
            List of validation results for this file
        """
        results = []
        file_name = file_path.name
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            results.append(ValidationResult(
                level=ValidationLevel.CRITICAL,
                category="file_parsing",
                message=f"Failed to parse YAML file: {e}",
                file_name=file_name
            ))
            return results
        
        # Check file structure
        if 'entries' not in data:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                category="file_structure",
                message="Missing 'entries' section in lexicon file",
                file_name=file_name
            ))
            return results
        
        if not isinstance(data['entries'], list):
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                category="file_structure", 
                message="'entries' section must be a list",
                file_name=file_name
            ))
            return results
        
        # Validate each entry
        seen_terms = set()
        seen_variations = set()
        
        for i, entry in enumerate(data['entries']):
            entry_results = self.validate_entry(entry, file_name, i)
            results.extend(entry_results)
            
            # Track duplicates
            if isinstance(entry, dict):
                term = entry.get('original_term', '').lower()
                if term:
                    if term in seen_terms:
                        results.append(ValidationResult(
                            level=ValidationLevel.WARNING,
                            category="duplicate_term",
                            message=f"Duplicate original_term: {term}",
                            file_name=file_name,
                            entry_term=term
                        ))
                    seen_terms.add(term)
                    
                    # Check variation duplicates
                    variations = entry.get('variations', [])
                    if isinstance(variations, list):
                        for variation in variations:
                            if isinstance(variation, str):
                                var_lower = variation.lower()
                                if var_lower in seen_variations:
                                    results.append(ValidationResult(
                                        level=ValidationLevel.WARNING,
                                        category="duplicate_variation",
                                        message=f"Duplicate variation: {variation}",
                                        file_name=file_name,
                                        entry_term=term
                                    ))
                                seen_variations.add(var_lower)
        
        return results
    
    def validate_entry(self, entry: Dict[str, Any], file_name: str, entry_index: int) -> List[ValidationResult]:
        """
        Validate a single lexicon entry.
        
        Args:
            entry: Entry dictionary
            file_name: Source file name
            entry_index: Index of entry in file
            
        Returns:
            List of validation results for this entry
        """
        results = []
        
        if not isinstance(entry, dict):
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                category="entry_structure",
                message=f"Entry {entry_index} is not a dictionary",
                file_name=file_name
            ))
            return results
        
        term = entry.get('original_term', '')
        
        # Required fields validation
        required_fields = ['original_term', 'variations', 'transliteration', 'is_proper_noun', 'category']
        for field in required_fields:
            if field not in entry:
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    category="missing_field",
                    message=f"Missing required field: {field}",
                    file_name=file_name,
                    entry_term=term
                ))
        
        # Field type validation
        if not isinstance(term, str) or not term.strip():
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                category="field_validation",
                message="original_term must be a non-empty string",
                file_name=file_name,
                entry_term=term
            ))
        
        # Variations validation
        variations = entry.get('variations', [])
        if not isinstance(variations, list):
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                category="field_validation",
                message="variations must be a list",
                file_name=file_name,
                entry_term=term
            ))
        elif len(variations) == 0:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                category="empty_variations",
                message="Entry has no variations - may reduce correction effectiveness",
                file_name=file_name,
                entry_term=term
            ))
        else:
            for var in variations:
                if not isinstance(var, str) or not var.strip():
                    results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        category="field_validation",
                        message=f"Invalid variation: {var}",
                        file_name=file_name,
                        entry_term=term
                    ))
        
        # Transliteration validation
        transliteration = entry.get('transliteration', '')
        if not isinstance(transliteration, str) or not transliteration.strip():
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                category="missing_transliteration",
                message="transliteration must be a non-empty string",
                file_name=file_name,
                entry_term=term
            ))
        else:
            # Check if Sanskrit terms have appropriate IAST diacriticals
            if self._should_have_diacriticals(term) and not self._has_diacriticals(transliteration):
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="iast_compliance",
                    message=f"Sanskrit term may need IAST diacriticals: {transliteration}",
                    file_name=file_name,
                    entry_term=term
                ))
        
        # Boolean field validation
        is_proper_noun = entry.get('is_proper_noun')
        if not isinstance(is_proper_noun, bool):
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                category="field_validation",
                message="is_proper_noun must be boolean",
                file_name=file_name,
                entry_term=term
            ))
        
        # Category validation
        category = entry.get('category', '')
        if category not in self.valid_categories:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                category="invalid_category",
                message=f"Invalid category '{category}'. Valid categories: {', '.join(sorted(self.valid_categories))}",
                file_name=file_name,
                entry_term=term
            ))
        
        # Confidence validation
        confidence = entry.get('confidence', 1.0)
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                category="field_validation",
                message=f"confidence must be a number between 0 and 1, got: {confidence}",
                file_name=file_name,
                entry_term=term
            ))
        
        # Check for common ASR error patterns
        for pattern, suggested in self.common_asr_patterns.items():
            if re.search(pattern, term, re.IGNORECASE):
                if suggested not in [v.lower() for v in variations]:
                    results.append(ValidationResult(
                        level=ValidationLevel.INFO,
                        category="asr_pattern_suggestion",
                        message=f"Consider adding '{suggested}' as variation for common ASR error pattern",
                        file_name=file_name,
                        entry_term=term
                    ))
        
        return results
    
    def validate_cross_file_consistency(self) -> List[ValidationResult]:
        """
        Validate consistency across multiple lexicon files.
        
        Returns:
            List of cross-file validation results
        """
        results = []
        
        try:
            # Load all lexicon entries
            all_entries = {}
            all_variations = {}
            
            for file_name in ["corrections.yaml", "proper_nouns.yaml", "phrases.yaml", "verses.yaml"]:
                file_path = self.lexicon_dir / file_name
                if not file_path.exists():
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if 'entries' in data:
                    for entry in data['entries']:
                        if isinstance(entry, dict):
                            term = entry.get('original_term', '').lower()
                            if term:
                                if term in all_entries:
                                    results.append(ValidationResult(
                                        level=ValidationLevel.WARNING,
                                        category="cross_file_duplicate",
                                        message=f"Term '{term}' appears in multiple files: {file_name} and {all_entries[term]['file']}",
                                        file_name=file_name,
                                        entry_term=term
                                    ))
                                else:
                                    all_entries[term] = {'file': file_name, 'entry': entry}
                                
                                # Check variations
                                variations = entry.get('variations', [])
                                if isinstance(variations, list):
                                    for var in variations:
                                        if isinstance(var, str):
                                            var_lower = var.lower()
                                            if var_lower in all_variations:
                                                results.append(ValidationResult(
                                                    level=ValidationLevel.WARNING,
                                                    category="cross_file_variation_conflict",
                                                    message=f"Variation '{var}' appears in multiple files",
                                                    file_name=file_name,
                                                    entry_term=term
                                                ))
                                            else:
                                                all_variations[var_lower] = {'file': file_name, 'term': term}
            
            self.logger.info(f"Cross-file validation completed. Checked {len(all_entries)} terms")
            
        except Exception as e:
            results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                category="cross_file_validation",
                message=f"Error during cross-file validation: {e}"
            ))
        
        return results
    
    def _should_have_diacriticals(self, term: str) -> bool:
        """Check if a term likely needs IAST diacriticals."""
        sanskrit_indicators = [
            'krishna', 'brahman', 'dharma', 'karma', 'yoga', 'vedanta',
            'upanishad', 'gita', 'arjuna', 'bhagavad', 'shiva', 'vishnu'
        ]
        return any(indicator in term.lower() for indicator in sanskrit_indicators)
    
    def _has_diacriticals(self, text: str) -> bool:
        """Check if text contains IAST diacritical marks."""
        return any(char in self.iast_diacriticals for char in text)
    
    def generate_validation_report(self, results: List[ValidationResult]) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            results: List of validation results
            
        Returns:
            Formatted validation report
        """
        if not results:
            return "✅ All lexicon files passed validation with no issues found."
        
        # Group results by level
        by_level = {}
        for result in results:
            level = result.level.value
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(result)
        
        # Group results by category
        by_category = {}
        for result in results:
            category = result.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)
        
        report = []
        report.append("LEXICON VALIDATION REPORT")
        report.append("=" * 50)
        report.append(f"Total Issues Found: {len(results)}")
        report.append("")
        
        # Summary by level
        report.append("ISSUES BY SEVERITY LEVEL:")
        for level in ['critical', 'error', 'warning', 'info']:
            count = len(by_level.get(level, []))
            if count > 0:
                report.append(f"  {level.upper()}: {count}")
        report.append("")
        
        # Summary by category  
        report.append("ISSUES BY CATEGORY:")
        for category, category_results in sorted(by_category.items()):
            report.append(f"  {category}: {len(category_results)}")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 30)
        
        for level in ['critical', 'error', 'warning', 'info']:
            level_results = by_level.get(level, [])
            if level_results:
                report.append(f"\n{level.upper()} ISSUES:")
                for result in level_results:
                    report.append(f"  • [{result.file_name}] {result.message}")
                    if result.entry_term:
                        report.append(f"    Term: {result.entry_term}")
        
        return "\n".join(report)
    
    def fix_category_issues(self) -> Dict[str, int]:
        """
        Automatically fix common category issues in lexicon files.
        
        Returns:
            Dictionary with counts of fixes applied
        """
        fixes_applied = {
            'category_corrections': 0,
            'files_modified': 0
        }
        
        # Category mapping for common fixes
        category_fixes = {
            'verse_reference': 'reference',
            'chapter_reference': 'reference'
        }
        
        lexicon_files = [
            "corrections.yaml",
            "proper_nouns.yaml", 
            "phrases.yaml",
            "verses.yaml"
        ]
        
        for file_name in lexicon_files:
            file_path = self.lexicon_dir / file_name
            if not file_path.exists():
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if 'entries' not in data:
                    continue
                
                file_modified = False
                
                for entry in data['entries']:
                    if isinstance(entry, dict):
                        category = entry.get('category', '')
                        if category in category_fixes:
                            entry['category'] = category_fixes[category]
                            fixes_applied['category_corrections'] += 1
                            file_modified = True
                            self.logger.info(f"Fixed category '{category}' -> '{category_fixes[category]}' in {file_name}")
                
                if file_modified:
                    # Create backup
                    backup_path = file_path.with_suffix('.yaml.backup')
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                    
                    # Write corrected file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)
                    
                    fixes_applied['files_modified'] += 1
                    self.logger.info(f"Fixed and updated {file_name} (backup created)")
                
            except Exception as e:
                self.logger.error(f"Error fixing categories in {file_name}: {e}")
        
        return fixes_applied


def main():
    """Command-line interface for lexicon validation."""
    import sys
    
    validator = LexiconValidator()
    
    print("Lexicon Validation Tool")
    print("=" * 30)
    
    # Run validation
    results = validator.validate_all_lexicons()
    
    # Generate and display report
    report = validator.generate_validation_report(results)
    print(report)
    
    # Check if we should fix category issues
    category_issues = [r for r in results if r.category == 'invalid_category']
    if category_issues:
        print(f"\nFound {len(category_issues)} category issues. Attempting to fix...")
        fixes = validator.fix_category_issues()
        print(f"Applied {fixes['category_corrections']} category corrections to {fixes['files_modified']} files")
        
        # Re-run validation to show improvements
        print("\nRe-running validation after fixes...")
        new_results = validator.validate_all_lexicons()
        new_report = validator.generate_validation_report(new_results)
        print(new_report)
    
    # Exit with appropriate code
    critical_or_error = [r for r in results if r.level in [ValidationLevel.CRITICAL, ValidationLevel.ERROR]]
    sys.exit(1 if critical_or_error else 0)


if __name__ == "__main__":
    main()