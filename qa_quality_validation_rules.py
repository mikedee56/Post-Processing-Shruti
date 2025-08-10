#!/usr/bin/env python3
"""
QA Quality Validation Framework for SRT Post-Processing
Systematic validation rules to ensure professional academic output quality
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class QualityIssue:
    """Represents a quality issue found in SRT content"""
    line_number: int
    issue_type: str
    description: str
    original_text: str
    suggested_fix: str
    severity: str  # 'critical', 'major', 'minor'

class SRTQualityValidator:
    """Validates and corrects quality issues in SRT files"""
    
    def __init__(self):
        self.punctuation_spacing_patterns = [
            (r'\.([A-Z])', r'. \1', 'Missing space after period'),
            (r'\?([A-Z])', r'? \1', 'Missing space after question mark'),
            (r'!([A-Z])', r'! \1', 'Missing space after exclamation mark'),
            (r',([A-Z])', r', \1', 'Missing space after comma'),
        ]
        
        self.number_word_patterns = [
            (r'\b1\s+(who|that|behind)', r'one \1', 'Number should be word in formal text'),
            (r'\b2nd\b', 'second', 'Ordinal number formatting'),
            (r'\b3rd\b', 'third', 'Ordinal number formatting'),
            (r'\b1st\b', 'first', 'Ordinal number formatting'),
            (r'\b4th\b', 'fourth', 'Ordinal number formatting'),
            (r'\b5th\b', 'fifth', 'Ordinal number formatting'),
            (r'\b6th\b', 'sixth', 'Ordinal number formatting'),
            (r'\b7th\b', 'seventh', 'Ordinal number formatting'),
            (r'\b8th\b', 'eighth', 'Ordinal number formatting'),
            (r'\b9th\b', 'ninth', 'Ordinal number formatting'),
            (r'\b10th\b', 'tenth', 'Ordinal number formatting'),
        ]
        
        self.grammatical_patterns = [
            (r'How does he smiled', 'How does he smile', 'Incorrect verb tense'),
            (r'himSelf', 'himself', 'Incorrect capitalization'),
            (r'no 1 can', 'no one can', 'Number should be word'),
        ]
        
        self.academic_formatting_patterns = [
            # Sanskrit term consistency
            (r'([a-z])([A-Z])', r'\1 \2', 'Missing space in compound terms'),
        ]
        
        # CRITICAL ENHANCEMENT: Dash consistency and academic standards
        self.dash_consistency_patterns = [
            # Standardize to em-dashes for formal academic writing
            (r'\s+-\s+', '—', 'Replace hyphen with em-dash for formal text'),
            (r'\s+--\s+', '—', 'Replace double hyphen with em-dash'),
            (r'(\w)-(\w)', r'\1—\2', 'Use em-dash for word separation'),
        ]
        
        # Enhanced editorial quality patterns
        self.editorial_quality_patterns = [
            # Awkward phrasing corrections
            (r'\bthat that\b', 'that', 'Remove duplicate "that"'),
            (r'\bthe the\b', 'the', 'Remove duplicate "the"'),
            (r'\band and\b', 'and', 'Remove duplicate "and"'),
            (r'\bof of\b', 'of', 'Remove duplicate "of"'),
            
            # Spiritual/academic tone improvements
            (r'\bthat\s+that\b', 'that', 'Remove redundant repetition'),
            (r'\bvery very\b', 'very', 'Remove redundant intensifier'),
            
            # Sanskrit diacritical consistency
            (r'\bKrishna\b', 'Kṛṣṇa', 'Use proper IAST transliteration'),
            (r'\bVishnu\b', 'Viṣṇu', 'Use proper IAST transliteration'),
            (r'\bShiva\b', 'Śiva', 'Use proper IAST transliteration'),
        ]
        
        # Academic polish validation patterns
        self.academic_polish_patterns = [
            # Capitalization validation
            (r'^\s*[a-z]', 'Sentence should start with capital letter'),
            (r'\.(\s+)[a-z]', 'Sentence should start with capital letter after period'),
            (r'\?(\s+)[a-z]', 'Sentence should start with capital letter after question mark'),
            (r'!(\s+)[a-z]', 'Sentence should start with capital letter after exclamation'),
            (r'—(\s*)[a-z]', 'Should capitalize after em-dash'),
            
            # Sanskrit term consistency
            (r'\bkrishna\b', 'Consider capitalizing Krishna when referring to deity'),
            (r'\brama\b(?!\s+(cooking|bed))', 'Consider capitalizing Rama when referring to deity'),
            (r'\bvishnu\b', 'Consider capitalizing Vishnu when referring to deity'),
            (r'\bshiva\b', 'Consider capitalizing Shiva when referring to deity'),
            
            # Format consistency
            (r'  +', 'Multiple spaces should be collapsed to single space'),
            (r'\s+[.,:;!?]', 'Remove space before punctuation'),
        ]
    
    def validate_srt_content(self, content: str) -> List[QualityIssue]:
        """Validate SRT content and return list of quality issues"""
        issues = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip timestamp lines and empty lines
            if self._is_timestamp_line(line) or not line.strip():
                continue
                
            # Check punctuation spacing issues
            for pattern, replacement, description in self.punctuation_spacing_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    issues.append(QualityIssue(
                        line_number=line_num,
                        issue_type='punctuation_spacing',
                        description=description,
                        original_text=line,
                        suggested_fix=re.sub(pattern, replacement, line),
                        severity='major'
                    ))
            
            # Check number/word formatting
            for pattern, replacement, description in self.number_word_patterns:
                if re.search(pattern, line):
                    issues.append(QualityIssue(
                        line_number=line_num,
                        issue_type='number_formatting',
                        description=description,
                        original_text=line,
                        suggested_fix=re.sub(pattern, replacement, line),
                        severity='minor'
                    ))
            
            # Check grammatical issues
            for pattern, replacement, description in self.grammatical_patterns:
                if re.search(pattern, line):
                    issues.append(QualityIssue(
                        line_number=line_num,
                        issue_type='grammar',
                        description=description,
                        original_text=line,
                        suggested_fix=re.sub(pattern, replacement, line),
                        severity='critical'
                    ))
            
            # CRITICAL ENHANCEMENT: Check dash consistency
            for pattern, replacement, description in self.dash_consistency_patterns:
                if re.search(pattern, line):
                    issues.append(QualityIssue(
                        line_number=line_num,
                        issue_type='dash_consistency',
                        description=description,
                        original_text=line,
                        suggested_fix=re.sub(pattern, replacement, line),
                        severity='minor'
                    ))
            
            # Check editorial quality
            for pattern, replacement, description in self.editorial_quality_patterns:
                if re.search(pattern, line):
                    issues.append(QualityIssue(
                        line_number=line_num,
                        issue_type='editorial_quality',
                        description=description,
                        original_text=line,
                        suggested_fix=re.sub(pattern, replacement, line),
                        severity='minor'
                    ))
            
            # Check academic polish requirements
            for pattern, description in self.academic_polish_patterns:
                if re.search(pattern, line):
                    issues.append(QualityIssue(
                        line_number=line_num,
                        issue_type='academic_polish',
                        description=description,
                        original_text=line,
                        suggested_fix='[Polish enhancement needed]',
                        severity='minor'
                    ))
                        
        return issues
    
    def apply_corrections(self, content: str) -> str:
        """Apply systematic corrections to SRT content"""
        corrected_content = content
        
        # Apply punctuation spacing corrections
        for pattern, replacement, _ in self.punctuation_spacing_patterns:
            corrected_content = re.sub(pattern, replacement, corrected_content)
        
        # Apply number/word corrections
        for pattern, replacement, _ in self.number_word_patterns:
            corrected_content = re.sub(pattern, replacement, corrected_content)
        
        # Apply grammatical corrections
        for pattern, replacement, _ in self.grammatical_patterns:
            corrected_content = re.sub(pattern, replacement, corrected_content)
        
        # CRITICAL ENHANCEMENT: Apply dash consistency corrections
        for pattern, replacement, _ in self.dash_consistency_patterns:
            corrected_content = re.sub(pattern, replacement, corrected_content)
        
        # Apply editorial quality corrections
        for pattern, replacement, _ in self.editorial_quality_patterns:
            corrected_content = re.sub(pattern, replacement, corrected_content)
            
        return corrected_content
    
    def _is_timestamp_line(self, line: str) -> bool:
        """Check if line contains SRT timestamp"""
        timestamp_pattern = r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}'
        return bool(re.match(timestamp_pattern, line))
    
    def generate_quality_report(self, issues: List[QualityIssue]) -> str:
        """Generate a comprehensive quality report"""
        if not issues:
            return "SUCCESS: No quality issues found. Content meets professional standards."
        
        critical_issues = [i for i in issues if i.severity == 'critical']
        major_issues = [i for i in issues if i.severity == 'major']
        minor_issues = [i for i in issues if i.severity == 'minor']
        
        report = f"""
QA QUALITY REPORT
=================

SUMMARY:
- Total Issues: {len(issues)}
- Critical: {len(critical_issues)}
- Major: {len(major_issues)}
- Minor: {len(minor_issues)}

CRITICAL ISSUES ({len(critical_issues)}):
"""
        for issue in critical_issues:
            report += f"Line {issue.line_number}: {issue.description}\n"
            report += f"  Original: {issue.original_text.strip()}\n"
            report += f"  Fixed: {issue.suggested_fix.strip()}\n\n"
        
        report += f"\nMAJOR ISSUES ({len(major_issues)}):\n"
        for issue in major_issues:
            report += f"Line {issue.line_number}: {issue.description}\n"
        
        report += f"\nMINOR ISSUES ({len(minor_issues)}):\n"
        for issue in minor_issues:
            report += f"Line {issue.line_number}: {issue.description}\n"
            
        return report

def validate_srt_file(file_path: str) -> Tuple[List[QualityIssue], str]:
    """Validate a single SRT file and return issues and report"""
    validator = SRTQualityValidator()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        issues = validator.validate_srt_content(content)
        report = validator.generate_quality_report(issues)
        
        return issues, report
        
    except Exception as e:
        return [], f"Error validating file {file_path}: {str(e)}"

def apply_corrections_to_file(input_file: str, output_file: str) -> bool:
    """Apply systematic corrections to an SRT file"""
    validator = SRTQualityValidator()
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        corrected_content = validator.apply_corrections(content)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(corrected_content)
        
        return True
        
    except Exception as e:
        print(f"Error correcting file {input_file}: {str(e)}")
        return False

def batch_correct_files(input_dir: str, output_suffix: str = "_QA_CORRECTED") -> Dict[str, bool]:
    """Apply corrections to all emergency_safe.srt files in a directory"""
    import os
    from pathlib import Path
    
    input_path = Path(input_dir)
    results = {}
    
    # Find all emergency_safe.srt files
    for file_path in input_path.glob("*_emergency_safe.srt"):
        input_file = str(file_path)
        output_file = str(file_path.with_name(file_path.stem.replace("_emergency_safe", output_suffix) + ".srt"))
        
        print(f"Processing: {file_path.name}")
        success = apply_corrections_to_file(input_file, output_file)
        results[file_path.name] = success
        
        if success:
            print(f"Created: {Path(output_file).name}")
        else:
            print(f"Failed: {file_path.name}")
    
    return results

def generate_batch_report(input_dir: str) -> str:
    """Generate quality reports for all emergency_safe.srt files"""
    import os
    from pathlib import Path
    
    input_path = Path(input_dir)
    batch_report = "BATCH QUALITY VALIDATION REPORT\n" + "=" * 50 + "\n\n"
    
    total_files = 0
    total_issues = 0
    
    for file_path in input_path.glob("*_emergency_safe.srt"):
        total_files += 1
        issues, report = validate_srt_file(str(file_path))
        total_issues += len(issues)
        
        batch_report += f"FILE: {file_path.name}\n"
        batch_report += "-" * 40 + "\n"
        batch_report += report + "\n\n"
    
    summary = f"BATCH SUMMARY:\n"
    summary += f"- Files Processed: {total_files}\n"
    summary += f"- Total Issues Found: {total_issues}\n"
    summary += f"- Average Issues per File: {total_issues/total_files if total_files > 0 else 0:.1f}\n\n"
    
    return summary + batch_report

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch-correct":
        # Batch correction mode
        input_dir = "data/processed_srts"
        print("Starting batch correction of all emergency_safe.srt files...")
        results = batch_correct_files(input_dir)
        
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        print(f"\nBATCH CORRECTION SUMMARY:")
        print(f"Successful: {successful}/{total}")
        print(f"Failed: {total - successful}/{total}")
        
        # Generate batch quality report
        print(f"\nGenerating quality report...")
        report = generate_batch_report(input_dir)
        
        with open("batch_quality_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to: batch_quality_report.txt")
        
    else:
        # Single file validation mode (existing functionality)
        test_file = "data/processed_srts/SrimadBhagavadGita112913_emergency_safe.srt"
        if len(sys.argv) > 1:
            test_file = sys.argv[1]
            
        print(f"Validating: {test_file}")
        issues, report = validate_srt_file(test_file)
        print(report)