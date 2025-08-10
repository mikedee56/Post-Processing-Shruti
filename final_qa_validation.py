#!/usr/bin/env python3
"""
Final QA Validation Framework - Phase 4 Implementation
Comprehensive regression testing and format validation for all corrected SRT files
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
from qa_quality_validation_rules import SRTQualityValidator

class FinalQAValidator:
    """Final validation for QA corrected SRT files"""
    
    def __init__(self):
        self.validator = SRTQualityValidator()
        
    def validate_srt_format(self, content: str) -> List[str]:
        """Validate SRT file format integrity"""
        errors = []
        lines = content.strip().split('\n')
        
        if not lines:
            errors.append("Empty file")
            return errors
            
        i = 0
        subtitle_count = 0
        
        while i < len(lines):
            # Check subtitle number
            if not lines[i].strip().isdigit():
                if lines[i].strip():  # Skip empty lines
                    errors.append(f"Line {i+1}: Expected subtitle number, got '{lines[i]}'")
                i += 1
                continue
                
            subtitle_count += 1
            expected_num = subtitle_count
            actual_num = int(lines[i].strip())
            
            if actual_num != expected_num:
                errors.append(f"Line {i+1}: Subtitle numbering error. Expected {expected_num}, got {actual_num}")
            
            i += 1
            
            # Check timestamp format
            if i >= len(lines):
                errors.append(f"Subtitle {subtitle_count}: Missing timestamp")
                break
                
            timestamp_pattern = r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}'
            if not re.match(timestamp_pattern, lines[i]):
                errors.append(f"Line {i+1}: Invalid timestamp format: '{lines[i]}'")
            
            i += 1
            
            # Check subtitle text (at least one line of text)
            text_lines = 0
            while i < len(lines) and lines[i].strip():
                text_lines += 1
                i += 1
                
            if text_lines == 0:
                errors.append(f"Subtitle {subtitle_count}: Missing subtitle text")
            
            # Skip empty line after subtitle
            if i < len(lines) and not lines[i].strip():
                i += 1
                
        return errors
        
    def validate_academic_standards(self, content: str) -> List[str]:
        """Validate academic writing standards"""
        issues = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            if self._is_subtitle_content(line):
                # Check for unprofessional patterns
                if re.search(r'\b(um|uh|er|ah)\b', line, re.IGNORECASE):
                    issues.append(f"Line {line_num}: Contains filler words")
                    
                # Check for proper sentence capitalization
                sentences = re.split(r'[.!?]+\s+', line)
                for sentence in sentences:
                    if sentence.strip() and not sentence.strip()[0].isupper():
                        issues.append(f"Line {line_num}: Sentence should start with capital letter")
                        
                # Check for double spaces
                if '  ' in line:
                    issues.append(f"Line {line_num}: Contains double spaces")
                    
        return issues
        
    def validate_spiritual_content(self, content: str) -> List[str]:
        """Validate proper handling of spiritual/Sanskrit content"""
        issues = []
        lines = content.split('\n')
        
        # Sanskrit terms that should be properly formatted
        sanskrit_terms = [
            'krishna', 'rama', 'shiva', 'vishnu', 'brahma', 'atman', 'dharma', 
            'karma', 'yoga', 'vedas', 'upanishads', 'gita', 'ramayana', 'mahabharata'
        ]
        
        for line_num, line in enumerate(lines, 1):
            if self._is_subtitle_content(line):
                # Check for respectful capitalization of proper nouns
                for term in sanskrit_terms:
                    if re.search(rf'\b{term}\b', line, re.IGNORECASE):
                        if not re.search(rf'\b{term.title()}\b', line):
                            # Allow for legitimate lowercase usage in compound words
                            if not re.search(rf'\w{term}\b|\b{term}\w', line, re.IGNORECASE):
                                issues.append(f"Line {line_num}: Sanskrit term '{term}' should be capitalized")
                                
        return issues
        
    def _is_subtitle_content(self, line: str) -> bool:
        """Check if line contains subtitle content (not number or timestamp)"""
        if not line.strip():
            return False
        if line.strip().isdigit():
            return False
        if re.match(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', line):
            return False
        return True
        
    def generate_final_report(self, results: Dict) -> str:
        """Generate comprehensive final QA report"""
        total_files = len(results)
        total_format_errors = sum(len(result['format_errors']) for result in results.values())
        total_academic_issues = sum(len(result['academic_issues']) for result in results.values())
        total_spiritual_issues = sum(len(result['spiritual_issues']) for result in results.values())
        total_quality_issues = sum(len(result['quality_issues']) for result in results.values())
        
        report = f"""
FINAL QA VALIDATION REPORT
==========================

EXECUTIVE SUMMARY:
- Files Processed: {total_files}
- Format Errors: {total_format_errors}
- Academic Issues: {total_academic_issues}  
- Spiritual Content Issues: {total_spiritual_issues}
- Remaining Quality Issues: {total_quality_issues}

DETAILED RESULTS:
"""
        
        for filename, result in results.items():
            format_count = len(result['format_errors'])
            academic_count = len(result['academic_issues'])
            spiritual_count = len(result['spiritual_issues'])
            quality_count = len(result['quality_issues'])
            
            status = "PASS" if (format_count + academic_count + spiritual_count + quality_count) == 0 else "ISSUES FOUND"
            
            report += f"""
FILE: {filename} - {status}
Format Errors: {format_count}
Academic Issues: {academic_count}
Spiritual Issues: {spiritual_count}
Quality Issues: {quality_count}
"""
            
            if format_count > 0:
                report += "Format Errors:\n"
                for error in result['format_errors'][:5]:  # Show first 5
                    report += f"  - {error}\n"
                if format_count > 5:
                    report += f"  - ... and {format_count - 5} more\n"
                    
            if academic_count > 0:
                report += "Academic Issues:\n" 
                for issue in result['academic_issues'][:3]:  # Show first 3
                    report += f"  - {issue}\n"
                if academic_count > 3:
                    report += f"  - ... and {academic_count - 3} more\n"
                    
        # Overall assessment
        if total_format_errors + total_academic_issues + total_spiritual_issues + total_quality_issues == 0:
            report += "\nOVERALL ASSESSMENT: ALL FILES PASS QUALITY STANDARDS"
            report += "\nReady for production deployment."
        else:
            report += f"\nOVERALL ASSESSMENT: {total_format_errors + total_academic_issues + total_spiritual_issues + total_quality_issues} ISSUES REQUIRE ATTENTION"
            
        return report

def run_final_validation(input_dir: str = "data/processed_srts") -> Dict:
    """Run comprehensive final validation on all QA_CORRECTED files"""
    
    print("RUNNING FINAL QA VALIDATION")
    print("=" * 40)
    
    validator = FinalQAValidator()
    input_path = Path(input_dir)
    results = {}
    
    for qa_file in input_path.glob("*_QA_CORRECTED.srt"):
        print(f"Validating: {qa_file.name}")
        
        with open(qa_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Run all validation checks
        format_errors = validator.validate_srt_format(content)
        academic_issues = validator.validate_academic_standards(content)
        spiritual_issues = validator.validate_spiritual_content(content)
        quality_issues = validator.validator.validate_srt_content(content)
        
        results[qa_file.name] = {
            'format_errors': format_errors,
            'academic_issues': academic_issues,
            'spiritual_issues': spiritual_issues,
            'quality_issues': quality_issues
        }
        
        total_issues = len(format_errors) + len(academic_issues) + len(spiritual_issues) + len(quality_issues)
        print(f"  Total issues: {total_issues}")
        
    # Generate and save report
    report = validator.generate_final_report(results)
    
    with open("final_qa_validation_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"\nFinal report saved to: final_qa_validation_report.txt")
    return results

if __name__ == "__main__":
    run_final_validation()