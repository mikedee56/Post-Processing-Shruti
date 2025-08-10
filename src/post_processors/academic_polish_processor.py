#!/usr/bin/env python3
"""
Academic Polish Enhancement System for SRT Post-Processing
Elevates professional output to academic-grade excellence with precise capitalization, 
Sanskrit term standardization, and format consistency
"""

import re
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import logging

@dataclass
class PolishIssue:
    """Represents a polish enhancement opportunity"""
    line_number: int
    issue_type: str
    description: str
    original_text: str
    suggested_fix: str
    priority: str  # 'critical', 'major', 'minor'

class AcademicPolishProcessor:
    """Transforms professional SRT content to academic excellence standards"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Academic capitalization patterns - sentence start rules
        self.capitalization_patterns = [
            # Sentence start after period, question mark, exclamation
            (r'\.(\s+)([a-z])', lambda m: f'.{m.group(1)}{m.group(2).upper()}', 'Capitalize first word of sentence'),
            (r'\?(\s+)([a-z])', lambda m: f'?{m.group(1)}{m.group(2).upper()}', 'Capitalize first word of sentence'),
            (r'!(\s+)([a-z])', lambda m: f'!{m.group(1)}{m.group(2).upper()}', 'Capitalize first word of sentence'),
            
            # Line start capitalization (for subtitle lines)
            (r'^([a-z])', lambda m: m.group(1).upper(), 'Capitalize first word of subtitle line'),
            
            # After em-dash (continuing thought patterns)
            (r'—(\s*)([a-z])', lambda m: f'—{m.group(1)}{m.group(2).upper()}', 'Capitalize after em-dash'),
            
            # After colon in formal contexts
            (r':(\s+)([a-z])', lambda m: f':{m.group(1)}{m.group(2).upper()}', 'Capitalize after colon'),
        ]
        
        # Sanskrit/Hindi term standardization - context-aware proper nouns
        self.sanskrit_term_patterns = [
            # Major deities - always capitalize
            (r'\bkrishna\b', 'Krishna', 'Capitalize deity name'),
            (r'\bkṛṣṇa\b', 'Kṛṣṇa', 'Capitalize deity name'),
            (r'\brama\b(?!\s+(cooking|bed))', 'Rama', 'Capitalize deity name (context-aware)'),
            (r'\bhanuman\b', 'Hanuman', 'Capitalize deity name'),
            (r'\bshiva\b', 'Shiva', 'Capitalize deity name'),
            (r'\bśiva\b', 'Śiva', 'Capitalize deity name'),
            (r'\bvishnu\b', 'Vishnu', 'Capitalize deity name'),
            (r'\bviṣṇu\b', 'Viṣṇu', 'Capitalize deity name'),
            (r'\bbrahma\b(?!\s+knowledge)', 'Brahma', 'Capitalize deity name (context-aware)'),
            
            # Sacred texts - always capitalize
            (r'\bbhagavad\s+gita\b', 'Bhagavad Gita', 'Capitalize sacred text'),
            (r'\bramayana\b', 'Ramayana', 'Capitalize sacred text'),
            (r'\brāmāyaṇa\b', 'Rāmāyaṇa', 'Capitalize sacred text'),
            (r'\bmahabharata\b', 'Mahabharata', 'Capitalize sacred text'),
            (r'\bmahābhārata\b', 'Mahābhārata', 'Capitalize sacred text'),
            (r'\bupanishads?\b', 'Upanishads', 'Capitalize sacred text'),
            (r'\bupaniṣads?\b', 'Upaniṣads', 'Capitalize sacred text'),
            (r'\bvedas?\b', 'Vedas', 'Capitalize sacred text'),
            
            # Philosophical concepts - context-sensitive
            (r'\bdharma\b(?=\s+[A-Z])', 'Dharma', 'Capitalize when used as proper concept'),
            (r'\bkarma\b(?=\s+[A-Z])', 'Karma', 'Capitalize when used as proper concept'),
            (r'\byoga\b(?=\s+(Vedanta|Sutra))', 'Yoga', 'Capitalize when part of proper title'),
            (r'\bvedanta\b', 'Vedanta', 'Capitalize philosophical system'),
            (r'\bsankhya\b', 'Sankhya', 'Capitalize philosophical system'),
            (r'\bsāṅkhya\b', 'Sāṅkhya', 'Capitalize philosophical system'),
            (r'\badvaita\b', 'Advaita', 'Capitalize philosophical doctrine'),
            
            # Specific terms in titles or formal contexts
            (r'\bguru\b(?=\s+[A-Z])', 'Guru', 'Capitalize in formal title context'),
            (r'\bswami\b(?=\s+[A-Z])', 'Swami', 'Capitalize in formal title context'),
            (r'\bācārya\b(?=\s+[A-Z])', 'Ācārya', 'Capitalize in formal title context'),
        ]
        
        # Format consistency patterns
        self.format_consistency_patterns = [
            # Standardize spacing around punctuation
            (r'\.{2,}', '...', 'Standardize ellipsis'),
            (r'\s+\.', '.', 'Remove space before period'),
            (r'\s+,', ',', 'Remove space before comma'),
            (r'\s+;', ';', 'Remove space before semicolon'),
            (r'\s+:', ':', 'Remove space before colon'),
            
            # Standardize quotation marks
            (r'"([^"]*)"', r'"\1"', 'Standardize quotation marks'),
            (r"'([^']*)'", r"'\1'", 'Standardize single quotes'),
            
            # Fix multiple spaces
            (r'  +', ' ', 'Collapse multiple spaces'),
            
            # Standardize dash usage
            (r'\s*-\s*', '—', 'Use em-dash for interruptions'),
            (r'\s*--\s*', '—', 'Use em-dash instead of double hyphen'),
        ]
        
        # Sanskrit terms that should remain lowercase in common usage
        self.common_usage_terms = {
            'karma', 'dharma', 'yoga', 'guru', 'mantra', 'chakra', 'prana',
            'ahimsa', 'moksha', 'samadhi', 'satsang', 'tapas'
        }
        
        # Academic proper nouns that should always be capitalized
        self.academic_proper_nouns = {
            'God', 'Divine', 'Absolute', 'Supreme', 'Self', 'Truth', 'Reality',
            'Consciousness', 'Brahman', 'Ātman', 'Paramātmā'
        }
    
    def polish_srt_content(self, content: str) -> Tuple[str, List[PolishIssue]]:
        """Apply academic polish enhancements to SRT content"""
        polished_content = content
        issues = []
        lines = content.split('\n')
        
        for line_num, original_line in enumerate(lines, 1):
            # Skip timestamp lines and empty lines
            if self._is_timestamp_line(original_line) or not original_line.strip():
                continue
            
            line = original_line
            line_issues = []
            
            # Apply capitalization fixes
            for pattern, replacement, description in self.capitalization_patterns:
                if isinstance(replacement, str):
                    # Simple string replacement
                    if re.search(pattern, line, re.IGNORECASE):
                        new_line = re.sub(pattern, replacement, line, flags=re.IGNORECASE)
                        if new_line != line:
                            line_issues.append(PolishIssue(
                                line_number=line_num,
                                issue_type='capitalization',
                                description=description,
                                original_text=line,
                                suggested_fix=new_line,
                                priority='major'
                            ))
                            line = new_line
                else:
                    # Function-based replacement
                    matches = list(re.finditer(pattern, line))
                    if matches:
                        new_line = re.sub(pattern, replacement, line)
                        if new_line != line:
                            line_issues.append(PolishIssue(
                                line_number=line_num,
                                issue_type='capitalization',
                                description=description,
                                original_text=line,
                                suggested_fix=new_line,
                                priority='major'
                            ))
                            line = new_line
            
            # Apply Sanskrit term standardization
            for pattern, replacement, description in self.sanskrit_term_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    new_line = re.sub(pattern, replacement, line, flags=re.IGNORECASE)
                    if new_line != line:
                        line_issues.append(PolishIssue(
                            line_number=line_num,
                            issue_type='sanskrit_standardization',
                            description=description,
                            original_text=line,
                            suggested_fix=new_line,
                            priority='major'
                        ))
                        line = new_line
            
            # Apply format consistency fixes
            for pattern, replacement, description in self.format_consistency_patterns:
                if re.search(pattern, line):
                    new_line = re.sub(pattern, replacement, line)
                    if new_line != line:
                        line_issues.append(PolishIssue(
                            line_number=line_num,
                            issue_type='format_consistency',
                            description=description,
                            original_text=line,
                            suggested_fix=new_line,
                            priority='minor'
                        ))
                        line = new_line
            
            # Replace the line in the content if changes were made
            if line != original_line:
                lines[line_num - 1] = line
                issues.extend(line_issues)
        
        polished_content = '\n'.join(lines)
        return polished_content, issues
    
    def fix_subtitle_numbering(self, content: str) -> Tuple[str, List[PolishIssue]]:
        """Fix subtitle numbering sequence issues"""
        lines = content.split('\n')
        issues = []
        expected_number = 1
        
        for line_num, line in enumerate(lines):
            # Check if line is a subtitle number
            if re.match(r'^\d+$', line.strip()):
                current_number = int(line.strip())
                if current_number != expected_number:
                    issues.append(PolishIssue(
                        line_number=line_num + 1,
                        issue_type='subtitle_numbering',
                        description=f'Subtitle numbering error. Expected {expected_number}, got {current_number}',
                        original_text=line,
                        suggested_fix=str(expected_number),
                        priority='critical'
                    ))
                    lines[line_num] = str(expected_number)
                expected_number += 1
        
        return '\n'.join(lines), issues
    
    def fix_missing_subtitle_text(self, content: str) -> Tuple[str, List[PolishIssue]]:
        """Fix missing subtitle text issues"""
        lines = content.split('\n')
        issues = []
        
        i = 0
        while i < len(lines):
            # Check if we have a subtitle number followed by timestamp but no text
            if (i < len(lines) - 2 and 
                re.match(r'^\d+$', lines[i].strip()) and  # subtitle number
                self._is_timestamp_line(lines[i + 1]) and  # timestamp
                (i + 2 >= len(lines) or  # end of file
                 lines[i + 2].strip() == '' or  # empty line
                 re.match(r'^\d+$', lines[i + 2].strip()))):  # next subtitle number
                
                subtitle_num = lines[i].strip()
                issues.append(PolishIssue(
                    line_number=i + 1,
                    issue_type='missing_subtitle_text',
                    description=f'Subtitle {subtitle_num}: Missing subtitle text',
                    original_text=f'Subtitle {subtitle_num}',
                    suggested_fix='[Text content needed]',
                    priority='critical'
                ))
            i += 1
        
        return '\n'.join(lines), issues
    
    def validate_srt_format_compliance(self, content: str) -> List[PolishIssue]:
        """Validate SRT format compliance"""
        lines = content.split('\n')
        issues = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check subtitle structure: number -> timestamp -> text -> blank line
            if re.match(r'^\d+$', line):
                subtitle_num = line
                
                # Check if followed by timestamp
                if i + 1 >= len(lines) or not self._is_timestamp_line(lines[i + 1]):
                    issues.append(PolishIssue(
                        line_number=i + 2,
                        issue_type='srt_format_compliance',
                        description=f'Subtitle {subtitle_num}: Missing or invalid timestamp',
                        original_text=lines[i + 1] if i + 1 < len(lines) else '[End of file]',
                        suggested_fix='[Valid timestamp required]',
                        priority='critical'
                    ))
                
                # Check timestamp format
                elif i + 1 < len(lines):
                    timestamp = lines[i + 1].strip()
                    if not re.match(r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$', timestamp):
                        issues.append(PolishIssue(
                            line_number=i + 2,
                            issue_type='srt_format_compliance',
                            description=f'Subtitle {subtitle_num}: Invalid timestamp format',
                            original_text=timestamp,
                            suggested_fix='HH:MM:SS,mmm --> HH:MM:SS,mmm',
                            priority='critical'
                        ))
            i += 1
        
        return issues
    
    def validate_spiritual_respectfulness(self, content: str) -> List[PolishIssue]:
        """Validate that spiritual content maintains respectful tone"""
        issues = []
        lines = content.split('\n')
        
        # Disrespectful patterns to flag for review
        disrespectful_patterns = [
            (r'\b(god|krishna|rama|shiva|vishnu)\s+is\s+(just|only|merely)', 'Potentially diminishing language toward divinity'),
            (r'\b(stupid|dumb|idiotic)\s+(hindu|sanskrit|yoga|meditation)', 'Potentially offensive language toward tradition'),
            (r'\b(primitive|backward|outdated)\s+(belief|practice|teaching)', 'Potentially dismissive language'),
        ]
        
        for line_num, line in enumerate(lines, 1):
            if self._is_timestamp_line(line) or not line.strip():
                continue
            
            for pattern, description in disrespectful_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(PolishIssue(
                        line_number=line_num,
                        issue_type='spiritual_respectfulness',
                        description=description,
                        original_text=line,
                        suggested_fix='[Human review required]',
                        priority='critical'
                    ))
        
        return issues
    
    def _is_timestamp_line(self, line: str) -> bool:
        """Check if line contains SRT timestamp"""
        timestamp_pattern = r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}'
        return bool(re.match(timestamp_pattern, line.strip()))
    
    def _is_subtitle_number_line(self, line: str) -> bool:
        """Check if line is a subtitle number"""
        return bool(re.match(r'^\d+$', line.strip()))
    
    def generate_polish_report(self, issues: List[PolishIssue]) -> str:
        """Generate comprehensive polish enhancement report"""
        if not issues:
            return "SUCCESS: Content meets academic excellence standards. No polish enhancements needed."
        
        critical_issues = [i for i in issues if i.priority == 'critical']
        major_issues = [i for i in issues if i.priority == 'major']
        minor_issues = [i for i in issues if i.priority == 'minor']
        
        # Categorize by type
        capitalization = [i for i in issues if i.issue_type == 'capitalization']
        sanskrit_std = [i for i in issues if i.issue_type == 'sanskrit_standardization']
        format_consistency = [i for i in issues if i.issue_type == 'format_consistency']
        subtitle_numbering = [i for i in issues if i.issue_type == 'subtitle_numbering']
        missing_text = [i for i in issues if i.issue_type == 'missing_subtitle_text']
        srt_compliance = [i for i in issues if i.issue_type == 'srt_format_compliance']
        spiritual_respect = [i for i in issues if i.issue_type == 'spiritual_respectfulness']
        
        report = f"""
ACADEMIC POLISH ENHANCEMENT REPORT
==================================

SUMMARY:
- Total Enhancements Applied: {len(issues)}
- Critical: {len(critical_issues)}
- Major: {len(major_issues)}
- Minor: {len(minor_issues)}

ENHANCEMENT CATEGORIES:
- Academic Capitalization: {len(capitalization)}
- Sanskrit Term Standardization: {len(sanskrit_std)}
- Format Consistency: {len(format_consistency)}
- Subtitle Numbering: {len(subtitle_numbering)}
- Missing Subtitle Text: {len(missing_text)}
- SRT Format Compliance: {len(srt_compliance)}
- Spiritual Respectfulness: {len(spiritual_respect)}

"""
        
        if critical_issues:
            report += f"CRITICAL ISSUES REQUIRING REVIEW ({len(critical_issues)}):\n"
            for issue in critical_issues:
                report += f"Line {issue.line_number}: {issue.description}\n"
                report += f"  Original: {issue.original_text.strip()}\n"
                report += f"  Suggested: {issue.suggested_fix.strip()}\n\n"
        
        if major_issues:
            report += f"MAJOR ENHANCEMENTS APPLIED ({len(major_issues)}):\n"
            for issue in major_issues[:10]:  # Show first 10
                report += f"Line {issue.line_number}: {issue.description}\n"
            if len(major_issues) > 10:
                report += f"... and {len(major_issues) - 10} more\n"
            report += "\n"
        
        if minor_issues:
            report += f"MINOR POLISH IMPROVEMENTS ({len(minor_issues)}):\n"
            report += f"Applied {len(minor_issues)} format consistency and style improvements.\n\n"
        
        return report

def polish_srt_file(input_file: str, output_file: str) -> Tuple[bool, str]:
    """Apply academic polish enhancements to a single SRT file"""
    processor = AcademicPolishProcessor()
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply polish enhancements
        polished_content, polish_issues = processor.polish_srt_content(content)
        
        # Fix subtitle numbering
        polished_content, numbering_issues = processor.fix_subtitle_numbering(polished_content)
        
        # Fix missing subtitle text
        polished_content, missing_text_issues = processor.fix_missing_subtitle_text(polished_content)
        
        # Validate SRT format compliance
        format_compliance_issues = processor.validate_srt_format_compliance(polished_content)
        
        # Validate spiritual respectfulness
        respect_issues = processor.validate_spiritual_respectfulness(polished_content)
        
        # Combine all issues
        all_issues = polish_issues + numbering_issues + missing_text_issues + format_compliance_issues + respect_issues
        
        # Write polished content
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(polished_content)
        
        # Generate report
        report = processor.generate_polish_report(all_issues)
        
        return True, report
        
    except Exception as e:
        return False, f"Error polishing file {input_file}: {str(e)}"

def batch_polish_files(input_dir: str, input_suffix: str = "_QA_CORRECTED", output_suffix: str = "_POLISHED") -> Dict[str, Tuple[bool, str]]:
    """Apply academic polish to all QA_CORRECTED files in a directory"""
    input_path = Path(input_dir)
    results = {}
    
    # Find all QA_CORRECTED.srt files
    for file_path in input_path.glob(f"*{input_suffix}.srt"):
        input_file = str(file_path)
        output_file = str(file_path.with_name(file_path.stem.replace(input_suffix, output_suffix) + ".srt"))
        
        print(f"Polishing: {file_path.name}")
        success, report = polish_srt_file(input_file, output_file)
        results[file_path.name] = (success, report)
        
        if success:
            print(f"Created: {Path(output_file).name}")
        else:
            print(f"Failed: {file_path.name}")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch-polish":
        # Batch polish mode
        input_dir = "data/processed_srts"
        print("Starting batch academic polish of all QA_CORRECTED.srt files...")
        results = batch_polish_files(input_dir)
        
        successful = sum(1 for success, _ in results.values() if success)
        total = len(results)
        
        print(f"\nBATCH POLISH SUMMARY:")
        print(f"Successful: {successful}/{total}")
        print(f"Failed: {total - successful}/{total}")
        
        # Save reports
        for filename, (success, report) in results.items():
            if success:
                report_filename = f"polish_report_{filename.replace('.srt', '.txt')}"
                with open(report_filename, "w", encoding="utf-8") as f:
                    f.write(report)
        
    else:
        # Single file polish mode
        input_file = sys.argv[1] if len(sys.argv) > 1 else "data/processed_srts/Sunday103011SBS35_QA_CORRECTED.srt"
        output_file = input_file.replace("_QA_CORRECTED", "_POLISHED")
        
        print(f"Polishing: {input_file}")
        success, report = polish_srt_file(input_file, output_file)
        
        if success:
            print(f"Created: {output_file}")
            print("\nPOLISH REPORT:")
            print(report)
        else:
            print(f"Failed: {report}")