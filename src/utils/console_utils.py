"""
Console utilities for safe unicode output.

This module provides utilities to handle unicode output safely across different
console environments, particularly on Windows where encoding issues can occur.
"""

import sys
import unicodedata
from typing import Optional, Any


def safe_print(*args, encoding: Optional[str] = None, fallback: str = '?', **kwargs):
    """
    Print text safely, handling unicode encoding issues.
    
    Args:
        *args: Arguments to print
        encoding: Target encoding (auto-detected if None)
        fallback: Fallback character for unrepresentable characters
        **kwargs: Additional print() arguments
    """
    if encoding is None:
        # Auto-detect console encoding
        encoding = getattr(sys.stdout, 'encoding', 'utf-8')
        if encoding is None or encoding.lower() == 'ascii':
            encoding = 'utf-8'
    
    # Process each argument
    safe_args = []
    for arg in args:
        try:
            # Convert to string if not already
            text = str(arg)
            
            # Try to encode with target encoding
            text.encode(encoding)
            safe_args.append(text)
            
        except UnicodeEncodeError:
            # If encoding fails, transliterate or replace problematic characters
            safe_text = transliterate_unicode(text, fallback=fallback)
            safe_args.append(safe_text)
        except Exception:
            # Last resort: convert to basic ASCII representation
            safe_args.append(repr(arg))
    
    try:
        print(*safe_args, **kwargs)
    except UnicodeEncodeError:
        # Final fallback: print ASCII representation
        ascii_args = [ascii(arg) for arg in safe_args]
        print(*ascii_args, **kwargs)


def transliterate_unicode(text: str, fallback: str = '?') -> str:
    """
    Transliterate unicode characters to ASCII-safe equivalents.
    
    Args:
        text: Input text with potential unicode characters
        fallback: Character to use for untranslatable characters
        
    Returns:
        ASCII-safe text
    """
    try:
        # First, try unicode normalization (NFD - decomposed form)
        normalized = unicodedata.normalize('NFD', text)
        
        # Remove combining characters and keep base characters
        ascii_text = ''
        for char in normalized:
            if unicodedata.category(char) != 'Mn':  # Skip combining marks
                try:
                    # Try to encode as ASCII
                    char.encode('ascii')
                    ascii_text += char
                except UnicodeEncodeError:
                    # Replace with fallback or ASCII equivalent
                    ascii_equivalent = get_ascii_equivalent(char)
                    ascii_text += ascii_equivalent if ascii_equivalent else fallback
        
        return ascii_text
        
    except Exception:
        # If transliteration fails, return ASCII representation
        return text.encode('ascii', errors='replace').decode('ascii')


def get_ascii_equivalent(char: str) -> Optional[str]:
    """
    Get ASCII equivalent for common Unicode characters.
    
    Args:
        char: Unicode character
        
    Returns:
        ASCII equivalent or None if no mapping exists
    """
    # Common Sanskrit/Hindi transliteration mappings
    unicode_to_ascii = {
        'ā': 'a', 'ī': 'i', 'ū': 'u', 'ṛ': 'r', 'ṝ': 'r',
        'ḷ': 'l', 'ḹ': 'l', 'ē': 'e', 'ō': 'o',
        'ṃ': 'm', 'ḥ': 'h',
        'ṅ': 'n', 'ñ': 'n', 'ṇ': 'n',
        'ṭ': 't', 'ḍ': 'd', 'ṣ': 's',
        'ś': 's', 'ṝ': 'r',
        # Add more mappings as needed
    }
    
    return unicode_to_ascii.get(char)


def format_console_output(text: str, max_width: int = 80) -> str:
    """
    Format text for console output with proper width handling.
    
    Args:
        text: Input text
        max_width: Maximum line width
        
    Returns:
        Formatted text
    """
    try:
        # Handle long lines
        if len(text) <= max_width:
            return text
            
        # Simple word wrapping
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            if current_length + word_length + 1 <= max_width:  # +1 for space
                current_line.append(word)
                current_length += word_length + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
        
        if current_line:
            lines.append(' '.join(current_line))
            
        return '\n'.join(lines)
        
    except Exception:
        # Fallback to original text
        return text


def print_test_results(title: str, results: dict, show_details: bool = True):
    """
    Print test results in a formatted, console-safe way.
    
    Args:
        title: Test section title
        results: Dictionary of test results
        show_details: Whether to show detailed results
    """
    safe_print(f"\n=== {title} ===")
    
    for test_name, result in results.items():
        if isinstance(result, dict):
            status = result.get('status', 'UNKNOWN')
            message = result.get('message', '')
            
            # Format status with safe characters
            status_symbol = 'PASS' if status == 'pass' else 'FAIL' if status == 'fail' else 'WARN'
            
            safe_print(f"{status_symbol}: {test_name}")
            if show_details and message:
                safe_print(f"  {message}")
        else:
            # Simple result
            safe_print(f"INFO: {test_name} - {result}")
    
    safe_print()  # Empty line


def print_qa_summary(validation_results: dict):
    """
    Print QA validation summary in a safe, formatted way.
    
    Args:
        validation_results: Dictionary containing validation results
    """
    safe_print("=" * 60)
    safe_print("STORY 2.4.1 QA VALIDATION SUMMARY")
    safe_print("=" * 60)
    
    # Overall status
    overall_status = validation_results.get('overall_status', 'UNKNOWN')
    safe_print(f"Overall Status: {overall_status}")
    safe_print()
    
    # Acceptance criteria results
    if 'acceptance_criteria' in validation_results:
        safe_print("Acceptance Criteria Results:")
        for ac_id, result in validation_results['acceptance_criteria'].items():
            status = 'PASS' if result.get('passed', False) else 'FAIL'
            safe_print(f"  {ac_id}: {status} - {result.get('description', '')}")
    
    # Component validation
    if 'components' in validation_results:
        safe_print("\nComponent Validation:")
        for component, status in validation_results['components'].items():
            safe_print(f"  {component}: {status}")
    
    # Issues found
    if 'issues' in validation_results and validation_results['issues']:
        safe_print("\nIssues Found:")
        for issue in validation_results['issues']:
            safe_print(f"  - {issue}")
    
    safe_print("=" * 60)