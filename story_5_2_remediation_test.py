#!/usr/bin/env python3
"""
Story 5.2 Remediation Test
Diagnostic test for MCP client integration pipeline issues.
"""

import sys
import tempfile
import os
sys.path.insert(0, 'src')
import logging

# Configure logging to minimize noise
logging.basicConfig(level=logging.CRITICAL)
for logger_name in ['sanskrit_hindi_identifier', 'utils', 'post_processors', 'ner_module']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

def test_mcp_integration_pipeline():
    """Test MCP integration pipeline functionality."""
    print("=== Story 5.2 MCP Integration Pipeline Remediation Test ===")
    print()
    
    # Test 1: Advanced Text Normalizer with MCP
    print("Test 1: Advanced Text Normalizer with MCP Integration")
    try:
        from utils.advanced_text_normalizer import AdvancedTextNormalizer
        
        config = {'enable_mcp_processing': True, 'enable_fallback': True}
        normalizer = AdvancedTextNormalizer(config)
        
        # Test critical cases
        test_cases = [
            ('chapter two verse twenty five', 'Chapter 2 verse 25'),
            ('And one by one, he killed six of their children.', 'And one by one, he killed six of their children.'),
            ('Year two thousand five.', 'Year 2005.')
        ]
        
        all_passed = True
        for input_text, expected in test_cases:
            result = normalizer.convert_numbers_with_context(input_text)
            passed = result == expected
            status = "PASS" if passed else "FAIL"
            print(f"  {status}: {input_text} -> {result}")
            if not passed:
                all_passed = False
                print(f"         Expected: {expected}")
        
        print(f"  Overall: {'SUCCESS' if all_passed else 'FAILURE'}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        all_passed = False
    
    print()
    
    # Test 2: End-to-End SRT Processing
    print("Test 2: End-to-End SRT Processing Pipeline")
    try:
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        from pathlib import Path
        
        # Create test SRT
        test_content = """1
00:00:01,000 --> 00:00:05,000
today we study krishna in chapter two verse twenty five.

2
00:00:06,000 --> 00:00:10,000
and one by one, the students learned about dharma."""
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            input_path = f.name
        
        output_path = input_path.replace('.srt', '_processed.srt')
        
        # Process
        processor = SanskritPostProcessor()
        metrics = processor.process_srt_file(Path(input_path), Path(output_path))
        
        print(f"  Segments processed: {metrics.total_segments}")
        print(f"  Segments modified: {metrics.segments_modified}")
        print(f"  Processing time: {metrics.processing_time:.3f}s")
        
        # Check output
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                result = f.read()
            
            # Validate key transformations
            validations = [
                ('Chapter 2 verse 25' in result, 'Scriptural conversion'),
                ('Krishna' in result, 'Sanskrit capitalization'),
                ('Dharma' in result, 'Sanskrit capitalization'),
                ('one by one' in result.lower(), 'Idiomatic preservation')
            ]
            
            all_valid = True
            for passed, description in validations:
                status = "PASS" if passed else "FAIL"
                print(f"  {status}: {description}")
                if not passed:
                    all_valid = False
            
            print(f"  Overall: {'SUCCESS' if all_valid else 'FAILURE'}")
        else:
            print("  ERROR: Output file not created")
            all_valid = False
        
        # Cleanup
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
            
    except Exception as e:
        print(f"  ERROR: {e}")
        all_valid = False
    
    print()
    
    # Test 3: Professional Standards Compliance
    print("Test 3: Professional Standards Compliance")
    try:
        from utils.mcp_client import create_mcp_client
        
        client = create_mcp_client()
        compliance_report = client.get_professional_compliance_report()
        performance_stats = client.get_performance_stats()
        
        print(f"  Professional compliance report: {type(compliance_report)}")
        print(f"  Performance stats: {type(performance_stats)}")
        print("  Overall: SUCCESS - Professional standards framework operational")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Overall: FAILURE - Professional standards framework compromised")
    
    print()
    print("=== Story 5.2 Remediation Test Complete ===")

if __name__ == "__main__":
    test_mcp_integration_pipeline()