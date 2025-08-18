#!/usr/bin/env python3
"""
Novel QA Framework - Independent Validation System (Unicode-safe)
Designed to expose true functionality and avoid bias toward dev's tests
"""

import sys
import time
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import traceback

# Add src to path
sys.path.insert(0, 'src')

class NovelQAFramework:
    """Independent QA validation system that uses novel approaches to test true functionality"""
    
    def __init__(self):
        self.test_results = {}
        self.failure_logs = []
        self.critical_issues = []
        self.mcp_validation_failed = False
        
    def log_failure(self, test_name: str, details: str, is_critical: bool = False):
        """Log failure with detailed information"""
        failure = {
            'test': test_name,
            'details': details,
            'timestamp': time.time(),
            'critical': is_critical
        }
        self.failure_logs.append(failure)
        if is_critical:
            self.critical_issues.append(failure)
            
    def test_mcp_integration_reality(self) -> Dict[str, Any]:
        """Novel test: Check if MCP is actually being used, not just claimed"""
        print("=== Novel Test 1: MCP Integration Reality Check ===")
        
        result = {
            'test_name': 'MCP_Integration_Reality',
            'passed': False,
            'details': {},
            'critical_issues': []
        }
        
        try:
            # Test 1: Can we import MCP libraries?
            try:
                import mcp
                result['details']['mcp_import'] = 'SUCCESS'
                print("  MCP Import: SUCCESS")
            except ImportError as e:
                result['details']['mcp_import'] = f'FAILED: {e}'
                result['critical_issues'].append('MCP library not available')
                self.mcp_validation_failed = True
                print(f"  MCP Import: FAILED - {e}")
                
            # Test 2: Is MCP actually being used in processing?
            from utils.advanced_text_normalizer import AdvancedTextNormalizer
            
            config = {'enable_mcp_processing': True, 'enable_fallback': True}
            normalizer = AdvancedTextNormalizer(config)
            
            # Check if MCP client exists and is functional
            if hasattr(normalizer, 'mcp_client') and normalizer.mcp_client:
                result['details']['mcp_client_exists'] = 'SUCCESS'
                print("  MCP Client: EXISTS")
                
                # Test actual MCP processing vs fallback
                test_text = "Chapter two verse twenty five"
                processed_text = normalizer.convert_numbers_with_context(test_text)
                
                # Check result quality - MCP should handle this correctly
                if 'Chapter 2 verse 25' in processed_text:
                    result['details']['mcp_processing'] = 'SUCCESS'
                    print("  MCP Processing: SUCCESS")
                else:
                    result['details']['mcp_processing'] = f'FAILED: Expected conversion not found'
                    result['critical_issues'].append('MCP processing not working correctly')
                    self.mcp_validation_failed = True
                    print(f"  MCP Processing: FAILED - Expected 'Chapter 2 verse 25', got '{processed_text}'")
                    
            else:
                result['details']['mcp_client_exists'] = 'FAILED: No MCP client'
                result['critical_issues'].append('MCP client not initialized')
                self.mcp_validation_failed = True
                print("  MCP Client: FAILED - Not initialized")
                
        except Exception as e:
            result['details']['exception'] = str(e)
            result['critical_issues'].append(f'Test execution failed: {e}')
            self.log_failure('MCP_Integration_Reality', str(e), True)
            print(f"  ERROR: {e}")
            
        result['passed'] = len(result['critical_issues']) == 0
        print(f"  RESULT: {'PASS' if result['passed'] else 'FAIL'}")
        return result
        
    def test_critical_bug_fixes(self) -> Dict[str, Any]:
        """Novel test: Verify the critical bugs mentioned in QA review are actually fixed"""
        print("=== Novel Test 2: Critical Bug Fixes Verification ===")
        
        result = {
            'test_name': 'Critical_Bug_Fixes',
            'passed': False,
            'details': {},
            'bug_issues': []
        }
        
        try:
            from utils.advanced_text_normalizer import AdvancedTextNormalizer
            
            config = {'enable_mcp_processing': True, 'enable_fallback': True}
            normalizer = AdvancedTextNormalizer(config)
            
            # Critical test cases that were specifically mentioned as failing
            critical_test_cases = [
                {
                    'input': 'And one by one, he killed six of their children.',
                    'expected': 'And one by one, he killed six of their children.',
                    'bug_type': 'IDIOMATIC_PRESERVATION',
                    'description': 'Should preserve idiomatic expressions'
                },
                {
                    'input': 'Year two thousand five.',
                    'expected': 'Year 2005.',
                    'bug_type': 'YEAR_CONVERSION_FIX',
                    'description': 'Critical bug fix for year conversion'
                },
                {
                    'input': 'Chapter two verse twenty five.',
                    'expected': 'Chapter 2 verse 25.',
                    'bug_type': 'SCRIPTURAL_CONVERSION',
                    'description': 'Should convert scriptural references'
                }
            ]
            
            all_bugs_fixed = True
            for i, test_case in enumerate(critical_test_cases):
                actual_output = normalizer.convert_numbers_with_context(test_case['input'])
                
                bug_result = {
                    'test_id': i + 1,
                    'bug_type': test_case['bug_type'],
                    'input': test_case['input'],
                    'expected': test_case['expected'],
                    'actual': actual_output,
                    'fixed': actual_output == test_case['expected']
                }
                
                result['details'][f'bug_test_{i+1}'] = bug_result
                
                print(f"  Bug Test {i+1} ({test_case['bug_type']}):")
                print(f"    Input: {test_case['input']}")
                print(f"    Expected: {test_case['expected']}")
                print(f"    Actual: {actual_output}")
                print(f"    Fixed: {'YES' if bug_result['fixed'] else 'NO'}")
                
                if not bug_result['fixed']:
                    all_bugs_fixed = False
                    result['bug_issues'].append(f"{test_case['bug_type']}: Expected '{test_case['expected']}', got '{actual_output}'")
                    
            result['details']['all_critical_bugs_fixed'] = all_bugs_fixed
            result['passed'] = all_bugs_fixed
            
        except Exception as e:
            result['details']['exception'] = str(e)
            result['bug_issues'].append(f'Bug fix test failed: {e}')
            self.log_failure('Critical_Bug_Fixes', str(e), True)
            print(f"  ERROR: {e}")
            
        print(f"  RESULT: {'PASS' if result['passed'] else 'FAIL'}")
        return result
        
    def test_performance_reality(self) -> Dict[str, Any]:
        """Novel test: Test actual performance against claimed targets"""
        print("=== Novel Test 3: Performance Reality Check ===")
        
        result = {
            'test_name': 'Performance_Reality',
            'passed': False,
            'details': {},
            'performance_issues': []
        }
        
        try:
            from post_processors.sanskrit_post_processor import SanskritPostProcessor
            from utils.srt_parser import SRTSegment
            
            processor = SanskritPostProcessor()
            
            # Create realistic test segments
            test_segments = []
            for i in range(20):  # 20 segments for solid measurement
                segment = SRTSegment(
                    index=i+1,
                    start_time=float(i*5),
                    end_time=float(i*5 + 4),
                    text=f"Today we study yoga and dharma from ancient scriptures segment {i+1}.",
                    raw_text=f"Today we study yoga and dharma from ancient scriptures segment {i+1}."
                )
                test_segments.append(segment)
                
            print(f"  Testing with {len(test_segments)} segments...")
            
            # Measure performance multiple times for accuracy
            performance_runs = []
            for run in range(3):
                start_time = time.time()
                for segment in test_segments:
                    processor._process_srt_segment(segment, processor.metrics_collector.create_file_metrics('perf_test'))
                end_time = time.time()
                
                total_time = end_time - start_time
                segments_per_second = len(test_segments) / total_time if total_time > 0 else 0
                performance_runs.append(segments_per_second)
                
                print(f"    Run {run+1}: {segments_per_second:.2f} seg/sec")
                
            average_performance = sum(performance_runs) / len(performance_runs)
            min_performance = min(performance_runs)
            max_performance = max(performance_runs)
            
            result['details']['performance_stats'] = {
                'average_segments_per_second': average_performance,
                'min_segments_per_second': min_performance,
                'max_segments_per_second': max_performance,
                'target_performance': 10.0,
                'meets_target': average_performance >= 10.0,
                'performance_gap': 10.0 - average_performance if average_performance < 10.0 else 0
            }
            
            print(f"  Average Performance: {average_performance:.2f} seg/sec")
            print(f"  Target Performance: 10.0 seg/sec")
            print(f"  Meets Target: {'YES' if average_performance >= 10.0 else 'NO'}")
            
            if average_performance < 10.0:
                gap = 10.0 - average_performance
                result['performance_issues'].append(f'Performance {average_performance:.2f} seg/sec below 10.0 target (gap: {gap:.2f})')
                
            result['passed'] = average_performance >= 10.0
            
        except Exception as e:
            result['details']['exception'] = str(e)
            result['performance_issues'].append(f'Performance test failed: {e}')
            self.log_failure('Performance_Reality', str(e), True)
            print(f"  ERROR: {e}")
            
        print(f"  RESULT: {'PASS' if result['passed'] else 'FAIL'}")
        return result
        
    def test_end_to_end_functionality(self) -> Dict[str, Any]:
        """Novel test: Full pipeline with realistic data"""
        print("=== Novel Test 4: End-to-End Functionality ===")
        
        result = {
            'test_name': 'End_to_End_Functionality',
            'passed': False,
            'details': {},
            'functionality_issues': []
        }
        
        try:
            from post_processors.sanskrit_post_processor import SanskritPostProcessor
            
            # Create a realistic SRT content with known issues
            realistic_srt_content = '''1
00:00:01,000 --> 00:00:05,000
Um, today we will discuss chapter two verse twenty five from the sacred texts.

2
00:00:06,000 --> 00:00:10,000
In the year two thousand five, scholars began studying krishna more systematically.

3
00:00:11,000 --> 00:00:15,000
And one by one, they discovered new insights about dharma and karma yoga.'''
            
            # Test full pipeline
            with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
                f.write(realistic_srt_content)
                temp_input = f.name
                
            temp_output = temp_input.replace('.srt', '_processed.srt')
            
            processor = SanskritPostProcessor()
            
            # Process the file
            start_time = time.time()
            metrics = processor.process_srt_file(Path(temp_input), Path(temp_output))
            processing_time = time.time() - start_time
            
            print(f"  Processing completed in {processing_time:.2f} seconds")
            print(f"  Segments processed: {metrics.total_segments}")
            print(f"  Segments modified: {metrics.segments_modified}")
            
            # Validate the output
            if Path(temp_output).exists():
                with open(temp_output, 'r', encoding='utf-8') as f:
                    processed_content = f.read()
                    
                # Quality checks
                quality_results = []
                
                # Check 1: Filler words removed
                if 'Um,' not in processed_content:
                    quality_results.append('PASS: Filler words removed')
                else:
                    quality_results.append('FAIL: Filler words not removed')
                    result['functionality_issues'].append('Filler words not removed')
                    
                # Check 2: Numbers converted correctly
                if 'chapter 2 verse 25' in processed_content:
                    quality_results.append('PASS: Scriptural reference converted')
                else:
                    quality_results.append('FAIL: Scriptural reference not converted')
                    result['functionality_issues'].append('Scriptural reference not converted')
                    
                # Check 3: Year converted
                if '2005' in processed_content:
                    quality_results.append('PASS: Year converted')
                else:
                    quality_results.append('FAIL: Year not converted')
                    result['functionality_issues'].append('Year not converted')
                    
                # Check 4: Sanskrit names capitalized
                if 'Krishna' in processed_content:
                    quality_results.append('PASS: Sanskrit names capitalized')
                else:
                    quality_results.append('FAIL: Sanskrit names not capitalized')
                    result['functionality_issues'].append('Sanskrit names not capitalized')
                    
                # Check 5: Idiomatic preserved
                if 'one by one' in processed_content:
                    quality_results.append('PASS: Idiomatic expressions preserved')
                else:
                    quality_results.append('FAIL: Idiomatic expressions not preserved')
                    result['functionality_issues'].append('Idiomatic expressions not preserved')
                    
                result['details']['quality_checks'] = quality_results
                
                for check in quality_results:
                    print(f"    {check}")
                    
            else:
                result['functionality_issues'].append('CRITICAL: Output file not created')
                print("  CRITICAL: Output file not created")
                
            # Cleanup
            import os
            if os.path.exists(temp_input):
                os.unlink(temp_input)
            if os.path.exists(temp_output):
                os.unlink(temp_output)
                
            result['passed'] = len(result['functionality_issues']) == 0
            
        except Exception as e:
            result['details']['exception'] = str(e)
            result['functionality_issues'].append(f'End-to-end test failed: {e}')
            self.log_failure('End_to_End_Functionality', str(e), True)
            print(f"  ERROR: {e}")
            
        print(f"  RESULT: {'PASS' if result['passed'] else 'FAIL'}")
        return result
        
    def run_comprehensive_novel_validation(self) -> Dict[str, Any]:
        """Run all novel validation tests and provide comprehensive report"""
        print("NOVEL QA FRAMEWORK - INDEPENDENT VALIDATION")
        print("=" * 60)
        print("Purpose: Expose true functionality with unbiased testing")
        print("=" * 60)
        
        validation_results = {
            'framework_version': '1.0.0',
            'timestamp': time.time(),
            'overall_status': 'UNKNOWN',
            'test_results': {},
            'critical_failures': [],
            'performance_issues': [],
            'recommendations': []
        }
        
        # Run all novel tests
        tests = [
            self.test_mcp_integration_reality,
            self.test_critical_bug_fixes,
            self.test_performance_reality,
            self.test_end_to_end_functionality
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_func in tests:
            try:
                test_result = test_func()
                validation_results['test_results'][test_result['test_name']] = test_result
                
                if test_result['passed']:
                    passed_tests += 1
                else:
                    if test_result.get('critical_issues'):
                        validation_results['critical_failures'].extend(test_result['critical_issues'])
                    if test_result.get('performance_issues'):
                        validation_results['performance_issues'].extend(test_result['performance_issues'])
                    if test_result.get('bug_issues'):
                        validation_results['critical_failures'].extend(test_result['bug_issues'])
                    if test_result.get('functionality_issues'):
                        validation_results['critical_failures'].extend(test_result['functionality_issues'])
                        
            except Exception as e:
                error_result = {
                    'test_name': test_func.__name__,
                    'passed': False,
                    'error': str(e),
                    'critical_issues': [f'Test execution failed: {e}']
                }
                validation_results['test_results'][test_func.__name__] = error_result
                validation_results['critical_failures'].append(f'Test execution failed: {e}')
                
        # Determine overall status
        if self.mcp_validation_failed:
            validation_results['overall_status'] = 'CRITICAL_FAILURE - MCP Integration Issues'
        elif len(validation_results['critical_failures']) > 0:
            validation_results['overall_status'] = 'FAILED - Critical Issues Found'
        elif passed_tests == total_tests:
            validation_results['overall_status'] = 'PASSED - All Novel Tests Successful'
        else:
            validation_results['overall_status'] = 'PARTIAL - Some Tests Failed'
            
        # Generate recommendations based on findings
        if self.mcp_validation_failed:
            validation_results['recommendations'].append(
                "CRITICAL: Implement proper MCP integration - fallback mode is unacceptable"
            )
            
        if validation_results['performance_issues']:
            validation_results['recommendations'].append(
                "HIGH: Address performance issues to meet 10+ seg/sec target"
            )
            
        validation_results['recommendations'].extend([
            "Implement continuous novel testing in CI/CD pipeline",
            "Establish independent QA validation separate from dev tests",
            "Create performance regression monitoring",
            "Implement real-world scenario testing"
        ])
        
        return validation_results

def main():
    """Run the novel QA framework validation"""
    framework = NovelQAFramework()
    results = framework.run_comprehensive_novel_validation()
    
    # Generate detailed report
    print("\n" + "=" * 60)
    print("NOVEL QA VALIDATION REPORT")
    print("=" * 60)
    
    print(f"\nOVERALL STATUS: {results['overall_status']}")
    print(f"Tests Passed: {sum(1 for test in results['test_results'].values() if test['passed'])}/{len(results['test_results'])}")
    
    if results['critical_failures']:
        print(f"\nCRITICAL FAILURES ({len(results['critical_failures'])}):")
        for i, failure in enumerate(results['critical_failures'], 1):
            print(f"   {i}. {failure}")
            
    if results['performance_issues']:
        print(f"\nPERFORMACE ISSUES ({len(results['performance_issues'])}):")
        for i, issue in enumerate(results['performance_issues'], 1):
            print(f"   {i}. {issue}")
            
    print(f"\nRECOMMENDATIONS ({len(results['recommendations'])}):")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"   {i}. {rec}")
        
    # Save results to file
    results_file = f"novel_qa_validation_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    print(f"\nDetailed results saved to: {results_file}")
    
    # Return exit code based on results
    if results['overall_status'].startswith('CRITICAL') or results['overall_status'].startswith('FAILED'):
        return 1
    else:
        return 0

if __name__ == "__main__":
    exit(main())