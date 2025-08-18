#!/usr/bin/env python3
"""
Novel QA Framework - Independent Validation System
Designed to expose true functionality and avoid bias toward dev's tests
"""

import sys
import time
import tempfile
import subprocess
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
            except ImportError as e:
                result['details']['mcp_import'] = f'FAILED: {e}'
                result['critical_issues'].append('MCP library not available')
                self.mcp_validation_failed = True
                
            # Test 2: Is MCP actually being used in processing?
            from utils.advanced_text_normalizer import AdvancedTextNormalizer
            
            config = {'enable_mcp_processing': True, 'enable_fallback': True}
            normalizer = AdvancedTextNormalizer(config)
            
            # Check if MCP client exists and is functional
            if hasattr(normalizer, 'mcp_client') and normalizer.mcp_client:
                result['details']['mcp_client_exists'] = 'SUCCESS'
                
                # Test actual MCP processing vs fallback
                test_text = "Chapter two verse twenty five"
                
                # Monitor for fallback warnings
                import logging
                import io
                log_capture = io.StringIO()
                handler = logging.StreamHandler(log_capture)
                logger = logging.getLogger()
                logger.addHandler(handler)
                
                processed_text = normalizer.convert_numbers_with_context(test_text)
                
                log_output = log_capture.getvalue()
                if 'fallback' in log_output.lower() or 'failed' in log_output.lower():
                    result['details']['mcp_processing'] = f'FAILED: Using fallback - {log_output}'
                    result['critical_issues'].append('MCP processing falling back to rules')
                    self.mcp_validation_failed = True
                else:
                    result['details']['mcp_processing'] = 'SUCCESS'
                    
                logger.removeHandler(handler)
            else:
                result['details']['mcp_client_exists'] = 'FAILED: No MCP client'
                result['critical_issues'].append('MCP client not initialized')
                self.mcp_validation_failed = True
                
            # Test 3: Performance comparison - MCP vs Fallback
            if not self.mcp_validation_failed:
                # Test with MCP enabled
                start_time = time.time()
                for _ in range(10):
                    normalizer.convert_numbers_with_context("Today we study chapter two verse twenty five")
                mcp_time = time.time() - start_time
                
                # Test with fallback only
                config_fallback = {'enable_mcp_processing': False, 'enable_fallback': True}
                normalizer_fallback = AdvancedTextNormalizer(config_fallback)
                
                start_time = time.time()
                for _ in range(10):
                    normalizer_fallback.convert_numbers_with_context("Today we study chapter two verse twenty five")
                fallback_time = time.time() - start_time
                
                result['details']['performance_comparison'] = {
                    'mcp_time': mcp_time,
                    'fallback_time': fallback_time,
                    'mcp_faster': mcp_time < fallback_time
                }
                
        except Exception as e:
            result['details']['exception'] = str(e)
            result['critical_issues'].append(f'Test execution failed: {e}')
            self.log_failure('MCP_Integration_Reality', str(e), True)
            
        result['passed'] = len(result['critical_issues']) == 0
        return result
        
    def test_performance_under_stress(self) -> Dict[str, Any]:
        """Novel test: Stress test with realistic data loads"""
        print("=== Novel Test 2: Performance Under Realistic Stress ===")
        
        result = {
            'test_name': 'Performance_Stress_Test',
            'passed': False,
            'details': {},
            'performance_issues': []
        }
        
        try:
            from post_processors.sanskrit_post_processor import SanskritPostProcessor
            from utils.srt_parser import SRTSegment
            
            processor = SanskritPostProcessor()
            
            # Create realistic test data - varied, complex segments
            complex_segments = []
            realistic_texts = [
                "Um, today we will discuss, uh, chapter two verse twenty five from the Bhagavad Gita.",
                "The, the teachings of Krishna, I mean, Krsna, are profound and, actually, quite complex.",
                "In the year two thousand five, scholars began studying these ancient texts more systematically.",
                "Patanjali and Shankaracharya, well, they were great teachers who, um, who guided many seekers.",
                "The Upanishads contain, let me see, approximately one hundred eight principal verses.",
                "Yoga Sutras, or should I say, the foundational text of yoga, was written, uh, centuries ago.",
                "Arjuna asked Krishna about dharma, karma, and the nature of, actually, the eternal soul.",
                "In Rishikesh and Varanasi, pilgrims study these sacred texts with, um, with great devotion.",
                "The concept of moksha, liberation, is central to, well, to Vedantic philosophy.",
                "Swami Vivekananda brought these teachings to the West in, let's see, eighteen ninety three."
            ]
            
            # Create 100 segments with realistic complexity
            for i in range(100):
                text = realistic_texts[i % len(realistic_texts)]
                segment = SRTSegment(
                    index=i+1,
                    start_time=float(i*5),
                    end_time=float(i*5 + 4),
                    text=text,
                    raw_text=text
                )
                complex_segments.append(segment)
                
            # Test 1: Processing speed with complex data
            start_time = time.time()
            processed_count = 0
            
            for segment in complex_segments:
                try:
                    processed_segment = processor._process_srt_segment(
                        segment, 
                        processor.metrics_collector.create_file_metrics('stress_test')
                    )
                    processed_count += 1
                except Exception as e:
                    result['performance_issues'].append(f'Segment {segment.index} failed: {e}')
                    
            total_time = time.time() - start_time
            segments_per_second = processed_count / total_time if total_time > 0 else 0
            
            result['details']['stress_test_performance'] = {
                'total_segments': len(complex_segments),
                'processed_successfully': processed_count,
                'total_time': total_time,
                'segments_per_second': segments_per_second,
                'target_performance': 10.0,
                'meets_target': segments_per_second >= 10.0
            }
            
            # Test 2: Memory usage monitoring
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process larger batch
            large_batch = complex_segments * 5  # 500 segments
            for segment in large_batch[:50]:  # Process subset to avoid timeout
                processor._process_srt_segment(
                    segment, 
                    processor.metrics_collector.create_file_metrics('memory_test')
                )
                
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            result['details']['memory_usage'] = {
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'memory_increase_mb': memory_increase,
                'acceptable_increase': memory_increase < 100  # Less than 100MB increase
            }
            
            # Overall assessment
            performance_good = segments_per_second >= 10.0
            memory_good = memory_increase < 100
            no_critical_failures = len(result['performance_issues']) == 0
            
            result['passed'] = performance_good and memory_good and no_critical_failures
            
            if not performance_good:
                self.log_failure('Performance_Stress_Test', 
                               f'Performance {segments_per_second:.2f} seg/sec below 10.0 target', True)
                               
        except Exception as e:
            result['details']['exception'] = str(e)
            self.log_failure('Performance_Stress_Test', str(e), True)
            
        return result
        
    def test_functional_correctness_novel(self) -> Dict[str, Any]:
        """Novel test: Test functionality with unexpected inputs and edge cases"""
        print("=== Novel Test 3: Functional Correctness with Edge Cases ===")
        
        result = {
            'test_name': 'Functional_Correctness_Novel',
            'passed': False,
            'details': {},
            'correctness_issues': []
        }
        
        try:
            from utils.advanced_text_normalizer import AdvancedTextNormalizer
            
            config = {'enable_mcp_processing': True, 'enable_fallback': True}
            normalizer = AdvancedTextNormalizer(config)
            
            # Novel test cases that dev probably didn't test
            novel_test_cases = [
                # Edge case: Mixed contexts
                {
                    'input': "In chapter two verse twenty five, we see twenty five different interpretations over twenty five years.",
                    'context': 'mixed_contexts',
                    'expected_behavior': 'Preserve idiomatic but convert scriptural'
                },
                # Edge case: Ambiguous numbers
                {
                    'input': "Two thousand five hundred and two thousand five are different years.",
                    'context': 'ambiguous_years',
                    'expected_behavior': 'Handle both year formats correctly'
                },
                # Edge case: Sanskrit with numbers
                {
                    'input': "Krishna spoke one hundred eight verses to Arjuna in chapter eighteen.",
                    'context': 'sanskrit_with_numbers',
                    'expected_behavior': 'Capitalize Sanskrit names, convert numbers appropriately'
                },
                # Edge case: Nested corrections
                {
                    'input': "Um, actually, let me correct that - rather, it's about, well, dharma yoga practice.",
                    'context': 'nested_corrections',
                    'expected_behavior': 'Handle multiple correction patterns'
                },
                # Edge case: Unicode and transliteration
                {
                    'input': "The term kṛṣṇa or krishna appears two hundred times in the text.",
                    'context': 'unicode_mixed',
                    'expected_behavior': 'Handle both IAST and romanized forms'
                }
            ]
            
            test_results = []
            for i, test_case in enumerate(novel_test_cases):
                try:
                    output = normalizer.convert_numbers_with_context(test_case['input'])
                    
                    # Analyze the output for correctness
                    analysis = {
                        'test_id': i + 1,
                        'input': test_case['input'],
                        'output': output,
                        'context': test_case['context'],
                        'expected_behavior': test_case['expected_behavior'],
                        'issues': []
                    }
                    
                    # Check for specific issues based on context
                    if test_case['context'] == 'mixed_contexts':
                        # Should preserve "twenty five different" but convert "chapter two verse twenty five"
                        if 'chapter 2 verse 25' not in output:
                            analysis['issues'].append('Failed to convert scriptural reference')
                        if output.count('25') > 1:
                            analysis['issues'].append('Over-converted idiomatic usage')
                            
                    elif test_case['context'] == 'ambiguous_years':
                        # Should handle both year formats
                        if '2500' not in output or '2005' not in output:
                            analysis['issues'].append('Failed to handle both year formats')
                            
                    elif test_case['context'] == 'sanskrit_with_numbers':
                        # Should capitalize Krishna
                        if 'Krishna' not in output:
                            analysis['issues'].append('Failed to capitalize Sanskrit proper noun')
                        if '108' not in output or '18' not in output:
                            analysis['issues'].append('Failed to convert numbers appropriately')
                            
                    test_results.append(analysis)
                    
                    if analysis['issues']:
                        result['correctness_issues'].extend(analysis['issues'])
                        
                except Exception as e:
                    error_details = f"Test case {i+1} failed: {e}"
                    result['correctness_issues'].append(error_details)
                    self.log_failure('Functional_Correctness_Novel', error_details, False)
                    
            result['details']['novel_test_results'] = test_results
            result['details']['total_issues'] = len(result['correctness_issues'])
            result['passed'] = len(result['correctness_issues']) == 0
            
        except Exception as e:
            result['details']['exception'] = str(e)
            self.log_failure('Functional_Correctness_Novel', str(e), True)
            
        return result
        
    def test_integration_stability_novel(self) -> Dict[str, Any]:
        """Novel test: End-to-end integration with real-world scenarios"""
        print("=== Novel Test 4: Integration Stability with Real Scenarios ===")
        
        result = {
            'test_name': 'Integration_Stability_Novel',
            'passed': False,
            'details': {},
            'integration_issues': []
        }
        
        try:
            from post_processors.sanskrit_post_processor import SanskritPostProcessor
            from utils.srt_parser import SRTParser
            
            # Create a realistic SRT file content
            realistic_srt_content = '''1
00:00:01,000 --> 00:00:05,000
Um, welcome everyone to today's, uh, lecture on the Bhagavad Gita.

2
00:00:06,000 --> 00:00:12,000
We will be studying chapter two verse twenty five, which speaks about, well, the eternal soul.

3
00:00:13,000 --> 00:00:20,000
Krishna, or should I say Krsna, teaches Arjuna about dharma and karma yoga.

4
00:00:21,000 --> 00:00:28,000
In the year two thousand five, scholars discovered, actually, let me correct that - rather, they rediscovered ancient manuscripts.

5
00:00:29,000 --> 00:00:35,000
The teachings of Patanjali and Shankaracharya are, um, foundational to understanding yoga philosophy.'''
            
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
            
            # Validate the output
            if Path(temp_output).exists():
                with open(temp_output, 'r', encoding='utf-8') as f:
                    processed_content = f.read()
                    
                result['details']['processing_metrics'] = {
                    'total_segments': metrics.total_segments,
                    'segments_modified': metrics.segments_modified,
                    'processing_time': processing_time,
                    'average_confidence': metrics.average_confidence
                }
                
                # Analyze output quality
                quality_checks = []
                
                # Check 1: Were filler words removed?
                if 'Um,' in processed_content or 'uh,' in processed_content:
                    quality_checks.append('FAILED: Filler words not removed')
                    
                # Check 2: Were numbers converted appropriately?
                if 'chapter 2 verse 25' not in processed_content:
                    quality_checks.append('FAILED: Scriptural reference not converted')
                    
                if '2005' not in processed_content:
                    quality_checks.append('FAILED: Year not converted')
                    
                # Check 3: Were Sanskrit terms capitalized?
                if 'Krishna' not in processed_content:
                    quality_checks.append('FAILED: Sanskrit proper nouns not capitalized')
                    
                # Check 4: Were conversational patterns handled?
                if 'I mean' in processed_content or 'rather,' not in processed_content:
                    quality_checks.append('FAILED: Conversational patterns not handled properly')
                    
                result['details']['quality_checks'] = quality_checks
                result['details']['processed_content_sample'] = processed_content[:500] + "..."
                
                if quality_checks:
                    result['integration_issues'].extend(quality_checks)
                    
            else:
                result['integration_issues'].append('CRITICAL: Output file not created')
                self.log_failure('Integration_Stability_Novel', 'Output file not created', True)
                
            # Cleanup
            import os
            if os.path.exists(temp_input):
                os.unlink(temp_input)
            if os.path.exists(temp_output):
                os.unlink(temp_output)
                
            result['passed'] = len(result['integration_issues']) == 0
            
        except Exception as e:
            result['details']['exception'] = str(e)
            result['integration_issues'].append(f'Integration test failed: {e}')
            self.log_failure('Integration_Stability_Novel', str(e), True)
            
        return result
        
    def run_comprehensive_novel_validation(self) -> Dict[str, Any]:
        """Run all novel validation tests and provide comprehensive report"""
        print("🔍 NOVEL QA FRAMEWORK - INDEPENDENT VALIDATION")
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
            self.test_performance_under_stress,
            self.test_functional_correctness_novel,
            self.test_integration_stability_novel
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
            
        # Generate recommendations
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
    print("📋 NOVEL QA VALIDATION REPORT")
    print("=" * 60)
    
    print(f"\n🎯 OVERALL STATUS: {results['overall_status']}")
    print(f"📊 Tests Passed: {sum(1 for test in results['test_results'].values() if test['passed'])}/{len(results['test_results'])}")
    
    if results['critical_failures']:
        print(f"\n❌ CRITICAL FAILURES ({len(results['critical_failures'])}):")
        for i, failure in enumerate(results['critical_failures'], 1):
            print(f"   {i}. {failure}")
            
    if results['performance_issues']:
        print(f"\n⚡ PERFORMANCE ISSUES ({len(results['performance_issues'])}):")
        for i, issue in enumerate(results['performance_issues'], 1):
            print(f"   {i}. {issue}")
            
    print(f"\n💡 RECOMMENDATIONS ({len(results['recommendations'])}):")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"   {i}. {rec}")
        
    # Save results to file
    results_file = f"novel_qa_validation_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Return exit code based on results
    if results['overall_status'].startswith('CRITICAL') or results['overall_status'].startswith('FAILED'):
        return 1
    else:
        return 0

if __name__ == "__main__":
    exit(main())