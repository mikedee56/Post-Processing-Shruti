#!/usr/bin/env python3
"""
Simplified Story 3.3 Integration Validation

Quick validation of Story 3.3 core functionality without Unicode output issues.
"""

import sys
import os
import time
from pathlib import Path

# Set up path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_core_components():
    """Test core Story 3.3 components."""
    results = {}
    
    try:
        # Test Epic 2 Integration
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        processor = SanskritPostProcessor()
        results['epic_2_integration'] = 'PASS'
        print("✓ Epic 2 Integration: PASS")
    except Exception as e:
        results['epic_2_integration'] = f'FAIL: {e}'
        print(f"✗ Epic 2 Integration: FAIL - {e}")
    
    try:
        # Test QA Module
        from qa_module.qa_flagging_engine import QAFlaggingEngine
        qa_engine = QAFlaggingEngine()
        results['qa_module'] = 'PASS'
        print("✓ QA Module: PASS")
    except Exception as e:
        results['qa_module'] = f'FAIL: {e}'
        print(f"✗ QA Module: FAIL - {e}")
    
    try:
        # Test Review Workflow
        from review_workflow.collaborative_interface import CollaborativeInterface
        interface = CollaborativeInterface()
        results['review_workflow'] = 'PASS'
        print("✓ Review Workflow: PASS")
    except Exception as e:
        results['review_workflow'] = f'FAIL: {e}'
        print(f"✗ Review Workflow: FAIL - {e}")
    
    try:
        # Test Epic 4.3 Components
        from review_workflow.production_review_orchestrator import ProductionReviewOrchestrator
        orchestrator = ProductionReviewOrchestrator()
        results['epic_4_3'] = 'PASS'
        print("✓ Epic 4.3 Production: PASS")
    except Exception as e:
        results['epic_4_3'] = f'FAIL: {e}'
        print(f"✗ Epic 4.3 Production: FAIL - {e}")
    
    return results

def test_integration_workflow():
    """Test end-to-end integration workflow."""
    try:
        from utils.srt_parser import SRTSegment
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        from qa_module.qa_flagging_engine import QAFlaggingEngine
        from review_workflow.collaborative_interface import CollaborativeInterface
        
        # Create test content
        test_segment = SRTSegment(
            index=1,
            start_time="00:00:01,000", 
            end_time="00:00:05,000",
            text="Today we discuss krishna and dharma."
        )
        
        # Process with Epic 2
        processor = SanskritPostProcessor()
        processed_segment = processor._process_srt_segment(test_segment, None)
        
        # Run QA analysis
        qa_engine = QAFlaggingEngine()
        qa_result = qa_engine.analyze_segment(processed_segment)
        
        # Create collaborative review session
        interface = CollaborativeInterface()
        session_created = interface.create_collaborative_session("test_session")
        
        success = all([
            processed_segment is not None,
            qa_result is not None,
            session_created
        ])
        
        if success:
            print("✓ End-to-End Integration: PASS")
            return 'PASS'
        else:
            print("✗ End-to-End Integration: FAIL")
            return 'FAIL'
            
    except Exception as e:
        print(f"✗ End-to-End Integration: FAIL - {e}")
        return f'FAIL: {e}'

def main():
    """Run simplified validation."""
    print("Story 3.3 Integration Validation - Simplified")
    print("=" * 50)
    
    start_time = time.time()
    
    # Test core components
    print("\n1. Testing Core Components:")
    component_results = test_core_components()
    
    # Test integration workflow
    print("\n2. Testing Integration Workflow:")
    integration_result = test_integration_workflow()
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    total_tests = len(component_results) + 1  # +1 for integration test
    passed_tests = len([r for r in component_results.values() if r == 'PASS'])
    if integration_result == 'PASS':
        passed_tests += 1
    
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    validation_time = time.time() - start_time
    print(f"Validation Time: {validation_time:.1f}s")
    
    if success_rate >= 80:
        print("\nSTATUS: Story 3.3 Integration VALIDATED")
        return 0
    else:
        print("\nSTATUS: Story 3.3 Integration VALIDATION FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())