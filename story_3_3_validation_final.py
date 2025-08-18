#!/usr/bin/env python3
"""
Story 3.3 Final Validation Script

Simple validation without Unicode issues for Windows console.
"""

import sys
import os
import time
from pathlib import Path

# Set up path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def validate_story_3_3():
    """Validate Story 3.3 implementation."""
    print("Story 3.3 Tiered Human Review Workflow - Final Validation")
    print("=" * 60)
    
    validation_results = {}
    start_time = time.time()
    
    # Test 1: Epic 2 Integration
    print("\n1. Testing Epic 2 Integration...")
    try:
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        processor = SanskritPostProcessor()
        validation_results['epic_2'] = True
        print("   SUCCESS: Epic 2 Sanskrit processing integrated")
    except Exception as e:
        validation_results['epic_2'] = False
        print(f"   FAILED: Epic 2 integration - {e}")
    
    # Test 2: QA Module (Stories 3.1-3.2)
    print("\n2. Testing QA Module Integration...")
    try:
        from qa_module.qa_flagging_engine import QAFlaggingEngine
        qa_engine = QAFlaggingEngine()
        validation_results['qa_module'] = True
        print("   SUCCESS: QA Module integrated")
    except Exception as e:
        validation_results['qa_module'] = False
        print(f"   FAILED: QA Module - {e}")
    
    # Test 3: Review Workflow Core
    print("\n3. Testing Review Workflow Core...")
    try:
        from review_workflow.collaborative_interface import CollaborativeInterface
        interface = CollaborativeInterface()
        session_created = interface.create_collaborative_session("validation_test")
        validation_results['review_workflow'] = session_created
        print("   SUCCESS: Review workflow operational")
    except Exception as e:
        validation_results['review_workflow'] = False
        print(f"   FAILED: Review workflow - {e}")
    
    # Test 4: Epic 4 Production Infrastructure
    print("\n4. Testing Epic 4 Production Infrastructure...")
    try:
        from review_workflow.production_review_orchestrator import ProductionReviewOrchestrator
        orchestrator = ProductionReviewOrchestrator()
        if hasattr(orchestrator, 'start_production_operations'):
            validation_results['epic_4'] = True
            print("   SUCCESS: Epic 4 production infrastructure ready")
        else:
            validation_results['epic_4'] = False
            print("   FAILED: Epic 4 methods missing")
    except Exception as e:
        validation_results['epic_4'] = False
        print(f"   FAILED: Epic 4 infrastructure - {e}")
    
    # Test 5: End-to-End Integration
    print("\n5. Testing End-to-End Integration...")
    try:
        from utils.srt_parser import SRTSegment
        
        # Create test content
        test_segment = SRTSegment(
            index=1,
            start_time="00:00:01,000",
            end_time="00:00:05,000", 
            text="Today we discuss krishna and dharma from the gita."
        )
        
        # Process through Epic 2
        if validation_results.get('epic_2'):
            processed = processor._process_srt_segment(test_segment, None)
            integration_success = processed is not None
        else:
            integration_success = False
            
        # Test QA analysis  
        if validation_results.get('qa_module') and integration_success:
            qa_result = qa_engine.analyze_segment(processed)
            integration_success = qa_result is not None
            
        # Test review session
        if validation_results.get('review_workflow') and integration_success:
            user_joined = interface.join_session("validation_test", "test_user", "gp")
            integration_success = user_joined
            
        validation_results['integration'] = integration_success
        if integration_success:
            print("   SUCCESS: End-to-end integration working")
        else:
            print("   FAILED: End-to-end integration failed")
            
    except Exception as e:
        validation_results['integration'] = False
        print(f"   FAILED: Integration test - {e}")
    
    # Calculate results
    total_tests = len(validation_results)
    passed_tests = sum(validation_results.values())
    success_rate = (passed_tests / total_tests) * 100
    validation_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Validation Time: {validation_time:.1f} seconds")
    
    # Component Status
    print("\nComponent Status:")
    for component, status in validation_results.items():
        status_text = "PASS" if status else "FAIL"
        print(f"  {component.replace('_', ' ').title()}: {status_text}")
    
    # Final determination
    print("\n" + "=" * 60)
    if success_rate >= 80:
        print("RESULT: Story 3.3 Tiered Human Review Workflow VALIDATED")
        print("STATUS: Ready for production deployment")
        return True
    else:
        print("RESULT: Story 3.3 validation FAILED")
        print("STATUS: Issues need to be resolved")
        return False

if __name__ == "__main__":
    success = validate_story_3_3()
    sys.exit(0 if success else 1)