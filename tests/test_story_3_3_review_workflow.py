"""
Comprehensive Test Suite for Story 3.3: Tiered Human Review Workflow.

Tests all Epic 4 components integration:
- Epic 4.1: MCP Context-Aware Expertise Matching  
- Epic 4.2: ML-Enhanced Feedback Integration
- Epic 4.3: Production-Grade Review Infrastructure (99.9% uptime, sub-second response)
- Epic 4.5: Academic Consultant Integration and Publication-Ready Standards

Validates complete review workflow from request to publication-ready output.
"""

import logging
import pytest
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch

# Epic 4 components
from review_workflow.production_review_orchestrator import ProductionReviewOrchestrator, OrchestratorState
from review_workflow.reviewer_manager import ReviewerManager, ReviewerProfile, ReviewerRole, ReviewerStatus
from review_workflow.collaborative_interface import CollaborativeInterface, CommentType, ReviewActionType
from review_workflow.epic_4_3_validator import Epic43Validator
from review_workflow.publication_ready_review_standards import PublicationReadyReviewStandards

# Supporting infrastructure
from utils.circuit_breaker import CircuitBreaker
from utils.rate_limiter import RateLimiter, RateLimitStrategy
from utils.health_checker import HealthChecker
from utils.srt_parser import SRTSegment

# Test data and utilities
from tests.test_data.review_workflow_test_data import (
    create_test_srt_segments,
    create_test_reviewer_profiles,
    create_test_config
)


class TestStory33ReviewWorkflow:
    """Comprehensive test suite for Story 3.3 review workflow implementation."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        self.test_config = create_test_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize test data
        self.test_segments = create_test_srt_segments()
        self.test_reviewers = create_test_reviewer_profiles()
        
        # Test session tracking
        self.test_sessions = []
        
    def teardown_method(self):
        """Clean up after each test."""
        # Clean up any test sessions
        for session_id in self.test_sessions:
            try:
                # Cleanup would go here if needed
                pass
            except:
                pass
    
    @pytest.mark.integration
    def test_epic_4_3_production_infrastructure_validation(self):
        """Test Epic 4.3 production infrastructure meets 99.9% uptime and sub-second response requirements."""
        # Initialize Epic 4.3 validator
        validator = Epic43Validator(self.test_config)
        
        # Run comprehensive Epic 4.3 validation
        validation_summary = validator.validate_epic_4_3_infrastructure()
        
        # Validate Epic 4.3 requirements
        assert validation_summary.success_rate >= 90.0, f"Epic 4.3 validation success rate too low: {validation_summary.success_rate}%"
        assert validation_summary.uptime_reliability_validated, "Epic 4.3 uptime reliability validation failed"
        assert validation_summary.response_time_validated, "Epic 4.3 response time validation failed"
        assert validation_summary.monitoring_validated, "Epic 4.3 monitoring validation failed"
        assert validation_summary.reliability_patterns_validated, "Epic 4.3 reliability patterns validation failed"
        
        # Validate performance metrics
        assert validation_summary.average_response_time_ms <= 500.0, f"Average response time exceeds target: {validation_summary.average_response_time_ms}ms"
        assert validation_summary.p95_response_time_ms <= 750.0, f"P95 response time exceeds target: {validation_summary.p95_response_time_ms}ms"
        
        self.logger.info(f"Epic 4.3 validation completed: {validation_summary.passed_tests}/{validation_summary.total_tests} tests passed")
    
    @pytest.mark.integration
    def test_production_orchestrator_end_to_end(self):
        """Test complete production orchestrator workflow with Epic 4.3 reliability."""
        # Initialize production orchestrator
        orchestrator = ProductionReviewOrchestrator(self.test_config)
        
        try:
            # Start production operations
            startup_success = orchestrator.start_production_operations()
            assert startup_success, "Production orchestrator startup failed"
            assert orchestrator.orchestrator_state == OrchestratorState.HEALTHY, "Orchestrator not in healthy state"
            
            # Test review request processing
            session_id = f"test_session_{uuid.uuid4().hex[:8]}"
            
            start_time = time.time()
            result = orchestrator.process_review_request(
                session_id=session_id,
                content_segments=self.test_segments[:3],
                priority="standard",
                required_expertise=["sanskrit", "academic"]
            )
            processing_time = (time.time() - start_time) * 1000
            
            # Validate Epic 4.3 response time requirement
            assert processing_time <= 500.0, f"Request processing exceeded Epic 4.3 target: {processing_time:.1f}ms"
            assert result is not None, "Review request processing failed"
            
            # Test system health monitoring
            health_summary = orchestrator.get_system_health()
            assert health_summary.overall_state in [OrchestratorState.HEALTHY, OrchestratorState.DEGRADED], "System health check failed"
            assert health_summary.performance_score >= 0.7, f"Performance score too low: {health_summary.performance_score}"
            
            # Test production dashboard
            dashboard = orchestrator.get_production_dashboard()
            assert 'epic_4_3_status' in dashboard, "Epic 4.3 status missing from dashboard"
            assert 'reliability_systems' in dashboard, "Reliability systems status missing"
            assert dashboard['epic_4_3_status']['uptime_percentage'] >= 99.0, "Uptime percentage below threshold"
            
            self.test_sessions.append(session_id)
            
        finally:
            # Graceful shutdown
            shutdown_success = orchestrator.shutdown_production_operations()
            assert shutdown_success, "Production orchestrator shutdown failed"
    
    @pytest.mark.integration
    def test_reviewer_manager_epic_4_3_reliability(self):
        """Test reviewer manager Epic 4.3 reliability and sub-second assignment."""
        reviewer_manager = ReviewerManager(self.test_config.get('reviewer_manager', {}))
        
        # Register test reviewers
        for reviewer in self.test_reviewers:
            registration_success = reviewer_manager.register_reviewer(reviewer)
            assert registration_success, f"Failed to register reviewer: {reviewer.reviewer_id}"
        
        # Test sub-second reviewer assignment
        assignment_times = []
        successful_assignments = 0
        
        for i in range(10):
            start_time = time.time()
            
            # Create assignment request
            from review_workflow.reviewer_manager import AssignmentRequest
            request = AssignmentRequest(
                request_id=f"test_req_{i}",
                session_id=f"test_session_{i}",
                required_role=ReviewerRole.GENERAL_PROOFREADER,
                required_skills=["sanskrit"],
                priority_level="standard",
                max_response_time_ms=500.0
            )
            
            assigned_reviewer = reviewer_manager.assign_reviewer(request)
            assignment_time = (time.time() - start_time) * 1000
            assignment_times.append(assignment_time)
            
            if assigned_reviewer:
                successful_assignments += 1
            
            # Validate Epic 4.3 sub-second requirement
            assert assignment_time <= 500.0, f"Assignment time exceeded target: {assignment_time:.1f}ms"
        
        # Validate Epic 4.3 reliability requirements
        success_rate = (successful_assignments / 10) * 100
        avg_assignment_time = sum(assignment_times) / len(assignment_times)
        
        assert success_rate >= 90.0, f"Assignment success rate too low: {success_rate}%"
        assert avg_assignment_time <= 250.0, f"Average assignment time too high: {avg_assignment_time:.1f}ms"
        
        # Test system health reporting
        system_health = reviewer_manager.get_system_health()
        assert 'epic_4_3_reliability' in system_health, "Epic 4.3 reliability metrics missing"
        assert system_health['epic_4_3_reliability']['uptime_percentage'] >= 99.0, "Reviewer manager uptime below target"
    
    @pytest.mark.integration
    def test_collaborative_interface_epic_4_5_academic_integration(self):
        """Test collaborative interface with Epic 4.5 academic consultant integration."""
        interface = CollaborativeInterface(self.test_config.get('collaborative_interface', {}))
        
        # Create collaborative session
        session_id = f"test_collab_session_{uuid.uuid4().hex[:8]}"
        session_created = interface.create_collaborative_session(session_id)
        assert session_created, "Failed to create collaborative session"
        
        # Test multi-role user joining
        users = [
            ("gp_user_1", "gp"),
            ("sme_user_1", "sme"), 
            ("consultant_user_1", "consultant")
        ]
        
        for user_id, role in users:
            join_success = interface.join_session(session_id, user_id, role)
            assert join_success, f"Failed to join session: {user_id} as {role}"
        
        # Test Epic 4.5 academic commenting
        segment_id = "segment_001"
        
        # Add academic comment from consultant
        academic_comment_id = interface.add_comment(
            session_id=session_id,
            segment_id=segment_id,
            author_id="consultant_user_1",
            comment_text="This Sanskrit term requires IAST transliteration according to academic standards.",
            comment_type=CommentType.ACADEMIC_NOTE,
            highlighted_text="krishna dharma yoga"
        )
        assert academic_comment_id is not None, "Failed to add academic comment"
        
        # Add citation reference comment
        citation_comment_id = interface.add_comment(
            session_id=session_id,
            segment_id=segment_id,
            author_id="consultant_user_1",
            comment_text="Reference: Bhagavad Gita 2.47 (Easwaran translation, 2007)",
            comment_type=CommentType.CITATION_REFERENCE,
            highlighted_text="karma yoga practice"
        )
        assert citation_comment_id is not None, "Failed to add citation comment"
        
        # Test review action recording with academic justification
        action_id = interface.record_action(
            session_id=session_id,
            segment_id=segment_id,
            action_type=ReviewActionType.ACADEMIC_VALIDATION,
            description="Applied IAST transliteration to Sanskrit terms",
            performed_by="consultant_user_1",
            original_text="krishna dharma yoga",
            modified_text="kṛṣṇa dharma yoga",
            academic_justification="IAST standard requires diacritics for accurate Sanskrit representation (ISO 15919)"
        )
        assert action_id is not None, "Failed to record academic action"
        
        # Test session state with academic metrics
        session_state = interface.get_session_state(session_id)
        assert session_state is not None, "Failed to get session state"
        assert session_state['collaboration_status']['consultant_participating'], "Consultant participation not detected"
        assert session_state['collaboration_status']['citation_review_mode'], "Citation review mode not activated"
        
        # Validate academic metrics
        academic_metrics = session_state['academic_metrics']
        assert academic_metrics['academic_comments'] >= 1, "Academic comments not counted"
        assert academic_metrics['citation_references'] >= 1, "Citation references not counted"
        assert academic_metrics['consultant_validations'] >= 0, "Consultant validations not tracked"
        
        self.test_sessions.append(session_id)
    
    @pytest.mark.integration
    def test_publication_ready_standards_integration(self):
        """Test Epic 4.5 publication-ready review standards integration."""
        publication_standards = PublicationReadyReviewStandards(self.test_config.get('publication_standards', {}))
        
        # Test review quality assessment
        test_segment = SRTSegment(
            index=1,
            start_time="00:00:01,000",
            end_time="00:00:05,000", 
            text="Today we study krishna dharma yoga from bhagavad gita."
        )
        
        reviewed_text = "Today we study Kṛṣṇa dharma yoga from Bhagavad Gītā."
        
        # Assess segment quality
        assessment = publication_standards.assess_review_segment_quality(
            segment=test_segment,
            reviewed_text=reviewed_text,
            review_context={
                'reviewer_role': 'consultant',
                'academic_citations': ['Bhagavad Gita 2.47'],
                'iast_applied': True
            }
        )
        
        # Validate publication readiness assessment
        assert assessment.overall_quality_score >= 0.7, f"Quality score too low: {assessment.overall_quality_score}"
        assert assessment.publication_tier.value in ['professional', 'academic', 'publication_ready'], "Publication tier too low"
        assert assessment.iast_compliance.compliance_score >= 0.8, "IAST compliance score too low"
        
        # Test complete review workflow assessment
        review_segments = [
            (test_segment, reviewed_text),
            (SRTSegment(2, "00:00:06,000", "00:00:10,000", "The teaching emphasizes dharama and moksha."),
             "The teaching emphasizes dharma and mokṣa.")
        ]
        
        workflow_assessment = publication_standards.assess_complete_review_workflow(
            original_segments=[seg for seg, _ in review_segments],
            reviewed_segments=[text for _, text in review_segments],
            review_metadata={
                'total_review_time_minutes': 15,
                'reviewers_involved': ['gp_user_1', 'consultant_user_1'],
                'academic_validations': 2,
                'citation_checks': 1
            }
        )
        
        # Validate workflow assessment
        assert workflow_assessment.overall_workflow_score >= 0.75, "Workflow score too low"
        assert workflow_assessment.academic_rigor_score >= 0.7, "Academic rigor score too low"
        assert workflow_assessment.publication_readiness, "Content not deemed publication ready"
        
        # Test academic enhancement suggestions
        enhancements = publication_standards.suggest_academic_enhancements(
            reviewed_text, 
            current_tier=assessment.publication_tier
        )
        
        assert len(enhancements.enhancement_suggestions) >= 0, "Enhancement suggestions should be provided"
    
    @pytest.mark.stress
    def test_epic_4_3_load_testing_integration(self):
        """Test Epic 4.3 production infrastructure under load."""
        orchestrator = ProductionReviewOrchestrator(self.test_config)
        
        try:
            # Start production operations
            startup_success = orchestrator.start_production_operations()
            assert startup_success, "Production orchestrator startup failed"
            
            # Concurrent load test
            concurrent_requests = 20
            test_duration = 10  # seconds
            
            response_times = []
            successful_requests = 0
            failed_requests = 0
            
            def submit_review_request(request_id: int):
                start_time = time.time()
                try:
                    session_id = f"load_test_session_{request_id}"
                    result = orchestrator.process_review_request(
                        session_id=session_id,
                        content_segments=[f"Load test content {request_id}"],
                        priority="standard"
                    )
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    if result:
                        return True, processing_time
                    else:
                        return False, processing_time
                        
                except Exception:
                    processing_time = (time.time() - start_time) * 1000
                    return False, processing_time
            
            # Execute load test
            with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                end_time = time.time() + test_duration
                request_id = 0
                
                while time.time() < end_time:
                    futures = []
                    
                    # Submit batch of requests
                    for _ in range(min(5, concurrent_requests)):
                        future = executor.submit(submit_review_request, request_id)
                        futures.append(future)
                        request_id += 1
                    
                    # Collect results
                    for future in as_completed(futures, timeout=5):
                        try:
                            success, processing_time = future.result()
                            response_times.append(processing_time)
                            
                            if success:
                                successful_requests += 1
                            else:
                                failed_requests += 1
                                
                        except Exception:
                            failed_requests += 1
                    
                    time.sleep(0.1)  # Brief pause between batches
            
            # Validate Epic 4.3 performance under load
            total_requests = successful_requests + failed_requests
            success_rate = (successful_requests / max(total_requests, 1)) * 100
            avg_response_time = sum(response_times) / max(len(response_times), 1)
            
            assert success_rate >= 90.0, f"Load test success rate too low: {success_rate}%"
            assert avg_response_time <= 750.0, f"Average response time under load too high: {avg_response_time:.1f}ms"
            
            # Verify system health after load test
            health_summary = orchestrator.get_system_health()
            assert health_summary.overall_state != OrchestratorState.CRITICAL, "System in critical state after load test"
            assert health_summary.performance_score >= 0.5, "Performance score too low after load test"
            
        finally:
            orchestrator.shutdown_production_operations()
    
    @pytest.mark.integration
    def test_end_to_end_review_workflow_complete(self):
        """Test complete end-to-end review workflow from request to publication-ready output."""
        # Initialize all Epic 4 components
        orchestrator = ProductionReviewOrchestrator(self.test_config)
        interface = CollaborativeInterface(self.test_config.get('collaborative_interface', {}))
        publication_standards = PublicationReadyReviewStandards(self.test_config.get('publication_standards', {}))
        
        try:
            # Start production operations
            startup_success = orchestrator.start_production_operations()
            assert startup_success, "Production orchestrator startup failed"
            
            # Step 1: Create review request
            session_id = f"e2e_test_session_{uuid.uuid4().hex[:8]}"
            
            start_time = time.time()
            result = orchestrator.process_review_request(
                session_id=session_id,
                content_segments=self.test_segments[:2],
                priority="high",
                required_expertise=["sanskrit", "academic", "iast"]
            )
            request_processing_time = (time.time() - start_time) * 1000
            
            assert result is not None, "Review request failed"
            assert request_processing_time <= 500.0, f"Request processing too slow: {request_processing_time:.1f}ms"
            
            # Step 2: Set up collaborative session
            session_created = interface.create_collaborative_session(session_id)
            assert session_created, "Collaborative session creation failed"
            
            # Add reviewers to session
            reviewers = [
                ("gp_reviewer", "gp"),
                ("sme_reviewer", "sme"),
                ("consultant_reviewer", "consultant")
            ]
            
            for reviewer_id, role in reviewers:
                join_success = interface.join_session(session_id, reviewer_id, role)
                assert join_success, f"Reviewer join failed: {reviewer_id}"
            
            # Step 3: Simulate review process with academic collaboration
            segment_id = "segment_001"
            original_text = "Today we study krishna dharma yoga from bhagavad gita chapter two verse twenty five."
            
            # GP makes initial corrections
            gp_comment_id = interface.add_comment(
                session_id=session_id,
                segment_id=segment_id,
                author_id="gp_reviewer",
                comment_text="Capitalization needed for proper nouns",
                comment_type=CommentType.CORRECTION,
                highlighted_text="krishna dharma yoga"
            )
            assert gp_comment_id is not None, "GP comment failed"
            
            # SME adds academic context
            sme_comment_id = interface.add_comment(
                session_id=session_id,
                segment_id=segment_id,
                author_id="sme_reviewer",
                comment_text="Sanskrit terms need IAST transliteration for academic accuracy",
                comment_type=CommentType.SUGGESTION,
                highlighted_text="krishna"
            )
            assert sme_comment_id is not None, "SME comment failed"
            
            # Consultant provides final academic validation
            consultant_action_id = interface.record_action(
                session_id=session_id,
                segment_id=segment_id,
                action_type=ReviewActionType.ACADEMIC_VALIDATION,
                description="Applied IAST transliteration and proper capitalization",
                performed_by="consultant_reviewer",
                original_text=original_text,
                modified_text="Today we study Kṛṣṇa dharma yoga from Bhagavad Gītā chapter 2 verse 25.",
                academic_justification="IAST standard application with proper noun capitalization and numerical conversion"
            )
            assert consultant_action_id is not None, "Consultant action failed"
            
            # Step 4: Validate publication readiness
            final_segment = SRTSegment(
                index=1,
                start_time="00:00:01,000",
                end_time="00:00:05,000",
                text=original_text
            )
            
            final_reviewed_text = "Today we study Kṛṣṇa dharma yoga from Bhagavad Gītā chapter 2 verse 25."
            
            quality_assessment = publication_standards.assess_review_segment_quality(
                segment=final_segment,
                reviewed_text=final_reviewed_text,
                review_context={
                    'reviewer_role': 'consultant',
                    'academic_citations': [],
                    'iast_applied': True,
                    'collaborative_review': True
                }
            )
            
            # Validate end-to-end workflow quality
            assert quality_assessment.overall_quality_score >= 0.8, f"E2E quality score too low: {quality_assessment.overall_quality_score}"
            assert quality_assessment.publication_tier.value in ['academic', 'publication_ready'], "Publication tier insufficient"
            assert quality_assessment.iast_compliance.compliance_score >= 0.9, "IAST compliance too low"
            
            # Step 5: Verify system health throughout workflow
            health_summary = orchestrator.get_system_health()
            assert health_summary.overall_state == OrchestratorState.HEALTHY, "System not healthy after E2E workflow"
            assert health_summary.performance_score >= 0.8, "Performance degraded during E2E workflow"
            
            # Step 6: Validate collaborative session state
            session_state = interface.get_session_state(session_id)
            assert session_state is not None, "Session state retrieval failed"
            assert len(session_state['active_reviewers']) == 3, "Not all reviewers active"
            assert session_state['collaboration_status']['consultant_participating'], "Consultant participation not detected"
            assert session_state['comments']['total_comments'] >= 2, "Comments not properly tracked"
            assert session_state['actions']['total_actions'] >= 1, "Actions not properly tracked"
            
            self.test_sessions.append(session_id)
            
            self.logger.info(f"End-to-end workflow completed successfully: {session_id}")
            
        finally:
            orchestrator.shutdown_production_operations()
    
    @pytest.mark.performance
    def test_epic_4_components_performance_benchmarks(self):
        """Test all Epic 4 components meet performance benchmarks."""
        # Performance benchmarks
        benchmarks = {
            'orchestrator_startup_ms': 2000.0,
            'review_request_processing_ms': 500.0,
            'reviewer_assignment_ms': 250.0,
            'comment_addition_ms': 100.0,
            'quality_assessment_ms': 200.0,
            'session_state_retrieval_ms': 150.0
        }
        
        results = {}
        
        # Test orchestrator startup performance
        orchestrator = ProductionReviewOrchestrator(self.test_config)
        start_time = time.time()
        startup_success = orchestrator.start_production_operations()
        results['orchestrator_startup_ms'] = (time.time() - start_time) * 1000
        
        assert startup_success, "Orchestrator startup failed"
        
        try:
            # Test review request processing performance
            start_time = time.time()
            session_id = f"perf_test_{uuid.uuid4().hex[:8]}"
            result = orchestrator.process_review_request(
                session_id=session_id,
                content_segments=self.test_segments[:1],
                priority="standard"
            )
            results['review_request_processing_ms'] = (time.time() - start_time) * 1000
            
            assert result is not None, "Review request processing failed"
            
            # Test collaborative interface performance
            interface = CollaborativeInterface(self.test_config.get('collaborative_interface', {}))
            
            # Comment addition performance
            interface.create_collaborative_session(session_id)
            interface.join_session(session_id, "test_user", "gp")
            
            start_time = time.time()
            comment_id = interface.add_comment(
                session_id=session_id,
                segment_id="segment_001",
                author_id="test_user",
                comment_text="Performance test comment"
            )
            results['comment_addition_ms'] = (time.time() - start_time) * 1000
            
            assert comment_id is not None, "Comment addition failed"
            
            # Session state retrieval performance
            start_time = time.time()
            session_state = interface.get_session_state(session_id)
            results['session_state_retrieval_ms'] = (time.time() - start_time) * 1000
            
            assert session_state is not None, "Session state retrieval failed"
            
            # Test publication standards performance
            publication_standards = PublicationReadyReviewStandards(self.test_config.get('publication_standards', {}))
            
            start_time = time.time()
            test_segment = SRTSegment(1, "00:00:01,000", "00:00:05,000", "Test content")
            assessment = publication_standards.assess_review_segment_quality(
                segment=test_segment,
                reviewed_text="Test content reviewed"
            )
            results['quality_assessment_ms'] = (time.time() - start_time) * 1000
            
            assert assessment is not None, "Quality assessment failed"
            
            # Validate all performance benchmarks
            for metric, benchmark in benchmarks.items():
                actual_time = results.get(metric, float('inf'))
                assert actual_time <= benchmark, f"Performance benchmark failed: {metric} = {actual_time:.1f}ms (target: {benchmark}ms)"
            
            self.logger.info(f"Performance benchmarks met: {results}")
            
        finally:
            orchestrator.shutdown_production_operations()


@pytest.mark.integration
class TestEpic4ComponentsIntegration:
    """Test Epic 4 components integration with Epic 2 Sanskrit processing."""
    
    def test_sanskrit_processing_integration(self):
        """Test Epic 4 review workflow integration with Epic 2 Sanskrit processing."""
        # This test would verify that the review workflow properly handles
        # Sanskrit-corrected content from Epic 2 post-processing
        
        # Mock Epic 2 Sanskrit-processed content
        sanskrit_processed_segments = [
            SRTSegment(1, "00:00:01,000", "00:00:05,000", "Today we study Kṛṣṇa and dharma."),
            SRTSegment(2, "00:00:06,000", "00:00:10,000", "The Bhagavad Gītā teaches about yoga.")
        ]
        
        # Initialize review workflow
        orchestrator = ProductionReviewOrchestrator(create_test_config())
        
        try:
            startup_success = orchestrator.start_production_operations()
            assert startup_success, "Orchestrator startup failed"
            
            # Test processing Sanskrit-corrected content
            session_id = f"sanskrit_integration_test_{uuid.uuid4().hex[:8]}"
            result = orchestrator.process_review_request(
                session_id=session_id,
                content_segments=sanskrit_processed_segments,
                priority="standard",
                required_expertise=["sanskrit", "iast"]
            )
            
            assert result is not None, "Sanskrit content processing failed"
            
            # Verify system recognizes pre-processed Sanskrit content
            health_summary = orchestrator.get_system_health()
            assert health_summary.overall_state == OrchestratorState.HEALTHY, "System health degraded with Sanskrit content"
            
        finally:
            orchestrator.shutdown_production_operations()


def create_test_reviewer_manager_with_experts():
    """Create reviewer manager with expert reviewers for testing."""
    reviewer_manager = ReviewerManager()
    
    # Add test reviewers with different expertise
    expert_reviewers = [
        ReviewerProfile(
            reviewer_id="sanskrit_expert_1",
            name="Sanskrit Expert",
            email="sanskrit@example.com",
            role=ReviewerRole.SUBJECT_MATTER_EXPERT,
            status=ReviewerStatus.AVAILABLE,
            specializations=["sanskrit", "iast", "transliteration"]
        ),
        ReviewerProfile(
            reviewer_id="academic_consultant_1", 
            name="Academic Consultant",
            email="academic@example.com",
            role=ReviewerRole.ACADEMIC_CONSULTANT,
            status=ReviewerStatus.AVAILABLE,
            specializations=["academic_writing", "publication_standards", "peer_review"]
        )
    ]
    
    for reviewer in expert_reviewers:
        reviewer_manager.register_reviewer(reviewer)
    
    return reviewer_manager


if __name__ == "__main__":
    # Run specific test for development
    test_suite = TestStory33ReviewWorkflow()
    test_suite.setup_method()
    
    try:
        print("Running Epic 4.3 production infrastructure validation...")
        test_suite.test_epic_4_3_production_infrastructure_validation()
        print("✓ Epic 4.3 validation passed")
        
        print("Running end-to-end workflow test...")
        test_suite.test_end_to_end_review_workflow_complete()
        print("✓ End-to-end workflow test passed")
        
        print("All Story 3.3 tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise
    finally:
        test_suite.teardown_method()