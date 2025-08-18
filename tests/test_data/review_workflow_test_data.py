"""
Test data utilities for Story 3.3 review workflow testing.

Provides sample data for testing all Epic 4 components:
- Test SRT segments with Sanskrit content
- Test reviewer profiles with different roles and expertise
- Test configuration for Epic 4 components
"""

from typing import List, Dict, Any
from datetime import datetime

from utils.srt_parser import SRTSegment
from review_workflow.reviewer_manager import ReviewerProfile, ReviewerRole, ReviewerStatus


def create_test_srt_segments() -> List[SRTSegment]:
    """Create test SRT segments for review workflow testing."""
    return [
        SRTSegment(
            index=1,
            start_time="00:00:01,000",
            end_time="00:00:05,000",
            text="Today we will discuss, um, the teachings of krishna and dharma from the bhagavad gita."
        ),
        SRTSegment(
            index=2,
            start_time="00:00:06,000", 
            end_time="00:00:10,000",
            text="Uh, this verse speaks about the, the eternal nature of the soul and yoga practice."
        ),
        SRTSegment(
            index=3,
            start_time="00:00:11,000",
            end_time="00:00:15,000", 
            text="In chapter two verse twenty five, we learn about the avyakto yam acintyo yam."
        ),
        SRTSegment(
            index=4,
            start_time="00:00:16,000",
            end_time="00:00:20,000",
            text="The practice of meditation and mindfulness leads to moksha and liberation."
        ),
        SRTSegment(
            index=5,
            start_time="00:00:21,000",
            end_time="00:00:25,000",
            text="Great teachers like shankaracharya and swami vivekananda have guided seekers."
        )
    ]


def create_test_reviewer_profiles() -> List[ReviewerProfile]:
    """Create test reviewer profiles with different roles and expertise."""
    return [
        ReviewerProfile(
            reviewer_id="gp_001",
            name="General Proofreader 1",
            email="gp1@example.com",
            role=ReviewerRole.GENERAL_PROOFREADER,
            status=ReviewerStatus.AVAILABLE,
            specializations=["grammar", "punctuation", "basic_corrections"],
            languages=["english"],
            availability_hours="09:00-17:00",
            current_workload=0,
            max_concurrent_sessions=3,
            performance_rating=4.2,
            total_reviews_completed=150
        ),
        ReviewerProfile(
            reviewer_id="gp_002", 
            name="General Proofreader 2",
            email="gp2@example.com",
            role=ReviewerRole.GENERAL_PROOFREADER,
            status=ReviewerStatus.AVAILABLE,
            specializations=["grammar", "style", "readability"],
            languages=["english"],
            availability_hours="10:00-18:00",
            current_workload=1,
            max_concurrent_sessions=3,
            performance_rating=4.0,
            total_reviews_completed=89
        ),
        ReviewerProfile(
            reviewer_id="sme_001",
            name="Sanskrit Subject Matter Expert",
            email="sme.sanskrit@example.com", 
            role=ReviewerRole.SUBJECT_MATTER_EXPERT,
            status=ReviewerStatus.AVAILABLE,
            specializations=["sanskrit", "hindi", "iast", "transliteration", "yoga_vedanta"],
            languages=["english", "sanskrit", "hindi"],
            availability_hours="08:00-16:00",
            current_workload=0,
            max_concurrent_sessions=2,
            performance_rating=4.8,
            total_reviews_completed=75,
            expertise_areas=["vedantic_philosophy", "bhagavad_gita", "upanishads"]
        ),
        ReviewerProfile(
            reviewer_id="sme_002",
            name="Academic Writing Expert", 
            email="sme.academic@example.com",
            role=ReviewerRole.SUBJECT_MATTER_EXPERT,
            status=ReviewerStatus.AVAILABLE,
            specializations=["academic_writing", "citation_standards", "publication_formatting"],
            languages=["english"],
            availability_hours="09:00-17:00",
            current_workload=0,
            max_concurrent_sessions=2,
            performance_rating=4.6,
            total_reviews_completed=42,
            expertise_areas=["academic_standards", "peer_review", "publication_quality"]
        ),
        ReviewerProfile(
            reviewer_id="consultant_001",
            name="Senior Academic Consultant",
            email="consultant@example.com",
            role=ReviewerRole.ACADEMIC_CONSULTANT,
            status=ReviewerStatus.AVAILABLE,
            specializations=["academic_validation", "research_standards", "iast_certification", "publication_review"],
            languages=["english", "sanskrit", "hindi"],
            availability_hours="10:00-14:00",
            current_workload=0,
            max_concurrent_sessions=1,
            performance_rating=4.9,
            total_reviews_completed=28,
            expertise_areas=["vedic_studies", "academic_publishing", "iast_standards"],
            academic_credentials=["PhD Indology", "IAST Certification", "Sanskrit Scholar"]
        ),
        ReviewerProfile(
            reviewer_id="admin_001",
            name="Review Administrator",
            email="admin@example.com",
            role=ReviewerRole.REVIEW_ADMINISTRATOR,
            status=ReviewerStatus.AVAILABLE,
            specializations=["workflow_management", "quality_control", "escalation_handling"],
            languages=["english"],
            availability_hours="08:00-18:00",
            current_workload=0,
            max_concurrent_sessions=5,
            performance_rating=4.7,
            total_reviews_completed=200
        )
    ]


def create_test_config() -> Dict[str, Any]:
    """Create test configuration for Epic 4 components."""
    return {
        'production': {
            'max_response_time_ms': 500.0,
            'target_uptime_percentage': 99.9,
            'health_check_interval_seconds': 5,  # Faster for testing
            'performance_metrics_interval_seconds': 10,  # Faster for testing
            'circuit_breaker_enabled': True,
            'adaptive_rate_limiting': True,
            'automatic_failover': True,
            'graceful_degradation': True,
            'enterprise_monitoring': True
        },
        'reviewer_manager': {
            'assignment_algorithm': 'optimal_matching',
            'load_balancing_enabled': True,
            'performance_tracking_enabled': True,
            'epic_4_3_reliability_enabled': True,
            'max_assignment_time_ms': 250.0,
            'workload_distribution_strategy': 'even_distribution'
        },
        'collaborative_interface': {
            'max_concurrent_users': 10,
            'comment_thread_max_depth': 5,
            'auto_save_interval_seconds': 15,  # Faster for testing
            'conflict_resolution_timeout_seconds': 60,  # Shorter for testing
            'academic_validation_required': True,
            'citation_suggestions_enabled': True
        },
        'publication_standards': {
            'default_quality_threshold': 0.7,
            'iast_compliance_required': True,
            'academic_citation_validation': True,
            'publication_tier_thresholds': {
                'draft': 0.5,
                'professional': 0.7,
                'academic': 0.8,
                'peer_reviewed': 0.9,
                'publication_ready': 0.95
            }
        },
        'validation': {
            'response_time_target_ms': 500.0,
            'uptime_target_percentage': 99.9,
            'load_test_duration_seconds': 5,  # Shorter for testing
            'concurrent_request_count': 10,  # Smaller for testing
            'stress_test_multiplier': 2,
            'reliability_test_iterations': 20  # Fewer for testing
        },
        'monitoring': {
            'metrics_collection_enabled': True,
            'performance_tracking_enabled': True,
            'health_check_enabled': True,
            'telemetry_enabled': True
        },
        'performance': {
            'monitoring_enabled': True,
            'response_time_tracking': True,
            'throughput_tracking': True
        },
        'telemetry': {
            'event_collection_enabled': True,
            'performance_metrics_enabled': True,
            'system_health_tracking': True
        },
        'health_checker': {
            'check_interval_seconds': 5,  # Faster for testing
            'failure_threshold': 3,
            'recovery_timeout_seconds': 10  # Shorter for testing
        },
        'citation_manager': {
            'citation_database_enabled': True,
            'automatic_citation_detection': True,
            'academic_standard_validation': True
        },
        'publication_formatter': {
            'iast_formatting_enabled': True,
            'academic_standards_compliance': True,
            'citation_formatting_enabled': True
        }
    }


def create_test_comments_data() -> List[Dict[str, Any]]:
    """Create test comment data for collaborative interface testing."""
    return [
        {
            'comment_type': 'correction',
            'comment_text': 'Capitalization needed for proper nouns',
            'highlighted_text': 'krishna',
            'suggested_text': 'Krishna',
            'author_role': 'gp'
        },
        {
            'comment_type': 'suggestion',
            'comment_text': 'Consider IAST transliteration for Sanskrit terms',
            'highlighted_text': 'krishna',
            'suggested_text': 'Kṛṣṇa',
            'author_role': 'sme'
        },
        {
            'comment_type': 'academic_note',
            'comment_text': 'IAST standard requires diacritics for accurate Sanskrit representation',
            'highlighted_text': 'krishna dharma yoga',
            'suggested_text': 'Kṛṣṇa dharma yoga',
            'author_role': 'consultant'
        },
        {
            'comment_type': 'citation_reference',
            'comment_text': 'Reference: Bhagavad Gita 2.47 (Easwaran translation, 2007)',
            'highlighted_text': 'karma yoga',
            'author_role': 'consultant'
        },
        {
            'comment_type': 'question',
            'comment_text': 'Should we expand "two verse twenty five" to "2 verse 25"?',
            'highlighted_text': 'two verse twenty five',
            'author_role': 'gp'
        }
    ]


def create_test_actions_data() -> List[Dict[str, Any]]:
    """Create test action data for collaborative interface testing."""
    return [
        {
            'action_type': 'text_edit',
            'description': 'Corrected capitalization of proper nouns',
            'original_text': 'krishna and dharma',
            'modified_text': 'Krishna and dharma',
            'performer_role': 'gp'
        },
        {
            'action_type': 'formatting_change',
            'description': 'Applied IAST transliteration',
            'original_text': 'Krishna',
            'modified_text': 'Kṛṣṇa',
            'performer_role': 'sme',
            'academic_justification': 'IAST standard for Sanskrit transliteration'
        },
        {
            'action_type': 'academic_validation',
            'description': 'Validated Sanskrit terminology and IAST compliance',
            'original_text': 'krishna dharma yoga from bhagavad gita',
            'modified_text': 'Kṛṣṇa dharma yoga from Bhagavad Gītā',
            'performer_role': 'consultant',
            'academic_justification': 'Applied IAST diacritics and proper noun capitalization per academic standards'
        },
        {
            'action_type': 'citation_add',
            'description': 'Added scriptural reference',
            'original_text': 'karma yoga practice',
            'modified_text': 'karma yoga practice (Bhagavad Gītā 2.47)',
            'performer_role': 'consultant',
            'academic_justification': 'Added primary source citation for verification'
        }
    ]


def create_mock_epic_2_processed_content() -> List[SRTSegment]:
    """Create mock Epic 2 Sanskrit-processed content for integration testing."""
    return [
        SRTSegment(
            index=1,
            start_time="00:00:01,000",
            end_time="00:00:05,000",
            text="Today we will discuss the teachings of Kṛṣṇa and dharma from the Bhagavad Gītā."
        ),
        SRTSegment(
            index=2,
            start_time="00:00:06,000",
            end_time="00:00:10,000",
            text="This verse speaks about the eternal nature of the soul and yoga practice."
        ),
        SRTSegment(
            index=3,
            start_time="00:00:11,000",
            end_time="00:00:15,000",
            text="In chapter 2 verse 25, we learn about the avyakto 'yam acintyō 'yam."
        ),
        SRTSegment(
            index=4,
            start_time="00:00:16,000",
            end_time="00:00:20,000",
            text="The practice of meditation and mindfulness leads to mokṣa and liberation."
        ),
        SRTSegment(
            index=5,
            start_time="00:00:21,000",
            end_time="00:00:25,000",
            text="Great teachers like Śaṅkarācārya and Swami Vivekananda have guided seekers."
        )
    ]


def create_performance_test_scenarios() -> List[Dict[str, Any]]:
    """Create performance test scenarios for Epic 4 component testing."""
    return [
        {
            'scenario_name': 'light_load',
            'concurrent_users': 5,
            'requests_per_user': 10,
            'duration_seconds': 30,
            'expected_response_time_ms': 200.0,
            'expected_success_rate': 99.0
        },
        {
            'scenario_name': 'moderate_load',
            'concurrent_users': 15,
            'requests_per_user': 20,
            'duration_seconds': 60,
            'expected_response_time_ms': 350.0,
            'expected_success_rate': 95.0
        },
        {
            'scenario_name': 'heavy_load',
            'concurrent_users': 30,
            'requests_per_user': 15,
            'duration_seconds': 45,
            'expected_response_time_ms': 500.0,
            'expected_success_rate': 90.0
        },
        {
            'scenario_name': 'stress_test',
            'concurrent_users': 50,
            'requests_per_user': 10,
            'duration_seconds': 30,
            'expected_response_time_ms': 750.0,
            'expected_success_rate': 80.0
        }
    ]


def create_academic_validation_test_cases() -> List[Dict[str, Any]]:
    """Create test cases for Epic 4.5 academic validation."""
    return [
        {
            'test_case': 'iast_compliance_basic',
            'original_text': 'krishna dharma yoga',
            'expected_corrections': ['Kṛṣṇa dharma yoga'],
            'academic_standard': 'iast',
            'expected_score': 0.9
        },
        {
            'test_case': 'proper_noun_capitalization',
            'original_text': 'bhagavad gita upanishads',
            'expected_corrections': ['Bhagavad Gītā Upaniṣads'],
            'academic_standard': 'capitalization',
            'expected_score': 0.8
        },
        {
            'test_case': 'numerical_conversion',
            'original_text': 'chapter two verse twenty five',
            'expected_corrections': ['chapter 2 verse 25'],
            'academic_standard': 'numerical',
            'expected_score': 0.7
        },
        {
            'test_case': 'citation_formatting',
            'original_text': 'as taught in the gita',
            'expected_corrections': ['as taught in the Gītā (Bhagavad Gītā 2.47)'],
            'academic_standard': 'citation',
            'expected_score': 0.9
        },
        {
            'test_case': 'comprehensive_academic',
            'original_text': 'krishna teaches karma yoga in chapter two of bhagavad gita',
            'expected_corrections': ['Kṛṣṇa teaches karma yoga in chapter 2 of Bhagavad Gītā'],
            'academic_standard': 'comprehensive',
            'expected_score': 0.95
        }
    ]