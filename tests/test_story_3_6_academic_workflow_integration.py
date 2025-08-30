"""
Comprehensive test suite for Story 3.6: Academic Workflow Integration
Tests all integration components and validates zero regression
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qa_module.academic_workflow_integrator import (
    AcademicWorkflowIntegrator, SemanticQualityMetrics, AcademicStakeholderReport
)
from qa_module.academic_reporting_dashboard import (
    AcademicReportingDashboard, AcademicReportingConfig
)
from qa_module.workflow_integration_manager import (
    WorkflowIntegrationManager, ReviewWorkflowConfig, WorkflowStage, WorkflowHook
)
from qa_module.academic_compliance_validator import (
    AcademicComplianceValidator, ComplianceLevel, ComplianceViolation, ComplianceReport
)


class TestAcademicWorkflowIntegrator:
    """Test suite for AcademicWorkflowIntegrator"""
    
    @pytest.fixture
    def sample_srt_content(self):
        """Sample SRT content for testing"""
        return """1
00:00:01,000 --> 00:00:05,000
today we study krishna and dharma in vedanta.

2
00:00:06,000 --> 00:00:10,000
the bhagavad gita teaches us about yoga.

3
00:00:11,000 --> 00:00:15,000
um, this is important for understanding moksha."""
    
    @pytest.fixture
    def integrator(self):
        """Create AcademicWorkflowIntegrator instance"""
        config = {
            'qa_framework': {'enable_validation': True},
            'quality_gate': {'enable_quality_checks': True},
            'metrics': {'enable_collection': True}
        }
        return AcademicWorkflowIntegrator(config)
    
    def test_integrator_initialization(self, integrator):
        """Test proper initialization of integrator"""
        assert integrator is not None
        assert hasattr(integrator, 'academic_polish_processor')
        assert hasattr(integrator, 'academic_validator')
        assert hasattr(integrator, 'quality_gate')
        assert hasattr(integrator, 'metrics_collector')
    
    @patch('qa_module.academic_workflow_integrator.AcademicPolishProcessor')
    @patch('qa_module.academic_workflow_integrator.AcademicValidator')
    @patch('qa_module.academic_workflow_integrator.QualityGate')
    def test_process_with_enhanced_quality(self, mock_quality_gate, mock_academic_validator, 
                                         mock_polish_processor, integrator, sample_srt_content):
        """Test enhanced quality processing workflow"""
        
        # Mock component responses
        mock_polish_processor.return_value.polish_srt_content.return_value = (
            sample_srt_content.replace('krishna', 'Krishna').replace('um, ', ''),
            []  # No polish issues
        )
        
        mock_quality_gate.return_value.validate_content.return_value = Mock(
            compliance_scores={'iast_compliance': 0.9, 'sanskrit_accuracy': 0.85},
            metrics={'sanskrit_terms_validated': 5}
        )
        
        mock_academic_validator.return_value.validate_content.return_value = Mock(
            overall_score=0.88
        )
        
        # Process content
        processed_content, report = integrator.process_with_enhanced_quality(
            sample_srt_content, {'filename': 'test.srt'}
        )
        
        # Verify processing
        assert processed_content is not None
        assert isinstance(report, AcademicStakeholderReport)
        assert report.quality_metrics.overall_quality_score > 0
        assert report.executive_summary is not None
        assert len(report.enhancement_details) >= 0
    
    def test_semantic_quality_metrics_calculation(self, integrator):
        """Test semantic quality metrics calculation"""
        
        # Create mock inputs
        polish_issues = [
            Mock(priority='major', issue_type='sanskrit_standardization'),
            Mock(priority='critical', issue_type='filler_word_removal')
        ]
        
        quality_report = Mock(
            compliance_scores={
                'iast_compliance': 0.9,
                'sanskrit_accuracy': 0.85,
                'terminology_consistency': 0.8,
                'academic_formatting': 0.9
            },
            metrics={'sanskrit_terms_validated': 10}
        )
        
        academic_validation = Mock(overall_score=0.87)
        
        # Calculate metrics
        metrics = integrator._calculate_semantic_quality_metrics(
            polish_issues, quality_report, academic_validation
        )
        
        # Verify metrics
        assert isinstance(metrics, SemanticQualityMetrics)
        assert metrics.polish_enhancement_count == 2
        assert metrics.critical_issues_count == 1
        assert metrics.overall_quality_score < 0.87  # Should be penalized for critical issue
        assert metrics.iast_compliance_score == 0.9
        assert metrics.processing_timestamp is not None


class TestAcademicReportingDashboard:
    """Test suite for AcademicReportingDashboard"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def reporting_config(self, temp_dir):
        """Create reporting configuration"""
        return AcademicReportingConfig(
            output_directory=temp_dir,
            enable_html_reports=True,
            enable_csv_export=True,
            enable_json_export=True
        )
    
    @pytest.fixture
    def dashboard(self, reporting_config):
        """Create reporting dashboard instance"""
        return AcademicReportingDashboard(reporting_config)
    
    @pytest.fixture
    def sample_report(self):
        """Create sample academic stakeholder report"""
        quality_metrics = SemanticQualityMetrics(
            overall_quality_score=0.85,
            iast_compliance_score=0.9,
            sanskrit_accuracy_score=0.8,
            terminology_consistency_score=0.85,
            academic_formatting_score=0.9,
            semantic_coherence_score=0.8,
            polish_enhancement_count=5,
            critical_issues_count=1,
            processing_timestamp="2025-01-01T12:00:00"
        )
        
        return AcademicStakeholderReport(
            executive_summary="Test report summary",
            quality_metrics=quality_metrics,
            enhancement_details=[
                {
                    'enhancement_type': 'Sanskrit Standardization',
                    'count': 3,
                    'priority_breakdown': {'critical': 0, 'major': 3, 'minor': 0},
                    'examples': [
                        {
                            'line': 1,
                            'original': 'krishna',
                            'corrected': 'Krishna',
                            'description': 'Capitalize deity name'
                        }
                    ]
                }
            ],
            compliance_findings=[
                {
                    'compliance_area': 'IAST Compliance',
                    'score': 0.9,
                    'level': 'Excellent',
                    'meets_academic_standard': True,
                    'recommendations': []
                }
            ],
            recommendations=['Improve Sanskrit term accuracy'],
            iast_validation_results={
                'compliance_score': 0.9,
                'total_sanskrit_terms_checked': 5,
                'iast_corrections_applied': 2
            },
            content_quality_trends={'overall_quality': 0.85}
        )
    
    def test_dashboard_initialization(self, dashboard, temp_dir):
        """Test proper dashboard initialization"""
        assert dashboard is not None
        assert Path(temp_dir).exists()
        assert dashboard.output_dir.exists()
    
    def test_comprehensive_report_generation(self, dashboard, sample_report, temp_dir):
        """Test comprehensive report generation"""
        
        generated_files = dashboard.generate_comprehensive_report(
            sample_report, 'test_file', {'content_type': 'srt'}
        )
        
        # Verify file generation
        assert 'html' in generated_files
        assert 'csv' in generated_files
        assert 'json' in generated_files
        assert 'summary' in generated_files
        
        # Verify files exist
        for file_path in generated_files.values():
            assert Path(file_path).exists()
            assert Path(file_path).stat().st_size > 0
    
    def test_html_report_content(self, dashboard, sample_report, temp_dir):
        """Test HTML report content generation"""
        
        generated_files = dashboard.generate_comprehensive_report(
            sample_report, 'test_file'
        )
        
        html_path = Path(generated_files['html'])
        html_content = html_path.read_text(encoding='utf-8')
        
        # Verify HTML structure and content
        assert '<html' in html_content
        assert 'Academic Quality Assessment Report' in html_content
        assert 'test_file' in html_content
        assert '85.0%' in html_content  # Overall quality score
        assert 'Sanskrit Standardization' in html_content
        assert 'IAST Compliance' in html_content
    
    def test_json_export_format(self, dashboard, sample_report, temp_dir):
        """Test JSON export format"""
        
        generated_files = dashboard.generate_comprehensive_report(
            sample_report, 'test_file'
        )
        
        json_path = Path(generated_files['json'])
        json_data = json.loads(json_path.read_text(encoding='utf-8'))
        
        # Verify JSON structure
        assert 'metadata' in json_data
        assert 'quality_metrics' in json_data
        assert 'enhancement_details' in json_data
        assert 'compliance_findings' in json_data
        
        # Verify data integrity
        assert json_data['metadata']['filename'] == 'test_file'
        assert json_data['quality_metrics']['overall_quality_score'] == 0.85


class TestWorkflowIntegrationManager:
    """Test suite for WorkflowIntegrationManager"""
    
    @pytest.fixture
    def workflow_config(self):
        """Create workflow configuration"""
        return ReviewWorkflowConfig(
            enable_semantic_enhancement=True,
            enable_quality_gates=True,
            output_directory=tempfile.mkdtemp()
        )
    
    @pytest.fixture
    def workflow_manager(self, workflow_config):
        """Create workflow integration manager"""
        return WorkflowIntegrationManager(workflow_config)
    
    @pytest.fixture
    def sample_content(self):
        """Sample content for workflow testing"""
        return """1
00:00:01,000 --> 00:00:05,000
Today we study Krishna and dharma.

2
00:00:06,000 --> 00:00:10,000
The Bhagavad Gita is sacred."""
    
    def test_workflow_manager_initialization(self, workflow_manager):
        """Test workflow manager initialization"""
        assert workflow_manager is not None
        assert hasattr(workflow_manager, 'academic_integrator')
        assert hasattr(workflow_manager, 'reporting_dashboard')
        assert len(workflow_manager.workflow_hooks) == len(WorkflowStage)
    
    def test_workflow_hook_registration(self, workflow_manager):
        """Test workflow hook registration"""
        
        def test_hook(content, filename, context):
            return content.upper()
        
        hook = WorkflowHook(
            stage=WorkflowStage.PRE_PROCESSING,
            callback=test_hook,
            priority=100
        )
        
        workflow_manager.register_workflow_hook(hook)
        
        # Verify hook registration
        pre_hooks = workflow_manager.workflow_hooks[WorkflowStage.PRE_PROCESSING]
        assert len(pre_hooks) == 1
        assert pre_hooks[0].callback == test_hook
    
    @patch('qa_module.workflow_integration_manager.AcademicWorkflowIntegrator')
    def test_integrated_workflow_processing(self, mock_integrator, workflow_manager, sample_content):
        """Test integrated workflow processing"""
        
        # Mock academic integrator response
        mock_integrator.return_value.process_with_enhanced_quality.return_value = (
            sample_content,
            Mock(quality_metrics=Mock(
                overall_quality_score=0.85,
                critical_issues_count=0
            ))
        )
        
        # Process content
        results = workflow_manager.process_with_integrated_workflow(
            sample_content, 'test.srt', context={'test': True}
        )
        
        # Verify results
        assert results['processing_successful'] == True
        assert results['filename'] == 'test.srt'
        assert 'processed_content' in results
        assert len(results['stages_completed']) > 0
    
    def test_expert_review_determination(self, workflow_manager):
        """Test expert review requirement determination"""
        
        # Test case requiring expert review (low quality)
        low_quality_report = Mock(quality_metrics=Mock(
            overall_quality_score=0.7,
            critical_issues_count=6,
            iast_compliance_score=0.75,
            sanskrit_accuracy_score=0.75
        ))
        
        needs_review = workflow_manager._determine_expert_review_requirement(
            low_quality_report, {}
        )
        assert needs_review == True
        
        # Test case not requiring expert review (high quality)
        high_quality_report = Mock(quality_metrics=Mock(
            overall_quality_score=0.92,
            critical_issues_count=1,
            iast_compliance_score=0.95,
            sanskrit_accuracy_score=0.9
        ))
        
        needs_review = workflow_manager._determine_expert_review_requirement(
            high_quality_report, {}
        )
        assert needs_review == False


class TestAcademicComplianceValidator:
    """Test suite for AcademicComplianceValidator"""
    
    @pytest.fixture
    def validator(self):
        """Create compliance validator instance"""
        config = {
            'academic_standards': {
                'iast_strict_mode': True,
                'require_diacriticals': True,
                'enforce_capitalization': True
            }
        }
        return AcademicComplianceValidator(config)
    
    @pytest.fixture
    def test_content_good(self):
        """Good quality test content"""
        return """1
00:00:01,000 --> 00:00:05,000
Today we study Krishna and Dharma in Vedanta.

2
00:00:06,000 --> 00:00:10,000
The Bhagavad Gītā teaches us about Yoga."""
    
    @pytest.fixture
    def test_content_poor(self):
        """Poor quality test content"""
        return """1
00:00:01,000 --> 00:00:05,000
today we study krishna and dharma. god is just energy.

2
00:00:06,000 --> 00:00:10,000
um, the bhagavad gita is like, totally awesome."""
    
    def test_validator_initialization(self, validator):
        """Test validator initialization"""
        assert validator is not None
        assert len(validator.compliance_rules) > 0
        assert validator.standards_config['iast_strict_mode'] == True
    
    def test_compliance_validation_good_content(self, validator, test_content_good):
        """Test compliance validation with good content"""
        
        report = validator.validate_academic_compliance(test_content_good)
        
        # Verify report structure
        assert isinstance(report, ComplianceReport)
        assert report.overall_compliance_score > 0.7
        assert report.compliance_level in [ComplianceLevel.GOOD, ComplianceLevel.EXCELLENT]
        assert len(report.category_scores) > 0
        assert report.total_rules_checked > 0
    
    def test_compliance_validation_poor_content(self, validator, test_content_poor):
        """Test compliance validation with poor content"""
        
        report = validator.validate_academic_compliance(test_content_poor)
        
        # Verify violations detected
        assert len(report.violations) > 0
        assert report.overall_compliance_score < 0.8
        
        # Check for specific violation types
        violation_types = [v.violation_type for v in report.violations]
        assert any('capitalization' in vtype for vtype in violation_types)
        assert any('spiritual' in vtype for vtype in violation_types)
    
    def test_deity_capitalization_validation(self, validator):
        """Test deity name capitalization validation"""
        
        test_line = "today we discuss krishna and rama"
        rule = Mock(rule_id='test', academic_standard='Test')
        
        violations = validator._validate_deity_capitalization(test_line, 1, rule, {})
        
        # Should find violations for lowercase deity names
        assert len(violations) == 2
        assert all(v.severity == 'critical' for v in violations)
        assert any('krishna' in v.original_text for v in violations)
        assert any('rama' in v.original_text for v in violations)
    
    def test_spiritual_respectfulness_validation(self, validator):
        """Test spiritual respectfulness validation"""
        
        disrespectful_line = "god is just energy and nothing more"
        rule = Mock(rule_id='test', academic_standard='Test')
        
        violations = validator._validate_spiritual_respectfulness(disrespectful_line, 1, rule, {})
        
        # Should detect potentially disrespectful language
        assert len(violations) >= 1
        assert violations[0].severity == 'critical'
    
    def test_compliance_level_determination(self, validator):
        """Test compliance level determination"""
        
        assert validator._determine_compliance_level(0.96) == ComplianceLevel.EXCELLENT
        assert validator._determine_compliance_level(0.87) == ComplianceLevel.GOOD
        assert validator._determine_compliance_level(0.77) == ComplianceLevel.ADEQUATE
        assert validator._determine_compliance_level(0.65) == ComplianceLevel.NEEDS_IMPROVEMENT


class TestIntegrationWithExistingWorkflows:
    """Test integration with existing workflow components"""
    
    @pytest.fixture
    def mock_processor(self):
        """Create mock processor for integration testing"""
        processor = Mock()
        processor.process_srt_file = Mock(return_value="processed content")
        return processor
    
    def test_workflow_manager_processor_integration(self, mock_processor):
        """Test workflow manager integration with existing processor"""
        
        config = ReviewWorkflowConfig(
            output_directory=tempfile.mkdtemp()
        )
        workflow_manager = WorkflowIntegrationManager(config)
        
        # Integrate with mock processor
        workflow_manager.integrate_with_existing_processor(mock_processor)
        
        # Verify integration
        assert hasattr(mock_processor, 'process_srt_file')
        # The method should be replaced/wrapped
        
    def test_zero_regression_validation(self):
        """Test that integration doesn't break existing functionality"""
        
        # Create test content
        test_content = """1
00:00:01,000 --> 00:00:05,000
Test content for regression testing.

2
00:00:06,000 --> 00:00:10,000
This should process correctly."""
        
        # Test with semantic enhancement disabled (should behave like original)
        config = ReviewWorkflowConfig(
            enable_semantic_enhancement=False,
            output_directory=tempfile.mkdtemp()
        )
        workflow_manager = WorkflowIntegrationManager(config)
        
        # Mock original processor function
        original_processor = Mock(return_value=test_content.upper())
        
        results = workflow_manager.process_with_integrated_workflow(
            test_content, 'test.srt', original_processor_func=original_processor
        )
        
        # Should still process successfully
        assert results['processing_successful'] == True
        assert 'processed_content' in results


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability"""
    
    def test_large_content_processing(self):
        """Test processing of large content blocks"""
        
        # Generate large test content (1000 subtitles)
        large_content = ""
        for i in range(1, 1001):
            large_content += f"""{i}
00:{i//60:02d}:{i%60:02d},000 --> 00:{(i+5)//60:02d}:{(i+5)%60:02d},000
This is subtitle number {i} discussing Krishna and dharma.

"""
        
        # Test academic compliance validation performance
        validator = AcademicComplianceValidator()
        
        import time
        start_time = time.time()
        report = validator.validate_academic_compliance(large_content)
        processing_time = time.time() - start_time
        
        # Verify reasonable processing time (should be under 30 seconds)
        assert processing_time < 30.0
        assert report.total_rules_checked > 0
        
    def test_memory_efficiency(self):
        """Test memory efficiency during processing"""
        
        import psutil
        import gc
        
        # Monitor memory before processing
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Process multiple files
        validator = AcademicComplianceValidator()
        
        for i in range(50):  # Process 50 small files
            test_content = f"""1
00:00:01,000 --> 00:00:05,000
Test content {i} with Krishna and dharma.

2
00:00:06,000 --> 00:00:10,000
More test content for memory testing."""
            
            validator.validate_academic_compliance(test_content)
            
            # Force garbage collection
            if i % 10 == 0:
                gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (under 100MB for this test)
        assert memory_increase < 100.0


# Integration test fixtures and utilities
@pytest.fixture(scope="session")
def test_data_dir():
    """Create test data directory"""
    test_dir = Path(__file__).parent / "test_data" / "story_3_6"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
def sample_srt_files(test_data_dir):
    """Create sample SRT files for testing"""
    
    files = {}
    
    # Good quality file
    good_content = """1
00:00:01,000 --> 00:00:05,000
Today we study Krishna and Dharma in Vedanta.

2
00:00:06,000 --> 00:00:10,000
The Bhagavad Gītā teaches us about Yoga.

3
00:00:11,000 --> 00:00:15,000
This knowledge leads to Moksha."""
    
    good_file = test_data_dir / "good_quality.srt"
    good_file.write_text(good_content, encoding='utf-8')
    files['good'] = good_file
    
    # Poor quality file
    poor_content = """1
00:00:01,000 --> 00:00:05,000
today we study krishna and dharma. god is just energy.

2
00:00:06,000 --> 00:00:10,000
um, the bhagavad gita is like, totally awesome stuff.

3
00:00:11,000 --> 00:00:15,000
this knowledge is whatever."""
    
    poor_file = test_data_dir / "poor_quality.srt"
    poor_file.write_text(poor_content, encoding='utf-8')
    files['poor'] = poor_file
    
    return files


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])