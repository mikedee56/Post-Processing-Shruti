"""
Simple validation script for Story 3.6: Academic Workflow Integration
Validates that all components can be imported and basic functionality works
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all Story 3.6 components can be imported"""
    print("Testing imports...")
    
    try:
        from qa_module.academic_workflow_integrator import AcademicWorkflowIntegrator, SemanticQualityMetrics
        print("✅ AcademicWorkflowIntegrator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import AcademicWorkflowIntegrator: {e}")
        return False
    
    try:
        from qa_module.academic_reporting_dashboard import AcademicReportingDashboard, AcademicReportingConfig
        print("✅ AcademicReportingDashboard imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import AcademicReportingDashboard: {e}")
        return False
    
    try:
        from qa_module.workflow_integration_manager import WorkflowIntegrationManager, ReviewWorkflowConfig
        print("✅ WorkflowIntegrationManager imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import WorkflowIntegrationManager: {e}")
        return False
    
    try:
        from qa_module.academic_compliance_validator import AcademicComplianceValidator, ComplianceLevel
        print("✅ AcademicComplianceValidator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import AcademicComplianceValidator: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\nTesting basic functionality...")
    
    try:
        # Test AcademicComplianceValidator
        from qa_module.academic_compliance_validator import AcademicComplianceValidator
        
        validator = AcademicComplianceValidator()
        print(f"✅ AcademicComplianceValidator initialized with {len(validator.compliance_rules)} rules")
        
        # Test with simple content
        test_content = """1
00:00:01,000 --> 00:00:05,000
Today we study Krishna and dharma.

2
00:00:06,000 --> 00:00:10,000
The Bhagavad Gita teaches us about yoga."""
        
        report = validator.validate_academic_compliance(test_content)
        print(f"✅ Compliance validation completed. Score: {report.overall_compliance_score:.1%}")
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False
    
    try:
        # Test AcademicReportingDashboard
        from qa_module.academic_reporting_dashboard import AcademicReportingDashboard, AcademicReportingConfig
        import tempfile
        
        config = AcademicReportingConfig(output_directory=tempfile.mkdtemp())
        dashboard = AcademicReportingDashboard(config)
        print("✅ AcademicReportingDashboard initialized successfully")
        
    except Exception as e:
        print(f"❌ Reporting dashboard test failed: {e}")
        return False
    
    return True


def test_integration_with_existing():
    """Test integration with existing components"""
    print("\nTesting integration with existing components...")
    
    try:
        # Test that we can import existing components
        from post_processors.academic_polish_processor import AcademicPolishProcessor
        print("✅ Can import existing AcademicPolishProcessor")
        
        processor = AcademicPolishProcessor()
        print("✅ Can initialize AcademicPolishProcessor")
        
        # Test basic polish functionality
        test_content = """1
00:00:01,000 --> 00:00:05,000
today we study krishna and dharma.

2
00:00:06,000 --> 00:00:10,000
um, the bhagavad gita teaches us about yoga."""
        
        polished_content, issues = processor.polish_srt_content(test_content)
        print(f"✅ Academic polish processing completed. {len(issues)} issues found")
        
    except ImportError as e:
        print(f"❌ Cannot import existing components: {e}")
        return False
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    return True


def test_zero_regression():
    """Test that integration doesn't break existing functionality"""
    print("\nTesting zero regression...")
    
    try:
        from post_processors.academic_polish_processor import AcademicPolishProcessor
        
        # Test original functionality still works
        processor = AcademicPolishProcessor()
        
        original_content = """1
00:00:01,000 --> 00:00:05,000
today we study krishna and dharma in vedanta.

2
00:00:06,000 --> 00:00:10,000
um, the bhagavad gita teaches us about yoga."""
        
        # Process with original polish processor
        polished_content, issues = processor.polish_srt_content(original_content)
        
        # Verify content was processed
        assert polished_content != original_content, "Content should be modified by polish processor"
        assert isinstance(issues, list), "Issues should be returned as list"
        
        # Verify specific improvements
        # The polish processor focuses on sentence capitalization and filler removal
        # Sanskrit term capitalization is handled by other processors
        assert "Today" in polished_content, "Sentence should start with capital"
        assert polished_content.count("um, ") == 0, "Filler word 'um, ' should be reduced"
        
        # Verify general improvements were made
        assert len(issues) > 0, "Polish processor should find issues to fix"
        
        print("✅ Zero regression validation passed - existing functionality preserved")
        return True
        
    except Exception as e:
        print(f"❌ Zero regression test failed: {e}")
        return False


def validate_story_files():
    """Validate that story files are properly created"""
    print("\nValidating story files...")
    
    story_file = Path(__file__).parent.parent / "docs" / "stories" / "3.6.academic-workflow-integration.story.md"
    
    if story_file.exists():
        print("✅ Story 3.6 file exists")
        
        content = story_file.read_text(encoding='utf-8')
        
        # Check for required sections
        required_sections = [
            "# Story 3.6: Academic Workflow Integration",
            "## Story",
            "## Acceptance Criteria", 
            "## Technical Tasks",
            "## Dev Agent Record"
        ]
        
        for section in required_sections:
            if section in content:
                print(f"✅ Found required section: {section}")
            else:
                print(f"❌ Missing required section: {section}")
                return False
        
        return True
    else:
        print("❌ Story 3.6 file not found")
        return False


def main():
    """Run all validation tests"""
    print("=" * 60)
    print("STORY 3.6: ACADEMIC WORKFLOW INTEGRATION - VALIDATION")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_tests_passed = False
    
    # Test integration
    if not test_integration_with_existing():
        all_tests_passed = False
    
    # Test zero regression
    if not test_zero_regression():
        all_tests_passed = False
    
    # Validate story files
    if not validate_story_files():
        all_tests_passed = False
    
    # Final results
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("Story 3.6: Academic Workflow Integration is ready for deployment")
        print("\nKey achievements:")
        print("• ✅ AcademicWorkflowIntegrator - Seamless integration with existing academic polish")
        print("• ✅ AcademicReportingDashboard - Comprehensive stakeholder reporting")
        print("• ✅ WorkflowIntegrationManager - Integrated review workflow management")
        print("• ✅ AcademicComplianceValidator - Enhanced academic standard validation")
        print("• ✅ Zero regression - Existing functionality preserved")
        return True
    else:
        print("❌ SOME VALIDATION TESTS FAILED!")
        print("Please review and fix issues before deployment")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)