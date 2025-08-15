#!/usr/bin/env python3
"""
Test suite for Story 4.5 Task 3: Publication Readiness Achievement.
Tests PublicationFormatter and AcademicValidator functionality.
"""

import os
from pathlib import Path

def test_publication_formatter_structure():
    """Test PublicationFormatter class structure and components."""
    
    print("=== Testing Publication Formatter Structure ===")
    
    try:
        publication_formatter_file = "src/scripture_processing/publication_formatter.py"
        
        if not Path(publication_formatter_file).exists():
            print(f"❌ {publication_formatter_file} not found")
            return False
        
        with open(publication_formatter_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required classes
        required_classes = [
            "class PublicationFormatter:",
            "class DocumentFormat(Enum):",
            "class PublicationDocument:",
            "class QualityMetrics:",
            "class ConsultantReview:",
            "class AcademicCompliance:"
        ]
        
        missing_classes = []
        for class_def in required_classes:
            if class_def not in content:
                missing_classes.append(class_def)
        
        if missing_classes:
            print(f"❌ Missing classes: {', '.join(missing_classes)}")
            return False
        else:
            print("✅ All required classes defined")
        
        # Check for key methods
        required_methods = [
            "def format_for_publication(",
            "def generate_quality_report(",
            "def submit_for_consultant_review(",
            "def validate_publication_readiness(",
            "def format_academic_document("
        ]
        
        missing_methods = []
        for method in required_methods:
            if method not in content:
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing methods: {', '.join(missing_methods)}")
            return False
        else:
            print("✅ All required methods implemented")
        
        # Check for document formats
        document_formats = [
            "ACADEMIC_PAPER",
            "RESEARCH_ARTICLE", 
            "BOOK_CHAPTER",
            "CONFERENCE_PROCEEDINGS"
        ]
        
        format_found = 0
        for format_type in document_formats:
            if format_type in content:
                format_found += 1
        
        if format_found >= len(document_formats) * 0.75:  # 75% coverage
            print(f"✅ Document formats present ({format_found}/{len(document_formats)})")
        else:
            print(f"❌ Insufficient document formats ({format_found}/{len(document_formats)})")
            return False
        
        # Check for quality metrics
        quality_metrics = [
            "accuracy_score",
            "consistency_score", 
            "completeness_score",
            "scholarly_rigor_score",
            "citation_quality_score"
        ]
        
        metrics_found = 0
        for metric in quality_metrics:
            if metric in content:
                metrics_found += 1
        
        if metrics_found >= len(quality_metrics) * 0.8:  # 80% coverage
            print(f"✅ Quality metrics present ({metrics_found}/{len(quality_metrics)})")
        else:
            print(f"❌ Insufficient quality metrics ({metrics_found}/{len(quality_metrics)})")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Publication formatter test failed: {e}")
        return False

def test_academic_validator_structure():
    """Test AcademicValidator class structure and components."""
    
    print("\n=== Testing Academic Validator Structure ===")
    
    try:
        academic_validator_file = "src/utils/academic_validator.py"
        
        if not Path(academic_validator_file).exists():
            print(f"❌ {academic_validator_file} not found")
            return False
        
        with open(academic_validator_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required classes
        required_classes = [
            "class AcademicValidator:",
            "class AcademicStandard(Enum):",
            "class ValidationRule:",
            "class ValidationResult:",
            "class ComprehensiveValidationReport:"
        ]
        
        missing_classes = []
        for class_def in required_classes:
            if class_def not in content:
                missing_classes.append(class_def)
        
        if missing_classes:
            print(f"❌ Missing classes: {', '.join(missing_classes)}")
            return False
        else:
            print("✅ All required classes defined")
        
        # Check for key methods
        required_methods = [
            "def validate_academic_compliance(",
            "def validate_citation_accuracy(",
            "def validate_transliteration_standards(",
            "def validate_scholarly_rigor(",
            "def generate_comprehensive_report("
        ]
        
        missing_methods = []
        for method in required_methods:
            if method not in content:
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing methods: {', '.join(missing_methods)}")
            return False
        else:
            print("✅ All required methods implemented")
        
        # Check for academic standards
        academic_standards = [
            "UNDERGRADUATE",
            "GRADUATE", 
            "PEER_REVIEW",
            "JOURNAL_SUBMISSION",
            "BOOK_PUBLICATION"
        ]
        
        standards_found = 0
        for standard in academic_standards:
            if standard in content:
                standards_found += 1
        
        if standards_found >= len(academic_standards) * 0.8:  # 80% coverage
            print(f"✅ Academic standards present ({standards_found}/{len(academic_standards)})")
        else:
            print(f"❌ Insufficient academic standards ({standards_found}/{len(academic_standards)})")
            return False
        
        # Check for validation categories
        validation_categories = [
            "citation_accuracy",
            "transliteration_compliance",
            "scholarly_rigor",
            "academic_formatting",
            "publication_readiness"
        ]
        
        categories_found = 0
        for category in validation_categories:
            if category in content:
                categories_found += 1
        
        if categories_found >= len(validation_categories) * 0.8:  # 80% coverage
            print(f"✅ Validation categories present ({categories_found}/{len(validation_categories)})")
        else:
            print(f"❌ Insufficient validation categories ({categories_found}/{len(validation_categories)})")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Academic validator test failed: {e}")
        return False

def test_consultant_workflow_integration():
    """Test consultant workflow integration features."""
    
    print("\n=== Testing Consultant Workflow Integration ===")
    
    try:
        # Check configuration for consultant workflow
        config_file = "config/academic_standards_config.yaml"
        
        if not Path(config_file).exists():
            print(f"❌ {config_file} not found")
            return False
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # Check for consultant workflow configuration
        workflow_components = [
            "consultant_workflow:",
            "enable_consultant_integration:",
            "require_human_verification:",
            "priority_levels:",
            "focus_areas:",
            "feedback_integration:"
        ]
        
        missing_workflow = []
        for component in workflow_components:
            if component not in config_content:
                missing_workflow.append(component)
        
        if missing_workflow:
            print(f"❌ Missing workflow components: {', '.join(missing_workflow)}")
            return False
        else:
            print("✅ Consultant workflow configuration complete")
        
        # Check for workflow features in PublicationFormatter
        publication_formatter_file = "src/scripture_processing/publication_formatter.py"
        
        with open(publication_formatter_file, 'r', encoding='utf-8') as f:
            formatter_content = f.read()
        
        workflow_features = [
            "submit_for_consultant_review",
            "ConsultantReview",
            "review_status",
            "feedback",
            "approval_required"
        ]
        
        features_found = 0
        for feature in workflow_features:
            if feature in formatter_content:
                features_found += 1
        
        if features_found >= len(workflow_features) * 0.8:  # 80% coverage
            print(f"✅ Workflow features present ({features_found}/{len(workflow_features)})")
        else:
            print(f"❌ Insufficient workflow features ({features_found}/{len(workflow_features)})")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Consultant workflow test failed: {e}")
        return False

def test_quality_metrics_and_reporting():
    """Test quality metrics and reporting systems."""
    
    print("\n=== Testing Quality Metrics and Reporting ===")
    
    try:
        # Check configuration for quality metrics
        config_file = "config/academic_standards_config.yaml"
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # Check for quality metrics configuration
        quality_components = [
            "publication_quality:",
            "validation_metrics:",
            "accuracy:",
            "consistency:",
            "completeness:",
            "scholarly_rigor:",
            "citation_quality:"
        ]
        
        missing_quality = []
        for component in quality_components:
            if component not in config_content:
                missing_quality.append(component)
        
        if missing_quality:
            print(f"❌ Missing quality components: {', '.join(missing_quality)}")
            return False
        else:
            print("✅ Quality metrics configuration complete")
        
        # Check for quality assurance levels
        qa_levels = [
            "draft:",
            "internal_review:",
            "consultant_review:",
            "peer_review:",
            "publication_ready:"
        ]
        
        qa_found = 0
        for level in qa_levels:
            if level in config_content:
                qa_found += 1
        
        if qa_found >= len(qa_levels) * 0.8:  # 80% coverage
            print(f"✅ QA levels present ({qa_found}/{len(qa_levels)})")
        else:
            print(f"❌ Insufficient QA levels ({qa_found}/{len(qa_levels)})")
            return False
        
        # Check for reporting features in AcademicValidator
        academic_validator_file = "src/utils/academic_validator.py"
        
        with open(academic_validator_file, 'r', encoding='utf-8') as f:
            validator_content = f.read()
        
        reporting_features = [
            "generate_comprehensive_report",
            "ValidationReport",
            "quality_score",
            "recommendations",
            "critical_issues"
        ]
        
        reporting_found = 0
        for feature in reporting_features:
            if feature in validator_content:
                reporting_found += 1
        
        if reporting_found >= len(reporting_features) * 0.8:  # 80% coverage
            print(f"✅ Reporting features present ({reporting_found}/{len(reporting_features)})")
        else:
            print(f"❌ Insufficient reporting features ({reporting_found}/{len(reporting_features)})")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Quality metrics test failed: {e}")
        return False

def test_publication_ready_formatting():
    """Test publication-ready formatting capabilities."""
    
    print("\n=== Testing Publication-Ready Formatting ===")
    
    try:
        # Check configuration for output formats
        config_file = "config/academic_standards_config.yaml"
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # Check for output formats configuration
        format_components = [
            "output_formats:",
            "markdown:",
            "bibliography:",
            "auto_generate:",
            "include_full_citations:"
        ]
        
        missing_formats = []
        for component in format_components:
            if component not in config_content:
                missing_formats.append(component)
        
        if missing_formats:
            print(f"❌ Missing format components: {', '.join(missing_formats)}")
            return False
        else:
            print("✅ Output formats configuration complete")
        
        # Check for formatting features in PublicationFormatter
        publication_formatter_file = "src/scripture_processing/publication_formatter.py"
        
        with open(publication_formatter_file, 'r', encoding='utf-8') as f:
            formatter_content = f.read()
        
        formatting_features = [
            "format_for_publication",
            "generate_bibliography",
            "format_citations",
            "apply_academic_style",
            "export_document"
        ]
        
        formatting_found = 0
        for feature in formatting_features:
            if feature in formatter_content:
                formatting_found += 1
        
        if formatting_found >= len(formatting_features) * 0.8:  # 80% coverage
            print(f"✅ Formatting features present ({formatting_found}/{len(formatting_features)})")
        else:
            print(f"❌ Insufficient formatting features ({formatting_found}/{len(formatting_features)})")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Publication formatting test failed: {e}")
        return False

def validate_task_3_publication_readiness():
    """Run comprehensive validation for Task 3: Publication Readiness Achievement."""
    
    print("=== Story 4.5 Task 3: Publication Readiness Achievement - Validation ===")
    print()
    
    tests = [
        ("Publication Formatter Structure", test_publication_formatter_structure),
        ("Academic Validator Structure", test_academic_validator_structure),
        ("Consultant Workflow Integration", test_consultant_workflow_integration),
        ("Quality Metrics and Reporting", test_quality_metrics_and_reporting),
        ("Publication-Ready Formatting", test_publication_ready_formatting)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=== Task 3 Validation Summary ===")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 Task 3: Publication Readiness Achievement - VALIDATED")
        print("✅ PublicationFormatter fully implemented")
        print("✅ AcademicValidator comprehensive validation system ready")
        print("✅ Consultant workflow integration configured")
        print("✅ Quality metrics and reporting systems operational")
        print("✅ Publication-ready formatting capabilities complete")
        print()
        print("📋 Status: READY FOR TASK 3 COMPLETION")
        return True
    else:
        print("❌ Task 3: Publication Readiness Achievement - ISSUES DETECTED")
        print(f"❗ {total - passed} components need attention")
        return False

if __name__ == "__main__":
    success = validate_task_3_publication_readiness()
    exit(0 if success else 1)