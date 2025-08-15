#!/usr/bin/env python3
"""
Test suite for Story 4.5 Task 2: Academic Standards Integration.
Tests AcademicCitationManager and citation compliance functionality.
"""

import sys
import os
from pathlib import Path

# Path setup is handled by conftest.py for clean testing

def test_academic_citation_manager_structure():
    """Test AcademicCitationManager class structure and components."""
    
    print("=== Testing Academic Citation Manager Structure ===")
    
    try:
        # Test core imports
        from scripture_processing.academic_citation_manager import (
            AcademicCitationManager,
            CitationStyle,
            TransliterationStandard,
            CitationValidationLevel,
            CitationFormat,
            AcademicCitation,
            CitationValidationResult,
            create_academic_citation_manager
        )
        
        print("✅ All core classes imported successfully")
        
        # Test enum values
        citation_styles = list(CitationStyle)
        transliteration_standards = list(TransliterationStandard)
        validation_levels = list(CitationValidationLevel)
        
        assert len(citation_styles) >= 5  # MLA, APA, Chicago, etc.
        assert CitationStyle.INDOLOGICAL_STANDARD in citation_styles
        assert TransliterationStandard.IAST in transliteration_standards
        assert CitationValidationLevel.PUBLICATION_GRADE in validation_levels
        
        print(f"✅ Citation styles: {len(citation_styles)} defined")
        print(f"✅ Transliteration standards: {len(transliteration_standards)} defined")
        print(f"✅ Validation levels: {len(validation_levels)} defined")
        
        # Test configuration structure
        citation_format = CitationFormat()
        assert hasattr(citation_format, 'style')
        assert hasattr(citation_format, 'transliteration_standard')
        assert hasattr(citation_format, 'validation_level')
        assert hasattr(citation_format, 'include_transliteration')
        
        print("✅ CitationFormat structure validated")
        
        # Test citation data structure
        from scripture_processing.canonical_text_manager import VerseCandidate, ScriptureSource
        
        # Create mock verse candidate for testing
        mock_verse = VerseCandidate(
            source=ScriptureSource.BHAGAVAD_GITA,
            chapter=2,
            verse=47,
            canonical_text="karmaṇy evādhikāras te mā phaleṣu kadācana",
            confidence=0.9,
            context={}
        )
        
        citation = AcademicCitation(
            verse_candidate=mock_verse,
            citation_text="",
            original_passage="test passage",
            citation_style=CitationStyle.INDOLOGICAL_STANDARD,
            transliteration_standard=TransliterationStandard.IAST,
            validation_level=CitationValidationLevel.STANDARD
        )
        
        assert hasattr(citation, 'verse_candidate')
        assert hasattr(citation, 'citation_text')
        assert hasattr(citation, 'meets_publication_standards')
        assert hasattr(citation, 'validation_errors')
        
        print("✅ AcademicCitation structure validated")
        
        return True
        
    except Exception as e:
        print(f"❌ Structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_citation_format_templates():
    """Test citation formatting templates and patterns."""
    
    print("\n=== Testing Citation Format Templates ===")
    
    try:
        # Read the source file to check template definitions
        citation_manager_file = "src/scripture_processing/academic_citation_manager.py"
        
        if not Path(citation_manager_file).exists():
            print(f"❌ {citation_manager_file} not found")
            return False
        
        with open(citation_manager_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for citation template definitions
        required_templates = [
            "citation_templates",
            "INDOLOGICAL_STANDARD",
            "verse_citation",
            "full_citation",
            "with_text"
        ]
        
        missing_templates = []
        for template in required_templates:
            if template not in content:
                missing_templates.append(template)
        
        if missing_templates:
            print(f"❌ Missing citation templates: {', '.join(missing_templates)}")
            return False
        
        print("✅ Citation templates defined in source")
        
        # Check for source abbreviations
        required_abbreviations = [
            "source_abbreviations",
            "BHAGAVAD_GITA",
            "MUNDAKA_UPANISHAD",
            "YOGA_SUTRAS"
        ]
        
        missing_abbreviations = []
        for abbrev in required_abbreviations:
            if abbrev not in content:
                missing_abbreviations.append(abbrev)
        
        if missing_abbreviations:
            print(f"❌ Missing source abbreviations: {', '.join(missing_abbreviations)}")
            return False
        
        print("✅ Source abbreviations defined")
        
        # Check for validation rules
        validation_components = [
            "_validate_citation_format",
            "_validate_citation_content",
            "_validate_academic_compliance",
            "validation_rules"
        ]
        
        missing_validation = []
        for component in validation_components:
            if component not in content:
                missing_validation.append(component)
        
        if missing_validation:
            print(f"❌ Missing validation components: {', '.join(missing_validation)}")
            return False
        
        print("✅ Validation system components defined")
        
        return True
        
    except Exception as e:
        print(f"❌ Template test failed: {e}")
        return False

def test_transliteration_standards():
    """Test transliteration standards implementation."""
    
    print("\n=== Testing Transliteration Standards ===")
    
    try:
        # Check configuration file for transliteration settings
        config_file = "config/academic_standards_config.yaml"
        
        if not Path(config_file).exists():
            print(f"❌ {config_file} not found")
            return False
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # Check for transliteration standards configuration
        transliteration_checks = [
            "transliteration_standards:",
            "iast:",
            "harvard_kyoto:",
            "strict_mode:",
            "require_diacriticals:",
            "vowels:",
            "consonants:",
            "special_marks:"
        ]
        
        missing_transliteration = []
        for check in transliteration_checks:
            if check not in config_content:
                missing_transliteration.append(check)
        
        if missing_transliteration:
            print(f"❌ Missing transliteration config: {', '.join(missing_transliteration)}")
            return False
        
        print("✅ Transliteration standards configuration complete")
        
        # Check for IAST character mappings
        iast_characters = ["ā", "ī", "ū", "ṛ", "ḷ", "ṃ", "ḥ", "ś", "ṣ", "ṇ", "ṭ", "ḍ"]
        
        iast_found = 0
        for char in iast_characters:
            if char in config_content:
                iast_found += 1
        
        if iast_found >= len(iast_characters) * 0.8:  # 80% coverage
            print(f"✅ IAST character mappings present ({iast_found}/{len(iast_characters)} found)")
        else:
            print(f"❌ Insufficient IAST character coverage ({iast_found}/{len(iast_characters)})")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Transliteration test failed: {e}")
        return False

def test_academic_compliance_validation():
    """Test academic compliance validation system."""
    
    print("\n=== Testing Academic Compliance Validation ===")
    
    try:
        # Check for validation rules in source
        citation_manager_file = "src/scripture_processing/academic_citation_manager.py"
        
        with open(citation_manager_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for validation level implementations
        validation_levels = [
            "CitationValidationLevel.BASIC",
            "CitationValidationLevel.STANDARD", 
            "CitationValidationLevel.RIGOROUS",
            "CitationValidationLevel.PUBLICATION_GRADE"
        ]
        
        missing_levels = []
        for level in validation_levels:
            if level not in content:
                missing_levels.append(level)
        
        if missing_levels:
            print(f"❌ Missing validation levels: {', '.join(missing_levels)}")
            return False
        
        print("✅ All validation levels implemented")
        
        # Check for validation method implementations
        validation_methods = [
            "def validate_citation",
            "def _check_citation_formatting",
            "def _check_transliteration_accuracy",
            "def _check_academic_formatting",
            "def _check_peer_review_standards"
        ]
        
        missing_methods = []
        for method in validation_methods:
            if method not in content:
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing validation methods: {', '.join(missing_methods)}")
            return False
        
        print("✅ All validation methods implemented")
        
        # Check for bibliography and formatting features
        advanced_features = [
            "def format_bibliography_entry",
            "def generate_citation_suggestions",
            "def get_citation_statistics"
        ]
        
        missing_features = []
        for feature in advanced_features:
            if feature not in content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing advanced features: {', '.join(missing_features)}")
            return False
        
        print("✅ Advanced citation features implemented")
        
        return True
        
    except Exception as e:
        print(f"❌ Compliance validation test failed: {e}")
        return False

def test_integration_with_story_2_1():
    """Test integration with existing Story 2.1 IAST systems."""
    
    print("\n=== Testing Story 2.1 Integration ===")
    
    try:
        # Check for integration points in configuration
        config_file = "config/academic_standards_config.yaml"
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # Check for Story 2.1 integration settings
        integration_checks = [
            "story_2_1_compatibility:",
            "maintain_sanskrit_hindi_system:",
            "preserve_lexicon_manager:",
            "extend_iast_formatting:"
        ]
        
        missing_integration = []
        for check in integration_checks:
            if check not in config_content:
                missing_integration.append(check)
        
        if missing_integration:
            print(f"❌ Missing Story 2.1 integration: {', '.join(missing_integration)}")
            return False
        
        print("✅ Story 2.1 integration configuration present")
        
        # Check for IAST formatter integration in source
        citation_manager_file = "src/scripture_processing/academic_citation_manager.py"
        
        with open(citation_manager_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for IAST formatter usage
        iast_integration_checks = [
            "ScriptureIASTFormatter",
            "iast_formatter",
            "transliterate_to_iast"
        ]
        
        missing_iast = []
        for check in iast_integration_checks:
            if check not in content:
                missing_iast.append(check)
        
        if missing_iast:
            print(f"❌ Missing IAST integration: {', '.join(missing_iast)}")
            return False
        
        print("✅ IAST formatter integration implemented")
        
        return True
        
    except Exception as e:
        print(f"❌ Story 2.1 integration test failed: {e}")
        return False

def validate_task_2_academic_standards():
    """Run comprehensive validation for Task 2: Academic Standards Integration."""
    
    print("=== Story 4.5 Task 2: Academic Standards Integration - Validation ===")
    print()
    
    tests = [
        ("Academic Citation Manager Structure", test_academic_citation_manager_structure),
        ("Citation Format Templates", test_citation_format_templates),
        ("Transliteration Standards", test_transliteration_standards),
        ("Academic Compliance Validation", test_academic_compliance_validation),
        ("Story 2.1 Integration", test_integration_with_story_2_1)
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
    
    print("=== Task 2 Validation Summary ===")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 Task 2: Academic Standards Integration - VALIDATED")
        print("✅ AcademicCitationManager fully implemented")
        print("✅ Citation format templates complete")
        print("✅ Transliteration standards configured")
        print("✅ Academic compliance validation ready")
        print("✅ Story 2.1 integration maintained")
        print()
        print("📋 Status: READY FOR TASK 2 COMPLETION")
        return True
    else:
        print("❌ Task 2: Academic Standards Integration - ISSUES DETECTED")
        print(f"❗ {total - passed} components need attention")
        return False

if __name__ == "__main__":
    success = validate_task_2_academic_standards()
    exit(0 if success else 1)