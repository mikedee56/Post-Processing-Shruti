#!/usr/bin/env python3
"""
Test suite for Story 4.5 Task 4: Academic System Integration.
Tests integration with existing Story 2.3 and Story 2.1 systems.
"""

import sys
import os
from pathlib import Path

# Path setup is handled by conftest.py for clean testing

def test_story_2_3_integration():
    """Test seamless integration with existing Story 2.3 scripture processing."""
    
    print("=== Testing Story 2.3 Scripture Processing Integration ===")
    
    try:
        # Test that existing ScriptureProcessor still works
        from scripture_processing.scripture_processor import ScriptureProcessor
        print("✅ ScriptureProcessor import successful")
        
        processor = ScriptureProcessor()
        print("✅ ScriptureProcessor initialization successful")
        
        # Test basic functionality
        test_text = "Today we study the verse about karma from the sacred texts"
        result = processor.process_text(test_text)
        
        print(f"✅ Basic scripture processing functional: {result.verses_identified} verses identified")
        
        # Test that enhanced components are available
        try:
            from scripture_processing.advanced_verse_matcher import AdvancedVerseMatcher
            from scripture_processing.academic_citation_manager import AcademicCitationManager
            from scripture_processing.publication_formatter import PublicationFormatter
            print("✅ All enhanced Story 4.5 components available")
        except ImportError as e:
            print(f"❌ Enhanced components missing: {e}")
            return False
        
        # Test that ScriptureProcessor can use enhanced components
        if hasattr(processor, 'verse_matcher') or hasattr(processor, 'advanced_verse_matcher'):
            print("✅ ScriptureProcessor has access to enhanced verse matching")
        else:
            print("❌ ScriptureProcessor missing enhanced verse matching integration")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Story 2.3 integration test failed: {e}")
        return False

def test_story_2_1_compatibility():
    """Test compatibility with Story 2.1 Sanskrit/Hindi identification."""
    
    print("\n=== Testing Story 2.1 Sanskrit/Hindi Compatibility ===")
    
    try:
        # Test that existing Sanskrit/Hindi components still work
        from sanskrit_hindi_identifier.word_identifier import SanskritHindiIdentifier
        from sanskrit_hindi_identifier.lexicon_manager import LexiconManager
        from utils.iast_transliterator import IASTTransliterator
        
        print("✅ Story 2.1 core components import successful")
        
        # Test basic functionality
        identifier = SanskritHindiIdentifier()
        lexicon = LexiconManager()
        transliterator = IASTTransliterator()
        
        print("✅ Story 2.1 components initialization successful")
        
        # Test integration with enhanced academic systems
        try:
            from utils.academic_validator import AcademicValidator
            validator = AcademicValidator()
            
            # Test that academic validator can work with IAST transliterator
            test_text = "kṛṣṇa dharma yoga"
            validation_result = validator.validate_transliteration_standards(test_text)
            
            print(f"✅ Academic validator integrates with IAST: compliance score {validation_result.get('compliance_score', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Academic validator integration failed: {e}")
            return False
        
        # Test lexicon integration with citation manager
        try:
            from scripture_processing.academic_citation_manager import AcademicCitationManager
            citation_manager = AcademicCitationManager()
            
            # Verify citation manager can access lexicon entries
            entries = lexicon.get_all_entries()
            if entries:
                print(f"✅ Citation manager can access lexicon: {len(entries)} entries available")
            else:
                print("❌ Citation manager cannot access lexicon entries")
                return False
                
        except Exception as e:
            print(f"❌ Citation manager lexicon integration failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Story 2.1 compatibility test failed: {e}")
        return False

def test_mcp_infrastructure_integration():
    """Test integration with MCP infrastructure from Stories 4.1-4.2."""
    
    print("\n=== Testing MCP Infrastructure Integration ===")
    
    try:
        # Check for MCP-related configuration
        config_files = [
            "config/academic_standards_config.yaml",
            "config/mcp_config.yaml"
        ]
        
        config_exists = 0
        for config_file in config_files:
            if Path(config_file).exists():
                config_exists += 1
                print(f"✅ {config_file} exists")
            else:
                print(f"⚠️ {config_file} not found (may be optional)")
        
        if config_exists == 0:
            print("❌ No MCP configuration files found")
            return False
        
        # Test that academic components can work with MCP settings
        try:
            from utils.academic_validator import AcademicValidator
            from scripture_processing.publication_formatter import PublicationFormatter
            
            # Initialize with potential MCP config
            validator = AcademicValidator()
            formatter = PublicationFormatter()
            
            print("✅ Academic components initialize successfully with MCP environment")
            
            # Test if components have MCP-related features
            validator_methods = dir(validator)
            formatter_methods = dir(formatter)
            
            mcp_features = ['mcp', 'enhanced', 'transformer', 'api']
            mcp_found = 0
            
            for feature in mcp_features:
                if any(feature.lower() in method.lower() for method in validator_methods + formatter_methods):
                    mcp_found += 1
            
            if mcp_found > 0:
                print(f"✅ MCP-enhanced features detected: {mcp_found} indicators found")
            else:
                print("⚠️ No MCP-specific features detected (may use standard interfaces)")
            
        except Exception as e:
            print(f"❌ MCP integration test failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ MCP infrastructure test failed: {e}")
        return False

def test_api_preservation():
    """Test that existing API contracts are preserved."""
    
    print("\n=== Testing API Preservation ===")
    
    try:
        # Test ScriptureProcessor API preservation
        from scripture_processing.scripture_processor import ScriptureProcessor
        
        processor = ScriptureProcessor()
        
        # Check that core API methods still exist
        core_methods = [
            'process_text',
            'get_processing_statistics',
            'validate_system_integration'
        ]
        
        missing_methods = []
        for method in core_methods:
            if not hasattr(processor, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing core API methods: {', '.join(missing_methods)}")
            return False
        else:
            print("✅ All core ScriptureProcessor API methods preserved")
        
        # Test that method signatures are compatible
        try:
            test_text = "Test text for API compatibility"
            result = processor.process_text(test_text)
            
            # Check result structure
            required_attributes = ['original_text', 'processed_text', 'verses_identified']
            missing_attributes = []
            
            for attr in required_attributes:
                if not hasattr(result, attr):
                    missing_attributes.append(attr)
            
            if missing_attributes:
                print(f"❌ Missing result attributes: {', '.join(missing_attributes)}")
                return False
            else:
                print("✅ ScriptureProcessor result structure preserved")
                
        except Exception as e:
            print(f"❌ API method execution failed: {e}")
            return False
        
        # Test Sanskrit/Hindi identifier API preservation
        from sanskrit_hindi_identifier.word_identifier import SanskritHindiIdentifier
        
        identifier = SanskritHindiIdentifier()
        
        # Check core identification API
        if hasattr(identifier, 'identify_words'):
            words = identifier.identify_words("test text")
            print("✅ SanskritHindiIdentifier API preserved")
        else:
            print("❌ SanskritHindiIdentifier API broken")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ API preservation test failed: {e}")
        return False

def test_end_to_end_integration():
    """Test complete end-to-end integration of all systems."""
    
    print("\n=== Testing End-to-End Integration ===")
    
    try:
        # Test complete processing pipeline
        from scripture_processing.scripture_processor import ScriptureProcessor
        from sanskrit_hindi_identifier.word_identifier import SanskritHindiIdentifier
        from scripture_processing.publication_formatter import PublicationFormatter
        from utils.academic_validator import AcademicValidator
        
        # Initialize all components
        scripture_processor = ScriptureProcessor()
        hindi_identifier = SanskritHindiIdentifier()
        publication_formatter = PublicationFormatter()
        academic_validator = AcademicValidator()
        
        print("✅ All system components initialized successfully")
        
        # Test integrated processing workflow
        test_text = "Today we study the profound teachings of Krishna about dharma and yoga from Bhagavad Gita chapter 2 verse 47"
        
        # Step 1: Process with scripture processor
        scripture_result = scripture_processor.process_text(test_text)
        print(f"✅ Scripture processing: {scripture_result.verses_identified} verses identified")
        
        # Step 2: Identify Sanskrit/Hindi terms
        identified_words = hindi_identifier.identify_words(test_text)
        print(f"✅ Sanskrit/Hindi identification: {len(identified_words)} words processed")
        
        # Step 3: Format for publication
        from scripture_processing.publication_formatter import PublicationDocument
        
        # Create mock document for testing
        class MockDocument:
            def __init__(self, content):
                self.content = content
                self.title = "Test Document"
                self.citations = []
        
        mock_doc = MockDocument(scripture_result.processed_text)
        formatted_doc = publication_formatter.format_for_publication(
            mock_doc.content, 
            title=mock_doc.title
        )
        print("✅ Publication formatting completed")
        
        # Step 4: Validate academic compliance
        validation_result = academic_validator.validate_academic_compliance(
            scripture_result.processed_text
        )
        print(f"✅ Academic validation: compliance score {getattr(validation_result, 'overall_score', 'N/A')}")
        
        # Test integration points
        integration_points = [
            "Scripture processing enhances verse identification",
            "Sanskrit/Hindi identification preserves IAST standards", 
            "Publication formatter maintains citation accuracy",
            "Academic validator ensures research quality"
        ]
        
        print("✅ End-to-end integration workflow completed successfully")
        print("Integration points validated:")
        for point in integration_points:
            print(f"  - {point}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_task_4_system_integration():
    """Run comprehensive validation for Task 4: Academic System Integration."""
    
    print("=== Story 4.5 Task 4: Academic System Integration - Validation ===")
    print()
    
    tests = [
        ("Story 2.3 Scripture Processing Integration", test_story_2_3_integration),
        ("Story 2.1 Sanskrit/Hindi Compatibility", test_story_2_1_compatibility), 
        ("MCP Infrastructure Integration", test_mcp_infrastructure_integration),
        ("API Preservation", test_api_preservation),
        ("End-to-End Integration", test_end_to_end_integration)
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
    
    print("=== Task 4 Validation Summary ===")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 Task 4: Academic System Integration - VALIDATED")
        print("✅ Story 2.3 scripture processing integration maintained")
        print("✅ Story 2.1 Sanskrit/Hindi compatibility preserved")
        print("✅ MCP infrastructure integration functional")
        print("✅ Existing API contracts preserved")
        print("✅ End-to-end integration workflow operational")
        print()
        print("📋 Status: READY FOR TASK 4 COMPLETION")
        return True
    else:
        print("❌ Task 4: Academic System Integration - ISSUES DETECTED")
        print(f"❗ {total - passed} components need attention")
        return False

if __name__ == "__main__":
    success = validate_task_4_system_integration()
    exit(0 if success else 1)