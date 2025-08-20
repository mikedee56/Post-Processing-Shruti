"""
Golden Dataset Validation System for Story 5.5: Testing & Quality Assurance Framework

This module provides comprehensive validation capabilities for golden datasets,
including accuracy measurement, performance benchmarking, and API error investigation.
"""

import json
import logging
import traceback
import tempfile
import time
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
import sys

# Configure logging for error investigation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Represents a validation result for a single test case."""
    test_id: str
    original_text: str
    expected_output: str
    actual_output: str
    passed: bool
    accuracy_score: float
    processing_time: float
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


@dataclass
class APIError:
    """Represents an API error encountered during testing."""
    error_type: str
    error_message: str
    error_code: Optional[str]
    stack_trace: str
    component: str
    timestamp: str
    context: Dict[str, Any]


@dataclass
class ValidationReport:
    """Comprehensive validation report for golden dataset testing."""
    dataset_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_accuracy: float
    average_processing_time: float
    performance_variance: float
    api_errors: List[APIError]
    category_results: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    validation_timestamp: str


class GoldenDatasetValidator:
    """
    Comprehensive golden dataset validation system with API error investigation.
    
    Provides accuracy measurement, performance benchmarking, and diagnostic
    capabilities for the ASR post-processing testing framework.
    """
    
    def __init__(self, data_manager=None):
        """Initialize the golden dataset validator."""
        if data_manager is None:
            from .test_data_manager import get_test_data_manager
            self.data_manager = get_test_data_manager()
        else:
            self.data_manager = data_manager
        
        self.api_errors: List[APIError] = []
        self.performance_metrics: List[float] = []
        
        # Add src to Python path for imports
        src_path = Path("src").absolute()
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        logger.info("GoldenDatasetValidator initialized")
    
    def _capture_api_error(self, error: Exception, component: str, context: Dict[str, Any] = None):
        """Capture and log API errors for investigation."""
        
        api_error = APIError(
            error_type=type(error).__name__,
            error_message=str(error),
            error_code=getattr(error, 'code', None),
            stack_trace=traceback.format_exc(),
            component=component,
            timestamp=datetime.now().isoformat(),
            context=context or {}
        )
        
        self.api_errors.append(api_error)
        logger.error(f"API Error in {component}: {error}")
        logger.debug(f"Stack trace: {api_error.stack_trace}")
        
        return api_error
    
    def _validate_system_dependencies(self) -> Dict[str, Any]:
        """Validate that all system dependencies are working correctly."""
        
        dependency_status = {
            "core_imports": {"status": "unknown", "errors": []},
            "post_processor": {"status": "unknown", "errors": []},
            "text_normalizer": {"status": "unknown", "errors": []},
            "mcp_client": {"status": "unknown", "errors": []},
            "ner_module": {"status": "unknown", "errors": []}
        }
        
        try:
            # Test core imports
            import yaml
            import pysrt
            dependency_status["core_imports"]["status"] = "working"
            logger.info("Core imports validated successfully")
        except Exception as e:
            dependency_status["core_imports"]["status"] = "failed"
            dependency_status["core_imports"]["errors"].append(str(e))
            self._capture_api_error(e, "core_imports")
        
        try:
            # Test post processor import and initialization
            from post_processors.sanskrit_post_processor import SanskritPostProcessor
            processor = SanskritPostProcessor()
            dependency_status["post_processor"]["status"] = "working"
            logger.info("SanskritPostProcessor validated successfully")
        except Exception as e:
            dependency_status["post_processor"]["status"] = "failed"
            dependency_status["post_processor"]["errors"].append(str(e))
            self._capture_api_error(e, "post_processor", {"action": "initialization"})
        
        try:
            # Test text normalizer
            from utils.advanced_text_normalizer import AdvancedTextNormalizer
            normalizer = AdvancedTextNormalizer()
            test_result = normalizer.convert_numbers_with_context("test two three")
            dependency_status["text_normalizer"]["status"] = "working"
            logger.info("AdvancedTextNormalizer validated successfully")
        except Exception as e:
            dependency_status["text_normalizer"]["status"] = "failed"
            dependency_status["text_normalizer"]["errors"].append(str(e))
            self._capture_api_error(e, "text_normalizer", {"action": "test_conversion"})
        
        try:
            # Test MCP client
            from utils.mcp_client import create_mcp_client
            mcp_client = create_mcp_client()
            stats = mcp_client.get_performance_stats()
            dependency_status["mcp_client"]["status"] = "working"
            logger.info("MCP client validated successfully")
        except Exception as e:
            dependency_status["mcp_client"]["status"] = "failed"
            dependency_status["mcp_client"]["errors"].append(str(e))
            self._capture_api_error(e, "mcp_client", {"action": "get_performance_stats"})
        
        try:
            # Test NER module
            from ner_module.yoga_vedanta_ner import YogaVedantaNER
            ner_model = YogaVedantaNER()
            dependency_status["ner_module"]["status"] = "working"
            logger.info("NER module validated successfully")
        except Exception as e:
            dependency_status["ner_module"]["status"] = "failed"
            dependency_status["ner_module"]["errors"].append(str(e))
            self._capture_api_error(e, "ner_module", {"action": "initialization"})
        
        return dependency_status
    
    def validate_dataset_accuracy(self, dataset_name: str) -> ValidationReport:
        """
        Validate golden dataset accuracy with comprehensive error investigation.
        
        Args:
            dataset_name: Name of the golden dataset to validate
            
        Returns:
            ValidationReport with detailed accuracy and error analysis
        """
        
        logger.info(f"Starting validation of golden dataset: {dataset_name}")
        
        # Initialize report
        report = ValidationReport(
            dataset_name=dataset_name,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            overall_accuracy=0.0,
            average_processing_time=0.0,
            performance_variance=0.0,
            api_errors=[],
            category_results={},
            recommendations=[],
            validation_timestamp=datetime.now().isoformat()
        )
        
        try:
            # Validate system dependencies first
            dependency_status = self._validate_system_dependencies()
            failed_dependencies = [
                dep for dep, status in dependency_status.items() 
                if status["status"] == "failed"
            ]
            
            if failed_dependencies:
                report.recommendations.append(
                    f"Critical dependencies failed: {', '.join(failed_dependencies)}. "
                    f"Fix these before proceeding with validation."
                )
                logger.error(f"Failed dependencies detected: {failed_dependencies}")
            
            # Load golden dataset
            try:
                entries = self.data_manager.load_golden_dataset(dataset_name)
                report.total_tests = len(entries)
                logger.info(f"Loaded {len(entries)} golden dataset entries")
            except Exception as e:
                self._capture_api_error(e, "data_manager", {"action": "load_golden_dataset"})
                report.api_errors = self.api_errors
                return report
            
            # Initialize processing components
            try:
                from post_processors.sanskrit_post_processor import SanskritPostProcessor
                processor = SanskritPostProcessor()
                logger.info("Initialized SanskritPostProcessor for validation")
            except Exception as e:
                self._capture_api_error(e, "post_processor", {"action": "initialization"})
                report.api_errors = self.api_errors
                return report
            
            # Process each entry
            validation_results: List[ValidationResult] = []
            processing_times: List[float] = []
            category_stats: Dict[str, Dict[str, int]] = {}
            
            for i, entry in enumerate(entries):
                test_id = f"{dataset_name}_{i:04d}"
                
                try:
                    # Measure processing time
                    start_time = time.perf_counter()
                    
                    # Process the text
                    result = processor.text_normalizer.normalize_with_advanced_tracking(entry.original_text)
                    actual_output = result.corrected_text
                    
                    processing_time = time.perf_counter() - start_time
                    processing_times.append(processing_time)
                    
                    # Calculate accuracy
                    passed = actual_output.strip() == entry.expected_text.strip()
                    accuracy_score = 1.0 if passed else 0.0
                    
                    # Create validation result
                    validation_result = ValidationResult(
                        test_id=test_id,
                        original_text=entry.original_text,
                        expected_output=entry.expected_text,
                        actual_output=actual_output,
                        passed=passed,
                        accuracy_score=accuracy_score,
                        processing_time=processing_time,
                        errors=[],
                        warnings=[],
                        metadata={
                            "category": entry.category,
                            "transformations": entry.transformations,
                            "confidence_score": entry.confidence_score
                        }
                    )
                    
                    validation_results.append(validation_result)
                    
                    # Update report counts
                    if passed:
                        report.passed_tests += 1
                    else:
                        report.failed_tests += 1
                    
                    # Track category statistics
                    category = entry.category
                    if category not in category_stats:
                        category_stats[category] = {"total": 0, "passed": 0}
                    
                    category_stats[category]["total"] += 1
                    if passed:
                        category_stats[category]["passed"] += 1
                    
                    logger.debug(f"Processed entry {i+1}/{len(entries)}: {'PASS' if passed else 'FAIL'}")
                
                except Exception as e:
                    # Capture processing errors
                    self._capture_api_error(e, "text_processing", {
                        "test_id": test_id,
                        "original_text": entry.original_text,
                        "category": entry.category
                    })
                    
                    validation_result = ValidationResult(
                        test_id=test_id,
                        original_text=entry.original_text,
                        expected_output=entry.expected_text,
                        actual_output=f"ERROR: {str(e)}",
                        passed=False,
                        accuracy_score=0.0,
                        processing_time=0.0,
                        errors=[str(e)],
                        warnings=[],
                        metadata={"category": entry.category, "error": True}
                    )
                    
                    validation_results.append(validation_result)
                    report.failed_tests += 1
            
            # Calculate overall metrics
            if report.total_tests > 0:
                report.overall_accuracy = report.passed_tests / report.total_tests
            
            if processing_times:
                report.average_processing_time = statistics.mean(processing_times)
                if len(processing_times) > 1:
                    stdev = statistics.stdev(processing_times)
                    report.performance_variance = (stdev / report.average_processing_time * 100) if report.average_processing_time > 0 else 0.0
            
            # Calculate category results
            for category, stats in category_stats.items():
                report.category_results[category] = {
                    "total": stats["total"],
                    "passed": stats["passed"],
                    "accuracy": stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0
                }
            
            # Generate recommendations
            self._generate_recommendations(report, validation_results, dependency_status)
            
            # Include API errors
            report.api_errors = self.api_errors
            
            logger.info(f"Validation completed: {report.overall_accuracy:.3f} accuracy, {len(self.api_errors)} API errors")
            
        except Exception as e:
            self._capture_api_error(e, "validation_framework", {"dataset_name": dataset_name})
            report.api_errors = self.api_errors
            logger.error(f"Validation framework error: {e}")
        
        return report
    
    def _generate_recommendations(
        self, 
        report: ValidationReport, 
        validation_results: List[ValidationResult],
        dependency_status: Dict[str, Any]
    ):
        """Generate actionable recommendations based on validation results."""
        
        recommendations = []
        
        # Accuracy recommendations
        if report.overall_accuracy < 0.9:
            recommendations.append(
                f"Overall accuracy ({report.overall_accuracy:.3f}) is below target (0.90). "
                f"Review failed test cases and improve processing algorithms."
            )
        
        # Performance recommendations
        if report.performance_variance > 10.0:
            recommendations.append(
                f"Performance variance ({report.performance_variance:.1f}%) exceeds target (10%). "
                f"Investigate processing consistency and optimization opportunities."
            )
        
        # Category-specific recommendations
        for category, results in report.category_results.items():
            if results["accuracy"] < 0.85:
                recommendations.append(
                    f"Category '{category}' accuracy ({results['accuracy']:.3f}) needs improvement. "
                    f"Focus on {category}-specific processing enhancements."
                )
        
        # API error recommendations
        if self.api_errors:
            error_types = set(error.error_type for error in self.api_errors)
            recommendations.append(
                f"API errors detected: {', '.join(error_types)}. "
                f"Review error details and fix underlying issues."
            )
        
        # Dependency recommendations
        failed_deps = [
            dep for dep, status in dependency_status.items() 
            if status["status"] == "failed"
        ]
        if failed_deps:
            recommendations.append(
                f"Failed dependencies: {', '.join(failed_deps)}. "
                f"Ensure all required components are properly installed and configured."
            )
        
        # Performance threshold recommendations
        if report.average_processing_time > 0.1:  # 100ms per text
            recommendations.append(
                f"Average processing time ({report.average_processing_time:.4f}s) may impact performance. "
                f"Consider optimization or parallel processing."
            )
        
        report.recommendations = recommendations
    
    def investigate_api_errors(self) -> Dict[str, Any]:
        """
        Investigate and analyze API errors for diagnostic purposes.
        
        Returns:
            Comprehensive error analysis report
        """
        
        error_analysis = {
            "total_errors": len(self.api_errors),
            "error_types": {},
            "error_components": {},
            "error_timeline": [],
            "common_patterns": [],
            "suggested_fixes": []
        }
        
        if not self.api_errors:
            logger.info("No API errors to investigate")
            return error_analysis
        
        # Analyze error types
        for error in self.api_errors:
            error_type = error.error_type
            if error_type not in error_analysis["error_types"]:
                error_analysis["error_types"][error_type] = {
                    "count": 0,
                    "components": set(),
                    "messages": set()
                }
            
            error_analysis["error_types"][error_type]["count"] += 1
            error_analysis["error_types"][error_type]["components"].add(error.component)
            error_analysis["error_types"][error_type]["messages"].add(error.error_message[:100])
        
        # Convert sets to lists for JSON serialization
        for error_type, data in error_analysis["error_types"].items():
            data["components"] = list(data["components"])
            data["messages"] = list(data["messages"])
        
        # Analyze error components
        for error in self.api_errors:
            component = error.component
            if component not in error_analysis["error_components"]:
                error_analysis["error_components"][component] = {
                    "count": 0,
                    "error_types": set()
                }
            
            error_analysis["error_components"][component]["count"] += 1
            error_analysis["error_components"][component]["error_types"].add(error.error_type)
        
        # Convert sets to lists
        for component, data in error_analysis["error_components"].items():
            data["error_types"] = list(data["error_types"])
        
        # Create error timeline
        for error in self.api_errors:
            error_analysis["error_timeline"].append({
                "timestamp": error.timestamp,
                "component": error.component,
                "error_type": error.error_type,
                "message": error.error_message[:50] + "..." if len(error.error_message) > 50 else error.error_message
            })
        
        # Identify common patterns
        if "ImportError" in error_analysis["error_types"]:
            error_analysis["common_patterns"].append("Import errors detected - check dependency installation")
        
        if "AttributeError" in error_analysis["error_types"]:
            error_analysis["common_patterns"].append("Attribute errors detected - check API compatibility")
        
        if "ConnectionError" in error_analysis["error_types"]:
            error_analysis["common_patterns"].append("Connection errors detected - check network/service availability")
        
        # Generate suggested fixes
        if error_analysis["error_components"].get("mcp_client", {}).get("count", 0) > 0:
            error_analysis["suggested_fixes"].append(
                "MCP client errors: Verify MCP server is running and accessible"
            )
        
        if error_analysis["error_components"].get("post_processor", {}).get("count", 0) > 0:
            error_analysis["suggested_fixes"].append(
                "Post processor errors: Check SanskritPostProcessor configuration and dependencies"
            )
        
        if error_analysis["error_components"].get("text_normalizer", {}).get("count", 0) > 0:
            error_analysis["suggested_fixes"].append(
                "Text normalizer errors: Verify AdvancedTextNormalizer initialization and MCP integration"
            )
        
        logger.info(f"API error investigation completed: {len(self.api_errors)} errors analyzed")
        return error_analysis
    
    def save_validation_report(self, report: ValidationReport, output_path: Optional[Path] = None):
        """Save validation report to file with error details."""
        
        if output_path is None:
            output_path = Path("tests/data/validation_reports")
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / f"validation_report_{report.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Convert report to dictionary
            report_data = asdict(report)
            
            # Include error analysis
            error_analysis = self.investigate_api_errors()
            report_data["error_analysis"] = error_analysis
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Validation report saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")
            raise
    
    def run_comprehensive_validation(self, dataset_names: Optional[List[str]] = None) -> Dict[str, ValidationReport]:
        """
        Run comprehensive validation across multiple datasets.
        
        Args:
            dataset_names: List of dataset names to validate. If None, validates all available datasets.
            
        Returns:
            Dictionary mapping dataset names to validation reports
        """
        
        if dataset_names is None:
            # Get all available datasets
            stats = self.data_manager.get_dataset_statistics()
            dataset_names = [
                detail["name"] for detail in stats["dataset_details"]
                if detail["category"] == "golden"
            ]
        
        validation_reports = {}
        
        for dataset_name in dataset_names:
            logger.info(f"Validating dataset: {dataset_name}")
            
            try:
                report = self.validate_dataset_accuracy(dataset_name)
                validation_reports[dataset_name] = report
                
                # Save individual report
                self.save_validation_report(report)
                
            except Exception as e:
                logger.error(f"Failed to validate dataset {dataset_name}: {e}")
                self._capture_api_error(e, "comprehensive_validation", {"dataset_name": dataset_name})
        
        logger.info(f"Comprehensive validation completed for {len(validation_reports)} datasets")
        return validation_reports


# Utility functions for testing framework integration
def validate_golden_dataset(dataset_name: str) -> ValidationReport:
    """Convenience function to validate a golden dataset."""
    validator = GoldenDatasetValidator()
    return validator.validate_dataset_accuracy(dataset_name)


def investigate_system_errors() -> Dict[str, Any]:
    """Convenience function to investigate system-wide API errors."""
    validator = GoldenDatasetValidator()
    dependency_status = validator._validate_system_dependencies()
    error_analysis = validator.investigate_api_errors()
    
    return {
        "dependency_status": dependency_status,
        "error_analysis": error_analysis,
        "investigation_timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Run validation and error investigation
    print("Running Golden Dataset Validation and API Error Investigation...")
    
    # Initialize validator
    validator = GoldenDatasetValidator()
    
    # Investigate system dependencies and errors
    investigation_results = investigate_system_errors()
    
    print(f"Dependency Status:")
    for dep, status in investigation_results["dependency_status"].items():
        print(f"  {dep}: {status['status']}")
        if status["errors"]:
            print(f"    Errors: {status['errors']}")
    
    print(f"\nAPI Errors: {investigation_results['error_analysis']['total_errors']}")
    
    if investigation_results["error_analysis"]["total_errors"] > 0:
        print("Error Types:")
        for error_type, data in investigation_results["error_analysis"]["error_types"].items():
            print(f"  {error_type}: {data['count']} occurrences")
        
        print("Suggested Fixes:")
        for fix in investigation_results["error_analysis"]["suggested_fixes"]:
            print(f"  - {fix}")