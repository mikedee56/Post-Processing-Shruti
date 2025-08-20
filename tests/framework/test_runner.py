#!/usr/bin/env python3
"""
Enhanced Test Execution Framework for ASR Post-Processing System
Provides comprehensive test automation infrastructure with parallel execution,
smart test discovery, result aggregation, and CI/CD integration.

Part of Story 5.5: Testing & Quality Assurance Framework
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing
from collections import defaultdict
import tempfile

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
import coverage


@dataclass
class TestCategory:
    """Test category configuration."""
    name: str
    pattern: str
    timeout: int = 300  # 5 minutes
    parallel: bool = True
    priority: int = 1  # Lower numbers = higher priority
    requirements: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Individual test result."""
    name: str
    category: str
    status: str  # passed, failed, skipped, error
    duration: float
    error_message: Optional[str] = None
    coverage_data: Optional[Dict[str, Any]] = None
    memory_usage: Optional[int] = None
    artifacts: List[str] = field(default_factory=list)


@dataclass
class TestSuiteResult:
    """Complete test suite results."""
    start_time: datetime
    end_time: datetime
    total_duration: float
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    coverage_percentage: float
    results: List[TestResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)


class TestEnvironmentManager:
    """Manages test environment setup and cleanup."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.temp_dirs = []
        self.processes = []
        self.cleanup_callbacks = []
    
    def setup_test_environment(self) -> Dict[str, Any]:
        """Set up comprehensive test environment."""
        env_config = {}
        
        # Create test directories
        test_data_dir = Path(self.config.get("test_data_dir", "tests/test_data"))
        test_data_dir.mkdir(parents=True, exist_ok=True)
        
        temp_dir = Path(tempfile.mkdtemp(prefix="asr_test_"))
        self.temp_dirs.append(temp_dir)
        
        # Set environment variables
        test_env = {
            "PYTHONPATH": str(Path(__file__).parent.parent.parent / "src"),
            "TEST_MODE": "true",
            "TEST_DATA_DIR": str(test_data_dir),
            "TEMP_DIR": str(temp_dir),
            "LOG_LEVEL": "ERROR",  # Reduce logging noise in tests
        }
        
        # Update environment
        os.environ.update(test_env)
        env_config.update(test_env)
        
        return env_config
    
    def cleanup_test_environment(self):
        """Clean up test environment."""
        # Clean up temporary directories
        import shutil
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Failed to cleanup {temp_dir}: {e}")
        
        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Warning: Cleanup callback failed: {e}")
        
        # Terminate processes
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass


class CoverageManager:
    """Manages code coverage measurement and reporting."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cov = None
        self.coverage_threshold = config.get("coverage_threshold", 0.90)
    
    def start_coverage(self):
        """Start coverage measurement."""
        self.cov = coverage.Coverage(
            source=["src"],
            omit=[
                "*/tests/*",
                "*/test_*",
                "*/__pycache__/*",
                "*/.*",
            ]
        )
        self.cov.start()
    
    def stop_coverage(self) -> Dict[str, Any]:
        """Stop coverage and generate report."""
        if not self.cov:
            return {}
        
        self.cov.stop()
        self.cov.save()
        
        # Generate coverage report
        coverage_data = {
            "percentage": 0.0,
            "covered_lines": 0,
            "total_lines": 0,
            "missing_lines": {},
            "files": {}
        }
        
        try:
            total_lines = 0
            covered_lines = 0
            files_data = {}
            
            for filename in self.cov.get_data().measured_files():
                if not filename.startswith(str(Path(__file__).parent.parent.parent / "src")):
                    continue
                
                analysis = self.cov.analysis2(filename)
                file_total = len(analysis[1]) + len(analysis[3])  # executed + missing
                file_covered = len(analysis[1])  # executed
                
                total_lines += file_total
                covered_lines += file_covered
                
                files_data[filename] = {
                    "total_lines": file_total,
                    "covered_lines": file_covered,
                    "percentage": (file_covered / file_total * 100) if file_total > 0 else 0,
                    "missing_lines": list(analysis[3])
                }
            
            coverage_percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
            
            coverage_data.update({
                "percentage": coverage_percentage,
                "covered_lines": covered_lines,
                "total_lines": total_lines,
                "files": files_data
            })
            
        except Exception as e:
            print(f"Warning: Failed to generate coverage report: {e}")
        
        return coverage_data


class TestDiscovery:
    """Intelligent test discovery and categorization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.categories = self._setup_test_categories()
    
    def _setup_test_categories(self) -> List[TestCategory]:
        """Set up test categories with configuration."""
        return [
            TestCategory(
                name="unit",
                pattern="test_*.py",
                timeout=60,
                parallel=True,
                priority=1
            ),
            TestCategory(
                name="integration",
                pattern="test_*integration*.py",
                timeout=300,
                parallel=True,
                priority=2
            ),
            TestCategory(
                name="performance",
                pattern="test_*performance*.py",
                timeout=600,
                parallel=False,  # Performance tests should not run in parallel
                priority=3
            ),
            TestCategory(
                name="end_to_end",
                pattern="test_*end_to_end*.py",
                timeout=900,
                parallel=False,
                priority=4
            ),
            TestCategory(
                name="quality",
                pattern="test_*quality*.py",
                timeout=180,
                parallel=True,
                priority=2
            )
        ]
    
    def discover_tests(self, test_dir: Path = None) -> Dict[str, List[Path]]:
        """Discover and categorize tests."""
        if test_dir is None:
            test_dir = Path(__file__).parent.parent
        
        categorized_tests = defaultdict(list)
        
        # Find all test files
        test_files = list(test_dir.glob("**/test_*.py"))
        
        # Categorize tests
        for test_file in test_files:
            categorized = False
            
            # Try to match against category patterns
            for category in self.categories:
                if self._matches_pattern(test_file, category.pattern):
                    categorized_tests[category.name].append(test_file)
                    categorized = True
                    break
            
            # Default to unit tests if no match
            if not categorized:
                categorized_tests["unit"].append(test_file)
        
        return dict(categorized_tests)
    
    def _matches_pattern(self, test_file: Path, pattern: str) -> bool:
        """Check if test file matches category pattern."""
        import fnmatch
        return fnmatch.fnmatch(test_file.name, pattern)


class ParallelTestRunner:
    """Parallel test execution with resource management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_workers = min(
            config.get("max_test_workers", multiprocessing.cpu_count()),
            multiprocessing.cpu_count()
        )
        self.results_lock = threading.Lock()
        self.results = []
    
    def run_tests_parallel(self, 
                          categorized_tests: Dict[str, List[Path]],
                          categories: List[TestCategory]) -> List[TestResult]:
        """Run tests in parallel with proper resource management."""
        
        # Sort categories by priority
        sorted_categories = sorted(categories, key=lambda c: c.priority)
        all_results = []
        
        for category in sorted_categories:
            if category.name not in categorized_tests:
                continue
                
            test_files = categorized_tests[category.name]
            if not test_files:
                continue
            
            print(f"Running {category.name} tests ({len(test_files)} files)...")
            
            if category.parallel and len(test_files) > 1:
                # Run in parallel
                category_results = self._run_parallel_category(
                    test_files, category
                )
            else:
                # Run sequentially
                category_results = self._run_sequential_category(
                    test_files, category
                )
            
            all_results.extend(category_results)
        
        return all_results
    
    def _run_parallel_category(self, 
                              test_files: List[Path], 
                              category: TestCategory) -> List[TestResult]:
        """Run category tests in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._run_single_test, test_file, category): test_file
                for test_file in test_files
            }
            
            for future in future_to_file:
                try:
                    result = future.result(timeout=category.timeout)
                    if result:
                        results.append(result)
                except Exception as e:
                    test_file = future_to_file[future]
                    results.append(TestResult(
                        name=str(test_file),
                        category=category.name,
                        status="error",
                        duration=0.0,
                        error_message=str(e)
                    ))
        
        return results
    
    def _run_sequential_category(self, 
                                test_files: List[Path], 
                                category: TestCategory) -> List[TestResult]:
        """Run category tests sequentially."""
        results = []
        
        for test_file in test_files:
            try:
                result = self._run_single_test(test_file, category)
                if result:
                    results.append(result)
            except Exception as e:
                results.append(TestResult(
                    name=str(test_file),
                    category=category.name,
                    status="error",
                    duration=0.0,
                    error_message=str(e)
                ))
        
        return results
    
    def _run_single_test(self, test_file: Path, category: TestCategory) -> Optional[TestResult]:
        """Run a single test file and return results."""
        start_time = time.time()
        
        try:
            # Run pytest on the specific file
            cmd = [
                sys.executable, "-m", "pytest",
                str(test_file),
                "-v",
                "--tb=short",
                f"--timeout={category.timeout}",
                "--json-report",
                f"--json-report-file={test_file.stem}_report.json"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=category.timeout,
                cwd=Path(__file__).parent.parent.parent
            )
            
            duration = time.time() - start_time
            
            # Parse results
            status = "passed"
            error_message = None
            
            if result.returncode != 0:
                if "FAILED" in result.stdout or "FAILED" in result.stderr:
                    status = "failed"
                elif "ERROR" in result.stdout or "ERROR" in result.stderr:
                    status = "error"
                
                error_message = result.stderr if result.stderr else result.stdout
            
            return TestResult(
                name=str(test_file),
                category=category.name,
                status=status,
                duration=duration,
                error_message=error_message
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                name=str(test_file),
                category=category.name,
                status="timeout",
                duration=duration,
                error_message=f"Test timed out after {category.timeout} seconds"
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=str(test_file),
                category=category.name,
                status="error",
                duration=duration,
                error_message=str(e)
            )


class TestReporter:
    """Comprehensive test result reporting and analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get("output_dir", "test_reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_comprehensive_report(self, suite_result: TestSuiteResult) -> Dict[str, str]:
        """Generate comprehensive test reports."""
        reports = {}
        
        # Generate different report formats
        reports["json"] = self._generate_json_report(suite_result)
        reports["html"] = self._generate_html_report(suite_result)
        reports["junit"] = self._generate_junit_report(suite_result)
        reports["summary"] = self._generate_summary_report(suite_result)
        
        return reports
    
    def _generate_json_report(self, suite_result: TestSuiteResult) -> str:
        """Generate JSON report."""
        report_data = {
            "test_suite": {
                "start_time": suite_result.start_time.isoformat(),
                "end_time": suite_result.end_time.isoformat(),
                "duration": suite_result.total_duration,
                "coverage_percentage": suite_result.coverage_percentage
            },
            "summary": {
                "total": suite_result.total_tests,
                "passed": suite_result.passed,
                "failed": suite_result.failed,
                "skipped": suite_result.skipped,
                "errors": suite_result.errors
            },
            "results": [
                {
                    "name": r.name,
                    "category": r.category,
                    "status": r.status,
                    "duration": r.duration,
                    "error_message": r.error_message
                }
                for r in suite_result.results
            ]
        }
        
        json_file = self.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return str(json_file)
    
    def _generate_html_report(self, suite_result: TestSuiteResult) -> str:
        """Generate HTML report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Results Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .error {{ color: orange; }}
                .skipped {{ color: gray; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Test Results Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Duration: {suite_result.total_duration:.2f} seconds</p>
                <p>Coverage: {suite_result.coverage_percentage:.1f}%</p>
                <p><span class="passed">Passed: {suite_result.passed}</span> | 
                   <span class="failed">Failed: {suite_result.failed}</span> | 
                   <span class="error">Errors: {suite_result.errors}</span> | 
                   <span class="skipped">Skipped: {suite_result.skipped}</span></p>
            </div>
            
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Category</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Error</th>
                </tr>
        """
        
        for result in suite_result.results:
            html_content += f"""
                <tr>
                    <td>{result.name}</td>
                    <td>{result.category}</td>
                    <td class="{result.status}">{result.status.upper()}</td>
                    <td>{result.duration:.2f}s</td>
                    <td>{result.error_message or ''}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        html_file = self.output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        return str(html_file)
    
    def _generate_junit_report(self, suite_result: TestSuiteResult) -> str:
        """Generate JUnit XML report for CI/CD integration."""
        import xml.etree.ElementTree as ET
        
        testsuite = ET.Element("testsuite")
        testsuite.set("name", "ASR Post-Processing Tests")
        testsuite.set("tests", str(suite_result.total_tests))
        testsuite.set("failures", str(suite_result.failed))
        testsuite.set("errors", str(suite_result.errors))
        testsuite.set("skipped", str(suite_result.skipped))
        testsuite.set("time", str(suite_result.total_duration))
        
        for result in suite_result.results:
            testcase = ET.SubElement(testsuite, "testcase")
            testcase.set("name", Path(result.name).stem)
            testcase.set("classname", result.category)
            testcase.set("time", str(result.duration))
            
            if result.status == "failed":
                failure = ET.SubElement(testcase, "failure")
                failure.text = result.error_message or "Test failed"
            elif result.status == "error":
                error = ET.SubElement(testcase, "error")
                error.text = result.error_message or "Test error"
            elif result.status == "skipped":
                ET.SubElement(testcase, "skipped")
        
        junit_file = self.output_dir / f"junit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
        tree = ET.ElementTree(testsuite)
        tree.write(junit_file, encoding='utf-8', xml_declaration=True)
        
        return str(junit_file)
    
    def _generate_summary_report(self, suite_result: TestSuiteResult) -> str:
        """Generate human-readable summary report."""
        summary_content = f"""
ASR POST-PROCESSING TEST SUITE RESULTS
=======================================

Execution Summary:
- Start Time: {suite_result.start_time.strftime('%Y-%m-%d %H:%M:%S')}
- End Time: {suite_result.end_time.strftime('%Y-%m-%d %H:%M:%S')}
- Duration: {suite_result.total_duration:.2f} seconds
- Coverage: {suite_result.coverage_percentage:.1f}%

Test Results:
- Total Tests: {suite_result.total_tests}
- Passed: {suite_result.passed} ({suite_result.passed/suite_result.total_tests*100:.1f}%)
- Failed: {suite_result.failed} ({suite_result.failed/suite_result.total_tests*100:.1f}%)
- Errors: {suite_result.errors} ({suite_result.errors/suite_result.total_tests*100:.1f}%)
- Skipped: {suite_result.skipped} ({suite_result.skipped/suite_result.total_tests*100:.1f}%)

Coverage Analysis:
- Target Coverage: {self.config.get('coverage_threshold', 0.90) * 100}%
- Actual Coverage: {suite_result.coverage_percentage:.1f}%
- Status: {'PASSED' if suite_result.coverage_percentage >= self.config.get('coverage_threshold', 0.90) * 100 else 'FAILED'}

Failed Tests:
"""
        
        failed_tests = [r for r in suite_result.results if r.status in ['failed', 'error']]
        if failed_tests:
            for test in failed_tests:
                summary_content += f"- {Path(test.name).stem} ({test.category}): {test.error_message}\n"
        else:
            summary_content += "None\n"
        
        summary_file = self.output_dir / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        return str(summary_file)


class TestAutomationFramework:
    """Main test automation framework orchestrator."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config = self._load_config(config_file)
        self.env_manager = TestEnvironmentManager(self.config)
        self.coverage_manager = CoverageManager(self.config)
        self.test_discovery = TestDiscovery(self.config)
        self.parallel_runner = ParallelTestRunner(self.config)
        self.reporter = TestReporter(self.config)
    
    def _load_config(self, config_file: Optional[Path] = None) -> Dict[str, Any]:
        """Load test configuration."""
        default_config = {
            "coverage_threshold": 0.90,
            "max_test_workers": multiprocessing.cpu_count(),
            "test_timeout": 300,
            "output_dir": "test_reports",
            "test_data_dir": "tests/test_data",
            "enable_coverage": True,
            "enable_parallel": True,
            "enable_performance_tests": True
        }
        
        if config_file and config_file.exists():
            import yaml
            with open(config_file) as f:
                file_config = yaml.safe_load(f)
            default_config.update(file_config)
        
        return default_config
    
    def run_comprehensive_tests(self) -> TestSuiteResult:
        """Run comprehensive test suite with all features."""
        start_time = datetime.now()
        
        print("Starting comprehensive test suite...")
        print(f"Configuration: {self.config}")
        
        try:
            # Set up test environment
            env_config = self.env_manager.setup_test_environment()
            print(f"Test environment set up: {env_config}")
            
            # Start coverage if enabled
            if self.config.get("enable_coverage", True):
                self.coverage_manager.start_coverage()
                print("Coverage measurement started")
            
            # Discover tests
            categorized_tests = self.test_discovery.discover_tests()
            total_tests = sum(len(tests) for tests in categorized_tests.values())
            print(f"Discovered {total_tests} tests in {len(categorized_tests)} categories")
            
            # Run tests
            if self.config.get("enable_parallel", True):
                test_results = self.parallel_runner.run_tests_parallel(
                    categorized_tests, self.test_discovery.categories
                )
            else:
                # Sequential execution fallback
                test_results = []
                for category in self.test_discovery.categories:
                    if category.name in categorized_tests:
                        results = self.parallel_runner._run_sequential_category(
                            categorized_tests[category.name], category
                        )
                        test_results.extend(results)
            
            # Stop coverage and get results
            coverage_data = {}
            if self.config.get("enable_coverage", True):
                coverage_data = self.coverage_manager.stop_coverage()
                print(f"Coverage: {coverage_data.get('percentage', 0):.1f}%")
            
            # Create test suite result
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            # Calculate summary statistics
            passed = len([r for r in test_results if r.status == "passed"])
            failed = len([r for r in test_results if r.status == "failed"])
            errors = len([r for r in test_results if r.status == "error"])
            skipped = len([r for r in test_results if r.status == "skipped"])
            
            suite_result = TestSuiteResult(
                start_time=start_time,
                end_time=end_time,
                total_duration=total_duration,
                total_tests=len(test_results),
                passed=passed,
                failed=failed,
                skipped=skipped,
                errors=errors,
                coverage_percentage=coverage_data.get("percentage", 0.0),
                results=test_results
            )
            
            # Generate reports
            print("Generating test reports...")
            report_files = self.reporter.generate_comprehensive_report(suite_result)
            suite_result.artifacts = list(report_files.values())
            
            print(f"Test suite completed in {total_duration:.2f} seconds")
            print(f"Results: {passed} passed, {failed} failed, {errors} errors, {skipped} skipped")
            print(f"Coverage: {coverage_data.get('percentage', 0):.1f}%")
            print(f"Reports generated: {list(report_files.keys())}")
            
            return suite_result
            
        except Exception as e:
            print(f"Test suite execution failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return minimal error result
            end_time = datetime.now()
            return TestSuiteResult(
                start_time=start_time,
                end_time=end_time,
                total_duration=(end_time - start_time).total_seconds(),
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=1,
                coverage_percentage=0.0,
                results=[TestResult(
                    name="test_suite_execution",
                    category="framework",
                    status="error",
                    duration=0.0,
                    error_message=str(e)
                )]
            )
        
        finally:
            # Always cleanup
            self.env_manager.cleanup_test_environment()


def main():
    """Main entry point for test automation framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ASR Post-Processing Test Automation Framework")
    parser.add_argument("--config", "-c", type=Path, help="Configuration file path")
    parser.add_argument("--category", "-t", help="Run only specific test category")
    parser.add_argument("--output", "-o", type=Path, help="Output directory for reports")
    parser.add_argument("--parallel", action="store_true", default=True, help="Enable parallel execution")
    parser.add_argument("--coverage", action="store_true", default=True, help="Enable coverage measurement")
    parser.add_argument("--timeout", type=int, default=300, help="Test timeout in seconds")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config_overrides = {}
    if args.output:
        config_overrides["output_dir"] = str(args.output)
    if not args.parallel:
        config_overrides["enable_parallel"] = False
    if not args.coverage:
        config_overrides["enable_coverage"] = False
    if args.timeout:
        config_overrides["test_timeout"] = args.timeout
    
    # Create and run test framework
    framework = TestAutomationFramework(args.config)
    
    # Override config with command line arguments
    framework.config.update(config_overrides)
    
    # Run tests
    result = framework.run_comprehensive_tests()
    
    # Exit with appropriate code
    if result.failed > 0 or result.errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()