#!/usr/bin/env python3
"""
Production Deployment Readiness Certification Script for Story 4.4

This script provides comprehensive validation of production deployment readiness,
infrastructure requirements, configuration management, and operational monitoring.
"""

import os
import sys
import json
import yaml
import time
import subprocess
import tempfile
import platform
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from post_processors.sanskrit_post_processor import SanskritPostProcessor
from utils.srt_parser import SRTParser
from utils.mcp_transformer_client import create_transformer_client
from sanskrit_hindi_identifier.enhanced_lexicon_manager import EnhancedLexiconManager
from utils.sanskrit_accuracy_validator import SanskritAccuracyValidator
from utils.research_metrics_collector import ResearchMetricsCollector


@dataclass
class SystemRequirement:
    """System requirement specification"""
    name: str
    required: bool
    current_value: Any
    minimum_value: Any
    status: str  # "pass", "fail", "warning"
    message: str


@dataclass
class DeploymentCheck:
    """Individual deployment readiness check"""
    check_name: str
    category: str
    status: str  # "pass", "fail", "warning", "skip"
    message: str
    details: Dict[str, Any]
    execution_time: float


@dataclass
class CertificationResults:
    """Complete certification results"""
    certification_id: str
    timestamp: datetime
    system_info: Dict[str, Any]
    overall_status: str  # "certified", "conditional", "failed"
    checks_passed: int
    checks_failed: int
    checks_warnings: int
    checks_skipped: int
    total_checks: int
    deployment_ready: bool
    requirements_met: bool
    all_checks: List[DeploymentCheck]
    recommendations: List[str]


class ProductionDeploymentValidator:
    """Production deployment readiness certification coordinator"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize deployment validator"""
        self.project_root = Path(__file__).parent.parent
        self.config_path = config_path or self.project_root / "config"
        self.data_dir = self.project_root / "data"
        self.src_dir = self.project_root / "src"
        self.scripts_dir = self.project_root / "scripts"
        
        # Certification tracking
        self.certification_id = f"cert_{int(time.time())}"
        self.checks = []
        self.recommendations = []
        
        # System requirements
        self.system_requirements = self._define_system_requirements()
        
        # Initialize components for testing
        try:
            self.sanskrit_processor = SanskritPostProcessor()
            self.components_available = True
        except Exception as e:
            self.components_available = False
            self.add_recommendation(f"Component initialization failed: {e}")
    
    def _define_system_requirements(self) -> Dict[str, SystemRequirement]:
        """Define production system requirements"""
        return {
            'python_version': SystemRequirement(
                name="Python Version",
                required=True,
                current_value=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                minimum_value="3.10.0",
                status="unknown",
                message=""
            ),
            'memory_gb': SystemRequirement(
                name="Available Memory",
                required=True,
                current_value=self._get_system_memory_gb(),
                minimum_value=4.0,
                status="unknown",
                message=""
            ),
            'disk_space_gb': SystemRequirement(
                name="Available Disk Space",
                required=True,
                current_value=self._get_disk_space_gb(),
                minimum_value=10.0,
                status="unknown",
                message=""
            ),
            'cpu_cores': SystemRequirement(
                name="CPU Cores",
                required=True,
                current_value=os.cpu_count(),
                minimum_value=2,
                status="unknown",
                message=""
            )
        }
    
    def _get_system_memory_gb(self) -> float:
        """Get system memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return 0.0
    
    def _get_disk_space_gb(self) -> float:
        """Get available disk space in GB"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.project_root)
            return free / (1024**3)
        except Exception:
            return 0.0
    
    def add_check(self, check_name: str, category: str, status: str, 
                  message: str, details: Dict[str, Any] = None, execution_time: float = 0.0):
        """Add a deployment check result"""
        check = DeploymentCheck(
            check_name=check_name,
            category=category,
            status=status,
            message=message,
            details=details or {},
            execution_time=execution_time
        )
        self.checks.append(check)
        return check
    
    def add_recommendation(self, recommendation: str):
        """Add a recommendation for deployment"""
        self.recommendations.append(recommendation)
    
    def run_certification(self) -> CertificationResults:
        """Run complete production deployment certification"""
        print(f"🚀 Starting Production Deployment Certification: {self.certification_id}")
        print("=" * 70)
        
        start_time = time.time()
        
        # Run all certification checks
        self._check_system_requirements()
        self._check_infrastructure_requirements()
        self._check_configuration_management()
        self._check_component_availability()
        self._check_data_dependencies()
        self._check_security_requirements()
        self._check_monitoring_readiness()
        self._check_backup_procedures()
        self._check_performance_baselines()
        self._check_documentation_completeness()
        
        total_time = time.time() - start_time
        
        # Analyze results
        results = self._analyze_certification_results(total_time)
        
        # Generate certification report
        self._generate_certification_report(results)
        
        return results
    
    def _check_system_requirements(self):
        """Check system requirements for production deployment"""
        print("\n📋 Checking System Requirements...")
        
        for req_name, requirement in self.system_requirements.items():
            start_time = time.time()
            
            try:
                if req_name == 'python_version':
                    current_version = tuple(map(int, requirement.current_value.split('.')))
                    min_version = tuple(map(int, requirement.minimum_value.split('.')))
                    
                    if current_version >= min_version:
                        status = "pass"
                        message = f"Python {requirement.current_value} meets requirement >= {requirement.minimum_value}"
                    else:
                        status = "fail"
                        message = f"Python {requirement.current_value} below minimum {requirement.minimum_value}"
                
                elif req_name in ['memory_gb', 'disk_space_gb']:
                    if requirement.current_value >= requirement.minimum_value:
                        status = "pass"
                        message = f"{requirement.name}: {requirement.current_value:.1f}GB >= {requirement.minimum_value}GB"
                    else:
                        status = "fail"
                        message = f"{requirement.name}: {requirement.current_value:.1f}GB < {requirement.minimum_value}GB"
                
                elif req_name == 'cpu_cores':
                    if requirement.current_value >= requirement.minimum_value:
                        status = "pass"
                        message = f"CPU cores: {requirement.current_value} >= {requirement.minimum_value}"
                    else:
                        status = "fail"
                        message = f"CPU cores: {requirement.current_value} < {requirement.minimum_value}"
                
                requirement.status = status
                requirement.message = message
                
                execution_time = time.time() - start_time
                self.add_check(
                    requirement.name,
                    "system_requirements",
                    status,
                    message,
                    {"current": requirement.current_value, "minimum": requirement.minimum_value},
                    execution_time
                )
                
                print(f"  {'✅' if status == 'pass' else '❌'} {requirement.name}: {message}")
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.add_check(
                    requirement.name,
                    "system_requirements",
                    "fail",
                    f"Failed to check {requirement.name}: {e}",
                    {"error": str(e)},
                    execution_time
                )
                print(f"  ❌ {requirement.name}: Error checking requirement - {e}")
    
    def _check_infrastructure_requirements(self):
        """Check infrastructure requirements and deployment procedures"""
        print("\n🏗️ Checking Infrastructure Requirements...")
        
        # Check required directories
        required_dirs = [
            "src", "data", "tests", "config", "scripts",
            "data/lexicons", "data/metrics", "data/test_samples"
        ]
        
        for dir_path in required_dirs:
            start_time = time.time()
            full_path = self.project_root / dir_path
            
            if full_path.exists():
                status = "pass"
                message = f"Required directory exists: {dir_path}"
            else:
                status = "fail" 
                message = f"Missing required directory: {dir_path}"
                self.add_recommendation(f"Create missing directory: {dir_path}")
            
            execution_time = time.time() - start_time
            self.add_check(
                f"directory_{dir_path.replace('/', '_')}",
                "infrastructure",
                status,
                message,
                {"path": str(full_path)},
                execution_time
            )
            print(f"  {'✅' if status == 'pass' else '❌'} {message}")
        
        # Check Python dependencies
        start_time = time.time()
        required_packages = [
            "pandas", "pyyaml", "click", "structlog", "pytest", "tqdm"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if not missing_packages:
            status = "pass"
            message = "All required Python packages are installed"
        else:
            status = "fail"
            message = f"Missing Python packages: {', '.join(missing_packages)}"
            self.add_recommendation(f"Install missing packages: pip install {' '.join(missing_packages)}")
        
        execution_time = time.time() - start_time
        self.add_check(
            "python_dependencies",
            "infrastructure", 
            status,
            message,
            {"missing_packages": missing_packages},
            execution_time
        )
        print(f"  {'✅' if status == 'pass' else '❌'} {message}")
    
    def _check_configuration_management(self):
        """Check production configuration management and environment compatibility"""
        print("\n⚙️ Checking Configuration Management...")
        
        # Check for configuration files
        config_files = [
            "config/default_config.yaml",
            "config/production_config.yaml"
        ]
        
        for config_file in config_files:
            start_time = time.time()
            config_path = self.project_root / config_file
            
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                    
                    status = "pass"
                    message = f"Configuration file valid: {config_file}"
                    details = {"keys": list(config_data.keys()) if config_data else []}
                    
                except Exception as e:
                    status = "fail"
                    message = f"Configuration file invalid: {config_file} - {e}"
                    details = {"error": str(e)}
                    self.add_recommendation(f"Fix configuration file: {config_file}")
            else:
                if "production" in config_file:
                    status = "warning"
                    message = f"Production configuration file missing: {config_file}"
                    self.add_recommendation(f"Create production configuration: {config_file}")
                else:
                    status = "fail"
                    message = f"Required configuration file missing: {config_file}"
                    self.add_recommendation(f"Create configuration file: {config_file}")
                details = {}
            
            execution_time = time.time() - start_time
            self.add_check(
                f"config_{Path(config_file).stem}",
                "configuration",
                status,
                message,
                details,
                execution_time
            )
            print(f"  {'✅' if status == 'pass' else '⚠️' if status == 'warning' else '❌'} {message}")
        
        # Check environment variables
        start_time = time.time()
        env_vars = ["PYTHONPATH"]
        missing_env_vars = []
        
        for env_var in env_vars:
            if env_var not in os.environ:
                missing_env_vars.append(env_var)
        
        if not missing_env_vars:
            status = "pass"
            message = "Environment variables properly configured"
        else:
            status = "warning"
            message = f"Optional environment variables not set: {', '.join(missing_env_vars)}"
            self.add_recommendation("Consider setting PYTHONPATH for easier imports")
        
        execution_time = time.time() - start_time
        self.add_check(
            "environment_variables",
            "configuration",
            status,
            message,
            {"missing": missing_env_vars},
            execution_time
        )
        print(f"  {'✅' if status == 'pass' else '⚠️'} {message}")
    
    def _check_component_availability(self):
        """Check availability and functionality of core components"""
        print("\n🔧 Checking Component Availability...")
        
        components = [
            ("SanskritPostProcessor", "post_processors.sanskrit_post_processor", "SanskritPostProcessor"),
            ("SRTParser", "utils.srt_parser", "SRTParser"),
            ("MCPTransformerClient", "utils.mcp_transformer_client", "create_transformer_client"),
            ("EnhancedLexiconManager", "sanskrit_hindi_identifier.enhanced_lexicon_manager", "EnhancedLexiconManager"),
            ("SanskritAccuracyValidator", "utils.sanskrit_accuracy_validator", "SanskritAccuracyValidator"),
            ("ResearchMetricsCollector", "utils.research_metrics_collector", "ResearchMetricsCollector")
        ]
        
        for component_name, module_path, class_name in components:
            start_time = time.time()
            
            try:
                # Try to import and instantiate
                module = __import__(module_path, fromlist=[class_name])
                component_class = getattr(module, class_name)
                
                if callable(component_class):
                    if "create_" in class_name:
                        # Factory function
                        instance = component_class()
                    else:
                        # Regular class
                        instance = component_class()
                    
                    status = "pass"
                    message = f"Component available and functional: {component_name}"
                    details = {"module": module_path, "class": class_name}
                else:
                    status = "fail"
                    message = f"Component not callable: {component_name}"
                    details = {"error": "not_callable"}
                
            except ImportError as e:
                status = "fail"
                message = f"Component import failed: {component_name} - {e}"
                details = {"error": str(e), "type": "import_error"}
                self.add_recommendation(f"Check component implementation: {component_name}")
                
            except Exception as e:
                status = "warning"
                message = f"Component initialization issue: {component_name} - {e}"
                details = {"error": str(e), "type": "initialization_error"}
                self.add_recommendation(f"Review component configuration: {component_name}")
            
            execution_time = time.time() - start_time
            self.add_check(
                f"component_{component_name.lower()}",
                "components",
                status,
                message,
                details,
                execution_time
            )
            print(f"  {'✅' if status == 'pass' else '⚠️' if status == 'warning' else '❌'} {message}")
    
    def _check_data_dependencies(self):
        """Check data dependencies and lexicon availability"""
        print("\n📊 Checking Data Dependencies...")
        
        # Check lexicon files
        lexicon_files = [
            "data/lexicons/proper_nouns.yaml",
            "data/lexicons/corrections.yaml",
            "data/lexicons/phrases.yaml",
            "data/lexicons/verses.yaml"
        ]
        
        for lexicon_file in lexicon_files:
            start_time = time.time()
            lexicon_path = self.project_root / lexicon_file
            
            if lexicon_path.exists():
                try:
                    with open(lexicon_path, 'r', encoding='utf-8') as f:
                        lexicon_data = yaml.safe_load(f)
                    
                    if lexicon_data:
                        status = "pass"
                        message = f"Lexicon file valid: {lexicon_file} ({len(lexicon_data)} entries)"
                        details = {"entries": len(lexicon_data)}
                    else:
                        status = "warning"
                        message = f"Lexicon file empty: {lexicon_file}"
                        details = {"entries": 0}
                        self.add_recommendation(f"Populate lexicon file: {lexicon_file}")
                    
                except Exception as e:
                    status = "fail"
                    message = f"Lexicon file invalid: {lexicon_file} - {e}"
                    details = {"error": str(e)}
                    self.add_recommendation(f"Fix lexicon file format: {lexicon_file}")
            else:
                status = "fail"
                message = f"Missing lexicon file: {lexicon_file}"
                details = {}
                self.add_recommendation(f"Create lexicon file: {lexicon_file}")
            
            execution_time = time.time() - start_time
            self.add_check(
                f"lexicon_{Path(lexicon_file).stem}",
                "data_dependencies",
                status,
                message,
                details,
                execution_time
            )
            print(f"  {'✅' if status == 'pass' else '⚠️' if status == 'warning' else '❌'} {message}")
        
        # Check test samples
        start_time = time.time()
        test_samples_dir = self.project_root / "data/test_samples"
        
        if test_samples_dir.exists():
            srt_files = list(test_samples_dir.glob("*.srt"))
            if srt_files:
                status = "pass"
                message = f"Test samples available: {len(srt_files)} SRT files"
                details = {"file_count": len(srt_files)}
            else:
                status = "warning"
                message = "No SRT test samples found"
                details = {"file_count": 0}
                self.add_recommendation("Add SRT test samples for validation")
        else:
            status = "fail"
            message = "Test samples directory missing"
            details = {}
            self.add_recommendation("Create test samples directory and add sample files")
        
        execution_time = time.time() - start_time
        self.add_check(
            "test_samples",
            "data_dependencies",
            status,
            message,
            details,
            execution_time
        )
        print(f"  {'✅' if status == 'pass' else '⚠️' if status == 'warning' else '❌'} {message}")
    
    def _check_security_requirements(self):
        """Check security requirements for production deployment"""
        print("\n🔒 Checking Security Requirements...")
        
        # Check file permissions
        start_time = time.time()
        sensitive_files = [
            "config/production_config.yaml",
            "data/lexicons"
        ]
        
        permission_issues = []
        for file_path in sensitive_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    if full_path.is_file():
                        # Check file permissions
                        mode = oct(full_path.stat().st_mode)[-3:]
                        if mode in ["644", "600"]:  # Acceptable permissions
                            continue
                        else:
                            permission_issues.append(f"{file_path}: {mode}")
                except Exception as e:
                    permission_issues.append(f"{file_path}: error checking permissions")
        
        if not permission_issues:
            status = "pass"
            message = "File permissions properly configured"
        else:
            status = "warning"
            message = f"File permission concerns: {', '.join(permission_issues)}"
            self.add_recommendation("Review and secure file permissions for production")
        
        execution_time = time.time() - start_time
        self.add_check(
            "file_permissions",
            "security",
            status,
            message,
            {"issues": permission_issues},
            execution_time
        )
        print(f"  {'✅' if status == 'pass' else '⚠️'} {message}")
        
        # Check for secrets in configuration
        start_time = time.time()
        config_files = list(self.config_path.glob("*.yaml")) + list(self.config_path.glob("*.yml"))
        secrets_found = []
        
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                    # Look for potential secrets
                    secret_indicators = ["password", "secret", "key", "token", "api_key"]
                    for indicator in secret_indicators:
                        if indicator in content and ":" in content:
                            secrets_found.append(f"{config_file.name}: contains '{indicator}'")
            except Exception:
                pass
        
        if not secrets_found:
            status = "pass"
            message = "No obvious secrets found in configuration files"
        else:
            status = "warning"
            message = f"Potential secrets in configuration: {', '.join(secrets_found)}"
            self.add_recommendation("Review configuration files for exposed secrets")
        
        execution_time = time.time() - start_time
        self.add_check(
            "secrets_check",
            "security",
            status,
            message,
            {"potential_secrets": secrets_found},
            execution_time
        )
        print(f"  {'✅' if status == 'pass' else '⚠️'} {message}")
    
    def _check_monitoring_readiness(self):
        """Check operational monitoring and alerting readiness"""
        print("\n📊 Checking Monitoring Readiness...")
        
        # Check for monitoring configuration
        start_time = time.time()
        monitoring_configs = [
            "config/monitoring.yaml",
            "config/logging.yaml",
            "config/alerting.yaml"
        ]
        
        available_configs = []
        for config_file in monitoring_configs:
            config_path = self.project_root / config_file
            if config_path.exists():
                available_configs.append(config_file)
        
        if available_configs:
            status = "pass"
            message = f"Monitoring configurations available: {', '.join(available_configs)}"
        else:
            status = "warning"
            message = "No monitoring configuration files found"
            self.add_recommendation("Create monitoring and alerting configuration files")
        
        execution_time = time.time() - start_time
        self.add_check(
            "monitoring_config",
            "monitoring",
            status,
            message,
            {"available_configs": available_configs},
            execution_time
        )
        print(f"  {'✅' if status == 'pass' else '⚠️'} {message}")
        
        # Check logging functionality
        start_time = time.time()
        try:
            import structlog
            logger = structlog.get_logger()
            logger.info("Test log message for deployment validation")
            
            status = "pass"
            message = "Logging system functional"
            details = {"logger": "structlog"}
        except Exception as e:
            status = "warning"
            message = f"Logging system issue: {e}"
            details = {"error": str(e)}
            self.add_recommendation("Review logging configuration")
        
        execution_time = time.time() - start_time
        self.add_check(
            "logging_system",
            "monitoring",
            status,
            message,
            details,
            execution_time
        )
        print(f"  {'✅' if status == 'pass' else '⚠️'} {message}")
    
    def _check_backup_procedures(self):
        """Check backup and recovery procedures"""
        print("\n💾 Checking Backup Procedures...")
        
        # Check for backup scripts
        start_time = time.time()
        backup_scripts = list(self.scripts_dir.glob("*backup*")) + list(self.scripts_dir.glob("*recovery*"))
        
        if backup_scripts:
            status = "pass"
            message = f"Backup scripts available: {', '.join(s.name for s in backup_scripts)}"
            details = {"scripts": [s.name for s in backup_scripts]}
        else:
            status = "warning"
            message = "No backup/recovery scripts found"
            details = {"scripts": []}
            self.add_recommendation("Create backup and recovery procedures")
        
        execution_time = time.time() - start_time
        self.add_check(
            "backup_scripts",
            "backup_recovery",
            status,
            message,
            details,
            execution_time
        )
        print(f"  {'✅' if status == 'pass' else '⚠️'} {message}")
        
        # Check data backup locations
        start_time = time.time()
        backup_dirs = ["backup", "backups", "data/backup"]
        available_backup_dirs = []
        
        for backup_dir in backup_dirs:
            backup_path = self.project_root / backup_dir
            if backup_path.exists():
                available_backup_dirs.append(backup_dir)
        
        if available_backup_dirs:
            status = "pass"
            message = f"Backup directories available: {', '.join(available_backup_dirs)}"
        else:
            status = "warning"
            message = "No backup directories found"
            self.add_recommendation("Create backup directory structure")
        
        execution_time = time.time() - start_time
        self.add_check(
            "backup_directories",
            "backup_recovery",
            status,
            message,
            {"directories": available_backup_dirs},
            execution_time
        )
        print(f"  {'✅' if status == 'pass' else '⚠️'} {message}")
    
    def _check_performance_baselines(self):
        """Check performance baselines and benchmarks"""
        print("\n🏃 Checking Performance Baselines...")
        
        # Check for performance test results
        start_time = time.time()
        metrics_dir = self.project_root / "data/metrics"
        
        if metrics_dir.exists():
            metric_files = list(metrics_dir.glob("*.json"))
            if metric_files:
                status = "pass"
                message = f"Performance metrics available: {len(metric_files)} files"
                details = {"metric_files": len(metric_files)}
            else:
                status = "warning"
                message = "No performance metrics found"
                details = {"metric_files": 0}
                self.add_recommendation("Run performance tests to establish baselines")
        else:
            status = "warning"
            message = "Metrics directory not found"
            details = {"metric_files": 0}
            self.add_recommendation("Create metrics directory and run performance tests")
        
        execution_time = time.time() - start_time
        self.add_check(
            "performance_metrics",
            "performance",
            status,
            message,
            details,
            execution_time
        )
        print(f"  {'✅' if status == 'pass' else '⚠️'} {message}")
        
        # Test basic processing performance
        if self.components_available:
            start_time = time.time()
            try:
                # Create a simple test
                test_content = """1
00:00:01,000 --> 00:00:05,000
Test content for performance validation."""
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
                    f.write(test_content)
                    test_file = Path(f.name)
                
                output_file = test_file.with_suffix('.processed.srt')
                
                process_start = time.time()
                metrics = self.sanskrit_processor.process_srt_file(test_file, output_file)
                process_time = time.time() - process_start
                
                # Clean up
                test_file.unlink()
                if output_file.exists():
                    output_file.unlink()
                
                if process_time < 1.0:  # Less than 1 second
                    status = "pass"
                    message = f"Performance test passed: {process_time:.3f}s"
                else:
                    status = "warning"
                    message = f"Performance test slow: {process_time:.3f}s"
                    self.add_recommendation("Optimize system performance")
                
                details = {"processing_time": process_time}
                
            except Exception as e:
                status = "fail"
                message = f"Performance test failed: {e}"
                details = {"error": str(e)}
                self.add_recommendation("Fix system components for performance testing")
        else:
            status = "skip"
            message = "Performance test skipped - components not available"
            details = {}
        
        execution_time = time.time() - start_time
        self.add_check(
            "performance_test",
            "performance",
            status,
            message,
            details,
            execution_time
        )
        print(f"  {'✅' if status == 'pass' else '⚠️' if status == 'warning' else '⏭️' if status == 'skip' else '❌'} {message}")
    
    def _check_documentation_completeness(self):
        """Check documentation completeness for production deployment"""
        print("\n📚 Checking Documentation Completeness...")
        
        # Check for required documentation
        required_docs = [
            "README.md",
            "CLAUDE.md",
            "docs/README.md",
            "docs/architecture.md",
            "docs/deployment.md"
        ]
        
        for doc_file in required_docs:
            start_time = time.time()
            doc_path = self.project_root / doc_file
            
            if doc_path.exists():
                try:
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if len(content) > 100:  # At least some content
                        status = "pass"
                        message = f"Documentation available: {doc_file}"
                        details = {"length": len(content)}
                    else:
                        status = "warning"
                        message = f"Documentation minimal: {doc_file}"
                        details = {"length": len(content)}
                        self.add_recommendation(f"Expand documentation: {doc_file}")
                except Exception as e:
                    status = "fail"
                    message = f"Documentation unreadable: {doc_file} - {e}"
                    details = {"error": str(e)}
            else:
                if "deployment" in doc_file:
                    status = "warning"
                    message = f"Deployment documentation missing: {doc_file}"
                    self.add_recommendation(f"Create deployment documentation: {doc_file}")
                else:
                    status = "fail"
                    message = f"Required documentation missing: {doc_file}"
                    self.add_recommendation(f"Create documentation: {doc_file}")
                details = {}
            
            execution_time = time.time() - start_time
            self.add_check(
                f"doc_{Path(doc_file).stem}",
                "documentation",
                status,
                message,
                details,
                execution_time
            )
            print(f"  {'✅' if status == 'pass' else '⚠️' if status == 'warning' else '❌'} {message}")
    
    def _analyze_certification_results(self, total_time: float) -> CertificationResults:
        """Analyze certification results and determine overall status"""
        checks_passed = sum(1 for check in self.checks if check.status == "pass")
        checks_failed = sum(1 for check in self.checks if check.status == "fail")
        checks_warnings = sum(1 for check in self.checks if check.status == "warning")
        checks_skipped = sum(1 for check in self.checks if check.status == "skip")
        total_checks = len(self.checks)
        
        # Determine overall status
        if checks_failed == 0:
            if checks_warnings <= 3:  # Allow some warnings
                overall_status = "certified"
                deployment_ready = True
            else:
                overall_status = "conditional"
                deployment_ready = True
        else:
            overall_status = "failed"
            deployment_ready = False
        
        # Check if all requirements are met
        requirements_met = all(req.status == "pass" for req in self.system_requirements.values())
        
        # Get system information
        system_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": os.cpu_count(),
            "total_memory_gb": self._get_system_memory_gb(),
            "available_disk_gb": self._get_disk_space_gb(),
            "certification_duration": total_time
        }
        
        return CertificationResults(
            certification_id=self.certification_id,
            timestamp=datetime.now(),
            system_info=system_info,
            overall_status=overall_status,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            checks_warnings=checks_warnings,
            checks_skipped=checks_skipped,
            total_checks=total_checks,
            deployment_ready=deployment_ready,
            requirements_met=requirements_met,
            all_checks=self.checks,
            recommendations=self.recommendations
        )
    
    def _generate_certification_report(self, results: CertificationResults):
        """Generate comprehensive certification report"""
        print("\n" + "=" * 70)
        print("🎯 PRODUCTION DEPLOYMENT CERTIFICATION REPORT")
        print("=" * 70)
        
        # Overall status
        status_emoji = "✅" if results.overall_status == "certified" else "⚠️" if results.overall_status == "conditional" else "❌"
        print(f"\n{status_emoji} Overall Status: {results.overall_status.upper()}")
        print(f"🚀 Deployment Ready: {'YES' if results.deployment_ready else 'NO'}")
        print(f"📋 Requirements Met: {'YES' if results.requirements_met else 'NO'}")
        
        # Summary statistics
        print(f"\n📊 Certification Summary:")
        print(f"  • Total Checks: {results.total_checks}")
        print(f"  • Passed: {results.checks_passed} ✅")
        print(f"  • Failed: {results.checks_failed} ❌")
        print(f"  • Warnings: {results.checks_warnings} ⚠️")
        print(f"  • Skipped: {results.checks_skipped} ⏭️")
        print(f"  • Success Rate: {(results.checks_passed / results.total_checks * 100):.1f}%")
        
        # System information
        print(f"\n🖥️ System Information:")
        print(f"  • Platform: {results.system_info['platform']}")
        print(f"  • Python: {results.system_info['python_version'].split()[0]}")
        print(f"  • CPU Cores: {results.system_info['cpu_count']}")
        print(f"  • Memory: {results.system_info['total_memory_gb']:.1f} GB")
        print(f"  • Disk Space: {results.system_info['available_disk_gb']:.1f} GB")
        
        # Failed checks
        if results.checks_failed > 0:
            print(f"\n❌ Failed Checks ({results.checks_failed}):")
            for check in results.all_checks:
                if check.status == "fail":
                    print(f"  • {check.check_name}: {check.message}")
        
        # Recommendations
        if results.recommendations:
            print(f"\n💡 Recommendations ({len(results.recommendations)}):")
            for i, recommendation in enumerate(results.recommendations, 1):
                print(f"  {i}. {recommendation}")
        
        # Save detailed report
        report_file = self.project_root / "data/metrics" / f"deployment_certification_{self.certification_id}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            # Convert datetime to string for JSON serialization
            report_data = asdict(results)
            report_data['timestamp'] = results.timestamp.isoformat()
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        print("=" * 70)
        
        return report_file


def main():
    """Main function for deployment validation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Deployment Readiness Certification")
    parser.add_argument("--config", "-c", type=Path, help="Configuration directory path")
    parser.add_argument("--output", "-o", type=Path, help="Output report file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ProductionDeploymentValidator(args.config)
    
    # Run certification
    try:
        results = validator.run_certification()
        
        # Exit with appropriate code
        if results.overall_status == "certified":
            sys.exit(0)
        elif results.overall_status == "conditional":
            sys.exit(1)  # Warnings present
        else:
            sys.exit(2)  # Failures present
            
    except Exception as e:
        print(f"❌ Certification failed with error: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()