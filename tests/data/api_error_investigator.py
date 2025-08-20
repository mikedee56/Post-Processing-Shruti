"""
API Error Investigation System for Story 5.5: Testing & Quality Assurance Framework

This module provides comprehensive API error investigation capabilities,
including runtime diagnostics, dependency validation, and system health monitoring.
"""

import json
import logging
import traceback
import subprocess
import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import time
import psutil
import platform


# Configure detailed logging for investigation
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SystemDiagnostic:
    """Represents a system diagnostic result."""
    component: str
    status: str  # 'healthy', 'warning', 'error', 'critical'
    details: Dict[str, Any]
    recommendations: List[str]
    timestamp: str


@dataclass
class APIErrorDetail:
    """Detailed API error information for investigation."""
    error_id: str
    error_type: str
    error_message: str
    error_code: Optional[str]
    component: str
    function_name: str
    line_number: int
    file_path: str
    stack_trace: str
    context: Dict[str, Any]
    system_state: Dict[str, Any]
    timestamp: str


@dataclass
class DependencyStatus:
    """Status of a system dependency."""
    name: str
    version: Optional[str]
    status: str  # 'installed', 'missing', 'incompatible', 'error'
    import_path: str
    error_details: Optional[str]
    requirements: Dict[str, Any]


class APIErrorInvestigator:
    """
    Comprehensive API error investigation system.
    
    Provides deep diagnostics for API errors, dependency issues,
    and system health problems in the ASR post-processing framework.
    """
    
    def __init__(self):
        """Initialize the API error investigator."""
        self.investigation_id = f"investigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.diagnostics: List[SystemDiagnostic] = []
        self.api_errors: List[APIErrorDetail] = []
        self.dependency_statuses: List[DependencyStatus] = []
        
        # Add src to Python path
        src_path = Path("src").absolute()
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        logger.info(f"APIErrorInvestigator initialized: {self.investigation_id}")
    
    def investigate_system_health(self) -> Dict[str, Any]:
        """
        Comprehensive system health investigation.
        
        Returns:
            Complete system health report with diagnostics
        """
        
        logger.info("Starting comprehensive system health investigation")
        
        health_report = {
            "investigation_id": self.investigation_id,
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "python_environment": self._investigate_python_environment(),
            "dependency_analysis": self._investigate_dependencies(),
            "component_diagnostics": self._investigate_components(),
            "api_error_analysis": self._investigate_api_errors(),
            "performance_metrics": self._investigate_performance(),
            "recommendations": [],
            "overall_status": "unknown"
        }
        
        # Determine overall status
        health_report["overall_status"] = self._determine_overall_status()
        
        # Generate comprehensive recommendations
        health_report["recommendations"] = self._generate_system_recommendations()
        
        logger.info(f"System health investigation completed: {health_report['overall_status']}")
        return health_report
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        
        try:
            return {
                "platform": platform.platform(),
                "architecture": platform.architecture(),
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": {
                    "total": psutil.disk_usage('.').total,
                    "free": psutil.disk_usage('.').free
                },
                "working_directory": str(Path.cwd()),
                "python_path": sys.path[:5]  # First 5 entries
            }
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {"error": str(e)}
    
    def _investigate_python_environment(self) -> Dict[str, Any]:
        """Investigate Python environment and virtual environment status."""
        
        env_info = {
            "virtual_env": None,
            "pip_version": None,
            "installed_packages": {},
            "environment_variables": {},
            "path_issues": []
        }
        
        try:
            # Check virtual environment
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                env_info["virtual_env"] = {
                    "active": True,
                    "path": sys.prefix,
                    "base_prefix": getattr(sys, 'base_prefix', sys.prefix)
                }
            else:
                env_info["virtual_env"] = {"active": False}
            
            # Get pip version
            try:
                result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                                     capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    env_info["pip_version"] = result.stdout.strip()
            except Exception as e:
                env_info["pip_version"] = f"Error: {str(e)}"
            
            # Check critical environment variables
            critical_env_vars = ['PYTHONPATH', 'PATH', 'VIRTUAL_ENV']
            for var in critical_env_vars:
                env_info["environment_variables"][var] = os.environ.get(var, None)
            
            # Check for path issues
            src_path = Path("src").absolute()
            if not src_path.exists():
                env_info["path_issues"].append(f"Source directory not found: {src_path}")
            
            if str(src_path) not in sys.path:
                env_info["path_issues"].append(f"Source directory not in Python path: {src_path}")
            
        except Exception as e:
            logger.error(f"Failed to investigate Python environment: {e}")
            env_info["error"] = str(e)
        
        return env_info
    
    def _investigate_dependencies(self) -> Dict[str, Any]:
        """Investigate all system dependencies and their status."""
        
        # Critical dependencies for the ASR post-processing system
        critical_dependencies = {
            "pysrt": {"import_path": "pysrt", "required": True},
            "yaml": {"import_path": "yaml", "required": True},
            "pandas": {"import_path": "pandas", "required": True},
            "numpy": {"import_path": "numpy", "required": True},
            "fuzzywuzzy": {"import_path": "fuzzywuzzy", "required": True},
            "Levenshtein": {"import_path": "Levenshtein", "required": True},
            "structlog": {"import_path": "structlog", "required": False},
            "sanskrit_parser": {"import_path": "sanskrit_parser", "required": False},
            "inltk": {"import_path": "inltk", "required": False}
        }
        
        dependency_analysis = {
            "total_dependencies": len(critical_dependencies),
            "installed_count": 0,
            "missing_count": 0,
            "error_count": 0,
            "dependency_details": {},
            "critical_issues": []
        }
        
        for dep_name, dep_info in critical_dependencies.items():
            status = self._check_dependency_status(dep_name, dep_info)
            self.dependency_statuses.append(status)
            dependency_analysis["dependency_details"][dep_name] = asdict(status)
            
            if status.status == "installed":
                dependency_analysis["installed_count"] += 1
            elif status.status == "missing":
                dependency_analysis["missing_count"] += 1
                if dep_info["required"]:
                    dependency_analysis["critical_issues"].append(f"Required dependency missing: {dep_name}")
            else:
                dependency_analysis["error_count"] += 1
                if dep_info["required"]:
                    dependency_analysis["critical_issues"].append(f"Required dependency error: {dep_name} - {status.error_details}")
        
        return dependency_analysis
    
    def _check_dependency_status(self, dep_name: str, dep_info: Dict[str, Any]) -> DependencyStatus:
        """Check the status of a specific dependency."""
        
        import_path = dep_info["import_path"]
        
        try:
            # Try to import the module
            module = importlib.import_module(import_path)
            
            # Get version if available
            version = None
            for attr in ['__version__', 'VERSION', 'version']:
                if hasattr(module, attr):
                    version = getattr(module, attr)
                    break
            
            return DependencyStatus(
                name=dep_name,
                version=str(version) if version else None,
                status="installed",
                import_path=import_path,
                error_details=None,
                requirements=dep_info
            )
            
        except ImportError as e:
            return DependencyStatus(
                name=dep_name,
                version=None,
                status="missing",
                import_path=import_path,
                error_details=str(e),
                requirements=dep_info
            )
            
        except Exception as e:
            return DependencyStatus(
                name=dep_name,
                version=None,
                status="error",
                import_path=import_path,
                error_details=str(e),
                requirements=dep_info
            )
    
    def _investigate_components(self) -> Dict[str, Any]:
        """Investigate the status of core system components."""
        
        components = {
            "post_processors": {
                "sanskrit_post_processor": "post_processors.sanskrit_post_processor.SanskritPostProcessor"
            },
            "utils": {
                "advanced_text_normalizer": "utils.advanced_text_normalizer.AdvancedTextNormalizer",
                "srt_parser": "utils.srt_parser.SRTParser",
                "mcp_client": "utils.mcp_client.create_mcp_client"
            },
            "sanskrit_hindi_identifier": {
                "word_identifier": "sanskrit_hindi_identifier.word_identifier.SanskritHindiIdentifier",
                "lexicon_manager": "sanskrit_hindi_identifier.lexicon_manager.LexiconManager"
            },
            "ner_module": {
                "yoga_vedanta_ner": "ner_module.yoga_vedanta_ner.YogaVedantaNER",
                "capitalization_engine": "ner_module.capitalization_engine.CapitalizationEngine"
            }
        }
        
        component_analysis = {
            "total_components": 0,
            "working_components": 0,
            "failed_components": 0,
            "component_details": {},
            "critical_failures": []
        }
        
        for category, category_components in components.items():
            component_analysis["component_details"][category] = {}
            
            for comp_name, comp_path in category_components.items():
                component_analysis["total_components"] += 1
                
                diagnostic = self._test_component(comp_name, comp_path, category)
                self.diagnostics.append(diagnostic)
                
                component_analysis["component_details"][category][comp_name] = asdict(diagnostic)
                
                if diagnostic.status == "healthy":
                    component_analysis["working_components"] += 1
                else:
                    component_analysis["failed_components"] += 1
                    component_analysis["critical_failures"].append(f"{category}.{comp_name}: {diagnostic.status}")
        
        return component_analysis
    
    def _test_component(self, comp_name: str, comp_path: str, category: str) -> SystemDiagnostic:
        """Test a specific system component."""
        
        diagnostic = SystemDiagnostic(
            component=f"{category}.{comp_name}",
            status="unknown",
            details={},
            recommendations=[],
            timestamp=datetime.now().isoformat()
        )
        
        try:
            # Parse the component path
            if '.' in comp_path:
                module_path, class_or_func = comp_path.rsplit('.', 1)
            else:
                module_path, class_or_func = comp_path, None
            
            # Import the module
            module = importlib.import_module(module_path)
            diagnostic.details["module_imported"] = True
            diagnostic.details["module_path"] = module_path
            
            if class_or_func:
                # Get the class or function
                component_obj = getattr(module, class_or_func)
                diagnostic.details["component_found"] = True
                
                # Try to instantiate or call
                if inspect.isclass(component_obj):
                    try:
                        instance = component_obj()
                        diagnostic.details["instantiation"] = "success"
                        diagnostic.status = "healthy"
                    except Exception as e:
                        diagnostic.details["instantiation"] = f"failed: {str(e)}"
                        diagnostic.status = "error"
                        diagnostic.recommendations.append(f"Fix instantiation error: {str(e)}")
                elif inspect.isfunction(component_obj):
                    try:
                        # For functions like create_mcp_client, try calling with no args
                        result = component_obj()
                        diagnostic.details["function_call"] = "success"
                        diagnostic.status = "healthy"
                    except Exception as e:
                        diagnostic.details["function_call"] = f"failed: {str(e)}"
                        diagnostic.status = "warning"
                        diagnostic.recommendations.append(f"Function call failed: {str(e)}")
                else:
                    diagnostic.details["component_type"] = type(component_obj).__name__
                    diagnostic.status = "healthy"
            else:
                diagnostic.status = "healthy"
                diagnostic.details["module_only"] = True
            
        except ImportError as e:
            diagnostic.status = "critical"
            diagnostic.details["import_error"] = str(e)
            diagnostic.recommendations.append(f"Fix import error: {str(e)}")
            
            # Capture detailed error
            self._capture_api_error(e, f"component_test_{comp_name}", {
                "component_path": comp_path,
                "category": category
            })
            
        except Exception as e:
            diagnostic.status = "error"
            diagnostic.details["error"] = str(e)
            diagnostic.recommendations.append(f"Fix component error: {str(e)}")
            
            # Capture detailed error
            self._capture_api_error(e, f"component_test_{comp_name}", {
                "component_path": comp_path,
                "category": category
            })
        
        return diagnostic
    
    def _investigate_api_errors(self) -> Dict[str, Any]:
        """Investigate and analyze captured API errors."""
        
        error_analysis = {
            "total_errors": len(self.api_errors),
            "error_categories": {},
            "error_patterns": [],
            "root_causes": [],
            "fix_priorities": []
        }
        
        if not self.api_errors:
            return error_analysis
        
        # Categorize errors
        for error in self.api_errors:
            error_type = error.error_type
            if error_type not in error_analysis["error_categories"]:
                error_analysis["error_categories"][error_type] = {
                    "count": 0,
                    "components": set(),
                    "common_messages": []
                }
            
            error_analysis["error_categories"][error_type]["count"] += 1
            error_analysis["error_categories"][error_type]["components"].add(error.component)
        
        # Convert sets to lists for JSON serialization
        for category in error_analysis["error_categories"].values():
            category["components"] = list(category["components"])
        
        # Identify patterns
        import_errors = [e for e in self.api_errors if e.error_type == "ImportError"]
        if import_errors:
            error_analysis["error_patterns"].append("Multiple import errors suggest missing dependencies")
        
        attr_errors = [e for e in self.api_errors if e.error_type == "AttributeError"]
        if attr_errors:
            error_analysis["error_patterns"].append("Attribute errors suggest API compatibility issues")
        
        # Identify root causes
        if import_errors:
            error_analysis["root_causes"].append("Missing or incorrectly installed dependencies")
        
        if any("mcp" in e.component.lower() for e in self.api_errors):
            error_analysis["root_causes"].append("MCP client configuration or connectivity issues")
        
        # Prioritize fixes
        if import_errors:
            error_analysis["fix_priorities"].append("HIGH: Fix dependency installation issues")
        
        if attr_errors:
            error_analysis["fix_priorities"].append("MEDIUM: Review API compatibility and imports")
        
        return error_analysis
    
    def _investigate_performance(self) -> Dict[str, Any]:
        """Investigate system performance characteristics."""
        
        performance_metrics = {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_io": {},
            "network_io": {},
            "python_performance": {}
        }
        
        try:
            # Get disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                performance_metrics["disk_io"] = {
                    "read_bytes": disk_io.read_bytes,
                    "write_bytes": disk_io.write_bytes
                }
            
            # Get network I/O
            network_io = psutil.net_io_counters()
            if network_io:
                performance_metrics["network_io"] = {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv
                }
            
            # Python-specific performance
            import gc
            performance_metrics["python_performance"] = {
                "garbage_collection_counts": gc.get_count(),
                "recursion_limit": sys.getrecursionlimit(),
                "thread_count": len(sys._current_frames())
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            performance_metrics["error"] = str(e)
        
        return performance_metrics
    
    def _capture_api_error(self, error: Exception, component: str, context: Dict[str, Any] = None):
        """Capture detailed API error information."""
        
        # Get stack trace details
        tb = traceback.extract_tb(error.__traceback__)
        last_frame = tb[-1] if tb else None
        
        error_detail = APIErrorDetail(
            error_id=f"error_{len(self.api_errors):04d}_{int(time.time())}",
            error_type=type(error).__name__,
            error_message=str(error),
            error_code=getattr(error, 'code', None),
            component=component,
            function_name=last_frame.name if last_frame else "unknown",
            line_number=last_frame.lineno if last_frame else 0,
            file_path=last_frame.filename if last_frame else "unknown",
            stack_trace=traceback.format_exc(),
            context=context or {},
            system_state={
                "memory_usage": psutil.virtual_memory().percent,
                "cpu_usage": psutil.cpu_percent(),
                "timestamp": datetime.now().isoformat()
            },
            timestamp=datetime.now().isoformat()
        )
        
        self.api_errors.append(error_detail)
        logger.error(f"Captured API error {error_detail.error_id}: {error_detail.error_message}")
    
    def _determine_overall_status(self) -> str:
        """Determine overall system health status."""
        
        critical_issues = 0
        warnings = 0
        
        # Check diagnostics
        for diagnostic in self.diagnostics:
            if diagnostic.status == "critical":
                critical_issues += 1
            elif diagnostic.status in ["error", "warning"]:
                warnings += 1
        
        # Check dependencies
        missing_required = sum(
            1 for dep in self.dependency_statuses 
            if dep.status == "missing" and dep.requirements.get("required", False)
        )
        
        critical_issues += missing_required
        
        # Determine status
        if critical_issues > 0:
            return "critical"
        elif warnings > 2:
            return "warning"
        elif warnings > 0:
            return "healthy_with_warnings"
        else:
            return "healthy"
    
    def _generate_system_recommendations(self) -> List[str]:
        """Generate comprehensive system recommendations."""
        
        recommendations = []
        
        # Dependency recommendations
        missing_deps = [dep for dep in self.dependency_statuses if dep.status == "missing"]
        if missing_deps:
            required_missing = [dep.name for dep in missing_deps if dep.requirements.get("required", False)]
            optional_missing = [dep.name for dep in missing_deps if not dep.requirements.get("required", False)]
            
            if required_missing:
                recommendations.append(f"CRITICAL: Install required dependencies: {', '.join(required_missing)}")
            
            if optional_missing:
                recommendations.append(f"OPTIONAL: Consider installing optional dependencies: {', '.join(optional_missing)}")
        
        # Component recommendations
        failed_components = [d for d in self.diagnostics if d.status in ["critical", "error"]]
        if failed_components:
            recommendations.append(f"Fix failed components: {', '.join(d.component for d in failed_components)}")
        
        # API error recommendations
        if self.api_errors:
            error_types = set(e.error_type for e in self.api_errors)
            recommendations.append(f"Address API errors: {', '.join(error_types)}")
        
        # Performance recommendations
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 80:
            recommendations.append(f"HIGH memory usage ({memory_usage:.1f}%) - consider optimization")
        
        return recommendations
    
    def save_investigation_report(self, health_report: Dict[str, Any], output_path: Optional[Path] = None):
        """Save comprehensive investigation report."""
        
        if output_path is None:
            output_path = Path("tests/data/investigation_reports")
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / f"health_investigation_{self.investigation_id}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(health_report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Investigation report saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save investigation report: {e}")
            raise


# Convenience functions for testing framework integration
def investigate_api_error() -> Dict[str, Any]:
    """Convenience function to investigate API errors."""
    investigator = APIErrorInvestigator()
    return investigator.investigate_system_health()


def run_quick_diagnostic() -> Dict[str, str]:
    """Run a quick diagnostic check."""
    investigator = APIErrorInvestigator()
    
    # Quick checks
    quick_results = {}
    
    try:
        import pysrt
        quick_results["pysrt"] = "OK"
    except Exception as e:
        quick_results["pysrt"] = f"ERROR: {e}"
    
    try:
        from post_processors.sanskrit_post_processor import SanskritPostProcessor
        processor = SanskritPostProcessor()
        quick_results["sanskrit_post_processor"] = "OK"
    except Exception as e:
        quick_results["sanskrit_post_processor"] = f"ERROR: {e}"
    
    try:
        from utils.advanced_text_normalizer import AdvancedTextNormalizer
        normalizer = AdvancedTextNormalizer()
        quick_results["advanced_text_normalizer"] = "OK"
    except Exception as e:
        quick_results["advanced_text_normalizer"] = f"ERROR: {e}"
    
    try:
        from utils.mcp_client import create_mcp_client
        client = create_mcp_client()
        quick_results["mcp_client"] = "OK"
    except Exception as e:
        quick_results["mcp_client"] = f"ERROR: {e}"
    
    return quick_results


if __name__ == "__main__":
    # Run comprehensive API error investigation
    print("Running Comprehensive API Error Investigation...")
    
    investigator = APIErrorInvestigator()
    health_report = investigator.investigate_system_health()
    
    print(f"\nInvestigation ID: {health_report['investigation_id']}")
    print(f"Overall Status: {health_report['overall_status']}")
    print(f"Total API Errors: {health_report['api_error_analysis']['total_errors']}")
    
    # Show dependency status
    print(f"\nDependency Status:")
    dep_analysis = health_report['dependency_analysis']
    print(f"  Installed: {dep_analysis['installed_count']}/{dep_analysis['total_dependencies']}")
    print(f"  Missing: {dep_analysis['missing_count']}")
    print(f"  Errors: {dep_analysis['error_count']}")
    
    # Show component status
    print(f"\nComponent Status:")
    comp_analysis = health_report['component_diagnostics']
    print(f"  Working: {comp_analysis['working_components']}/{comp_analysis['total_components']}")
    print(f"  Failed: {comp_analysis['failed_components']}")
    
    # Show recommendations
    if health_report['recommendations']:
        print(f"\nRecommendations:")
        for i, rec in enumerate(health_report['recommendations'][:5], 1):
            print(f"  {i}. {rec}")
    
    # Save report
    report_path = investigator.save_investigation_report(health_report)
    print(f"\nDetailed report saved to: {report_path}")