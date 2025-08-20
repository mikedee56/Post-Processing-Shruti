"""
Dependency validation and compatibility checking utilities.

This module provides comprehensive dependency validation, compatibility checking,
and resolution strategies for the ASR Post-Processing system.

Author: Dev Agent (Story 5.3)
Version: 1.0
"""

import sys
import importlib
import pkg_resources
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class DependencyStatus(Enum):
    """Dependency status levels."""
    WORKING = "working"
    WARNING = "warning"
    ERROR = "error"
    MISSING = "missing"

@dataclass
class DependencyResult:
    """Result of dependency validation."""
    name: str
    status: DependencyStatus
    version: Optional[str]
    message: str
    details: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None

@dataclass
class CompatibilityMatrix:
    """Compatibility matrix for dependency versions."""
    package: str
    compatible_versions: List[str]
    incompatible_versions: List[str]
    recommended_version: str
    notes: Optional[str] = None

class DependencyValidator:
    """
    Comprehensive dependency validation and compatibility checking system.
    
    Validates all project dependencies, checks for compatibility issues,
    and provides resolution recommendations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize dependency validator."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Define critical dependencies and their requirements
        self.critical_dependencies = {
            'pandas': {
                'min_version': '2.0.0',
                'import_test': 'import pandas as pd; pd.DataFrame()',
                'description': 'Core data processing library'
            },
            'numpy': {
                'min_version': '1.24.0',
                'import_test': 'import numpy as np; np.array([1,2,3])',
                'description': 'Numerical computing library'
            },
            'pysrt': {
                'min_version': '1.1.2',
                'import_test': 'import pysrt',
                'description': 'SRT file processing'
            },
            'pyyaml': {
                'min_version': '6.0',
                'import_test': 'import yaml',
                'description': 'YAML configuration parsing'
            },
            'fuzzywuzzy': {
                'min_version': '0.18.0',
                'import_test': 'from fuzzywuzzy import fuzz',
                'description': 'Fuzzy string matching'
            },
            'structlog': {
                'min_version': '23.0.0',
                'import_test': 'import structlog',
                'description': 'Structured logging'
            }
        }
        
        # Define problematic dependencies and their fixes
        self.problematic_dependencies = {
            'indic_transliteration': {
                'issues': ['Module import errors', 'sanskritnumeral not available'],
                'workarounds': ['Import specific modules', 'Use alternative transliteration'],
                'test_imports': [
                    'from indic_transliteration.sanskritnumerals import digit_2_word',
                    'from indic_transliteration.sanskritnumerals import word_2_digit'
                ]
            },
            'indic-nlp-library': {
                'issues': ['Version compatibility', 'Unicode handling'],
                'workarounds': ['Use specific version pinning', 'Handle encoding carefully'],
                'test_imports': [
                    'from indicnlp.common import common_utils'
                ]
            },
            'sanskrit_parser': {
                'issues': ['Heavy memory usage', 'Slow initialization'],
                'workarounds': ['Lazy loading', 'Memory optimization'],
                'test_imports': [
                    'import sanskrit_parser'
                ]
            }
        }
        
        # Compatibility matrix
        self.compatibility_matrix = [
            CompatibilityMatrix(
                package="indic-transliteration",
                compatible_versions=["2.3.73", "2.3.70"],
                incompatible_versions=["<2.0.0"],
                recommended_version="2.3.73",
                notes="Use specific import paths for modules"
            ),
            CompatibilityMatrix(
                package="indic-nlp-library", 
                compatible_versions=["0.92", "0.91"],
                incompatible_versions=["<0.80"],
                recommended_version="0.92",
                notes="Ensure proper Unicode handling"
            )
        ]
    
    def validate_all_dependencies(self) -> Dict[str, DependencyResult]:
        """Validate all project dependencies."""
        results = {}
        
        # Validate critical dependencies
        for name, config in self.critical_dependencies.items():
            results[name] = self._validate_dependency(name, config)
        
        # Validate problematic dependencies with special handling
        for name, config in self.problematic_dependencies.items():
            results[name] = self._validate_problematic_dependency(name, config)
        
        return results
    
    def _validate_dependency(self, name: str, config: Dict) -> DependencyResult:
        """Validate a single dependency."""
        try:
            # Check if package is installed
            try:
                version = pkg_resources.get_distribution(name).version
            except pkg_resources.DistributionNotFound:
                return DependencyResult(
                    name=name,
                    status=DependencyStatus.MISSING,
                    version=None,
                    message=f"Package {name} is not installed",
                    recommendations=[f"Install with: pip install {name}>={config['min_version']}"]
                )
            
            # Check version requirement
            if config.get('min_version'):
                if pkg_resources.parse_version(version) < pkg_resources.parse_version(config['min_version']):
                    return DependencyResult(
                        name=name,
                        status=DependencyStatus.ERROR,
                        version=version,
                        message=f"Version {version} < required {config['min_version']}",
                        recommendations=[f"Upgrade with: pip install {name}>={config['min_version']}"]
                    )
            
            # Test import functionality
            if config.get('import_test'):
                try:
                    exec(config['import_test'])
                    import_status = "working"
                    import_message = "Import test passed"
                except Exception as e:
                    import_status = "error"
                    import_message = f"Import test failed: {str(e)}"
                    return DependencyResult(
                        name=name,
                        status=DependencyStatus.ERROR,
                        version=version,
                        message=import_message,
                        recommendations=["Check installation integrity", "Reinstall package"]
                    )
            
            return DependencyResult(
                name=name,
                status=DependencyStatus.WORKING,
                version=version,
                message="Dependency working correctly"
            )
            
        except Exception as e:
            return DependencyResult(
                name=name,
                status=DependencyStatus.ERROR,
                version=None,
                message=f"Validation failed: {str(e)}",
                details={'traceback': traceback.format_exc()}
            )
    
    def _validate_problematic_dependency(self, name: str, config: Dict) -> DependencyResult:
        """Validate problematic dependency with special handling."""
        try:
            # Check if package is installed
            try:
                version = pkg_resources.get_distribution(name.replace('_', '-')).version
            except pkg_resources.DistributionNotFound:
                return DependencyResult(
                    name=name,
                    status=DependencyStatus.MISSING,
                    version=None,
                    message=f"Package {name} is not installed"
                )
            
            # Test specific imports that are known to be problematic
            import_results = []
            for test_import in config.get('test_imports', []):
                try:
                    exec(test_import)
                    import_results.append(f"✓ {test_import}")
                except Exception as e:
                    import_results.append(f"✗ {test_import}: {str(e)}")
            
            # Determine status based on import results
            failed_imports = [r for r in import_results if r.startswith("✗")]
            
            if not failed_imports:
                status = DependencyStatus.WORKING
                message = "All import tests passed"
            elif len(failed_imports) < len(import_results):
                status = DependencyStatus.WARNING  
                message = f"Partial functionality: {len(failed_imports)} failed imports"
            else:
                status = DependencyStatus.ERROR
                message = "All import tests failed"
            
            return DependencyResult(
                name=name,
                status=status,
                version=version,
                message=message,
                details={
                    'import_results': import_results,
                    'known_issues': config.get('issues', []),
                    'workarounds': config.get('workarounds', [])
                },
                recommendations=config.get('workarounds', [])
            )
            
        except Exception as e:
            return DependencyResult(
                name=name,
                status=DependencyStatus.ERROR,
                version=None,
                message=f"Validation failed: {str(e)}",
                details={'traceback': traceback.format_exc()}
            )
    
    def generate_dependency_report(self) -> Dict[str, Any]:
        """Generate comprehensive dependency report."""
        validation_results = self.validate_all_dependencies()
        
        # Categorize results
        working = {k: v for k, v in validation_results.items() if v.status == DependencyStatus.WORKING}
        warnings = {k: v for k, v in validation_results.items() if v.status == DependencyStatus.WARNING}
        errors = {k: v for k, v in validation_results.items() if v.status == DependencyStatus.ERROR}
        missing = {k: v for k, v in validation_results.items() if v.status == DependencyStatus.MISSING}
        
        # Generate recommendations
        recommendations = []
        for result in validation_results.values():
            if result.recommendations:
                recommendations.extend(result.recommendations)
        
        # System compatibility check
        python_version = sys.version_info
        platform_compatible = python_version >= (3, 10)
        
        report = {
            'system_info': {
                'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                'platform_compatible': platform_compatible,
                'total_dependencies': len(validation_results)
            },
            'summary': {
                'working': len(working),
                'warnings': len(warnings), 
                'errors': len(errors),
                'missing': len(missing)
            },
            'status_details': {
                'working': working,
                'warnings': warnings,
                'errors': errors,
                'missing': missing
            },
            'recommendations': list(set(recommendations)),  # Remove duplicates
            'compatibility_matrix': [
                {
                    'package': matrix.package,
                    'recommended_version': matrix.recommended_version,
                    'compatible_versions': matrix.compatible_versions,
                    'notes': matrix.notes
                }
                for matrix in self.compatibility_matrix
            ]
        }
        
        return report
    
    def get_installation_script(self) -> str:
        """Generate installation script for fixing dependencies."""
        validation_results = self.validate_all_dependencies()
        
        commands = ["# Dependency installation script generated by Story 5.3"]
        commands.append("# Run this script to fix dependency issues")
        commands.append("")
        
        # Upgrade pip first
        commands.append("python -m pip install --upgrade pip")
        commands.append("")
        
        # Install missing packages
        missing_packages = [
            name for name, result in validation_results.items()
            if result.status == DependencyStatus.MISSING
        ]
        
        if missing_packages:
            commands.append("# Install missing packages")
            for package in missing_packages:
                if package in self.critical_dependencies:
                    min_version = self.critical_dependencies[package]['min_version']
                    commands.append(f"python -m pip install {package}>={min_version}")
                else:
                    commands.append(f"python -m pip install {package}")
            commands.append("")
        
        # Fix version conflicts
        error_packages = [
            name for name, result in validation_results.items()
            if result.status == DependencyStatus.ERROR and result.version
        ]
        
        if error_packages:
            commands.append("# Fix version conflicts")
            for package in error_packages:
                if package in self.critical_dependencies:
                    min_version = self.critical_dependencies[package]['min_version']
                    commands.append(f"python -m pip install --upgrade {package}>={min_version}")
            commands.append("")
        
        # Install compatibility matrix recommendations
        commands.append("# Install recommended versions for compatibility")
        for matrix in self.compatibility_matrix:
            commands.append(f"python -m pip install {matrix.package}=={matrix.recommended_version}")
        
        commands.append("")
        commands.append("# Verify installation")
        commands.append("python -c \"from src.utils.dependency_validator import DependencyValidator; validator = DependencyValidator(); report = validator.generate_dependency_report(); print('Validation complete:', report['summary'])\"")
        
        return "\n".join(commands)
    
    def fix_common_issues(self) -> Dict[str, Any]:
        """Apply automatic fixes for common dependency issues."""
        fixes_applied = {}
        
        # Fix IndicNLP import issues by testing alternative imports
        try:
            # Try the problematic import
            from indic_transliteration import sanskritnumeral
            fixes_applied['indic_transliteration'] = "Standard import working"
        except ImportError:
            try:
                # Try alternative import path
                from indic_transliteration.sanskritnumerals import digit_2_word, word_2_digit
                fixes_applied['indic_transliteration'] = "Alternative import path working"
            except ImportError:
                fixes_applied['indic_transliteration'] = "Import fixes failed - manual intervention required"
        
        # Fix Unicode encoding issues
        try:
            import os
            if os.name == 'nt':  # Windows
                # Set UTF-8 encoding for Windows
                import sys
                if hasattr(sys.stdout, 'reconfigure'):
                    sys.stdout.reconfigure(encoding='utf-8')
                fixes_applied['unicode_encoding'] = "UTF-8 encoding configured for Windows"
        except Exception as e:
            fixes_applied['unicode_encoding'] = f"Unicode fix failed: {str(e)}"
        
        return fixes_applied

# Utility functions for dependency management
def validate_dependencies() -> Dict[str, DependencyResult]:
    """Quick dependency validation function."""
    validator = DependencyValidator()
    return validator.validate_all_dependencies()

def generate_dependency_report() -> Dict[str, Any]:
    """Generate dependency report."""
    validator = DependencyValidator()
    return validator.generate_dependency_report()

def create_requirements_lockfile() -> str:
    """Create a locked requirements file with current versions."""
    import pkg_resources
    
    installed_packages = {pkg.project_name: pkg.version for pkg in pkg_resources.working_set}
    
    # Define our project dependencies
    project_dependencies = [
        'pandas', 'numpy', 'scipy', 'pyyaml', 'pysrt', 'fuzzywuzzy',
        'python-Levenshtein', 'rapidfuzz', 'structlog', 'click', 'tqdm',
        'pytest', 'pytest-cov', 'indic-transliteration', 'indic-nlp-library',
        'sanskrit_parser', 'chardet', 'colorama'
    ]
    
    lockfile_content = ["# Generated lockfile - Story 5.3 dependency resolution"]
    lockfile_content.append("# This file pins exact versions for reproducible builds")
    lockfile_content.append("")
    
    for package in project_dependencies:
        normalized_name = package.replace('_', '-')
        if normalized_name in installed_packages:
            version = installed_packages[normalized_name]
            lockfile_content.append(f"{package}=={version}")
        else:
            lockfile_content.append(f"# {package} - NOT INSTALLED")
    
    return "\n".join(lockfile_content)

# Console-safe printing utility
def safe_print(text: str) -> str:
    """Convert unicode characters to console-safe equivalents."""
    replacements = {
        '✓': '[PASS]',
        '⚠': '[WARN]',
        '✗': '[FAIL]',
        '?': '[MISS]'
    }
    
    safe_text = text
    for unicode_char, replacement in replacements.items():
        safe_text = safe_text.replace(unicode_char, replacement)
    
    return safe_text

# Test function
def test_dependency_validator():
    """Test the dependency validator functionality."""
    validator = DependencyValidator()
    
    print("Running dependency validation tests...")
    results = validator.validate_all_dependencies()
    
    print(f"Validated {len(results)} dependencies")
    
    for name, result in results.items():
        status_symbol = {
            DependencyStatus.WORKING: "[PASS]",
            DependencyStatus.WARNING: "[WARN]", 
            DependencyStatus.ERROR: "[FAIL]",
            DependencyStatus.MISSING: "[MISS]"
        }[result.status]
        
        print(f"{status_symbol} {name} ({result.version or 'unknown'}): {result.message}")

if __name__ == "__main__":
    test_dependency_validator()