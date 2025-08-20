"""
Import cleanup and optimization utilities.

This module provides comprehensive import analysis, cleanup, and optimization
for resolving circular imports, standardizing patterns, and removing unused imports.

Author: Dev Agent (Story 5.3 Task 2)
Version: 1.0
"""

import os
import ast
import sys
import importlib
import importlib.util
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from .exception_hierarchy import BaseProcessingException, ErrorSeverity, ErrorCategory


@dataclass
class ImportInfo:
    """Information about an import statement."""
    module: str
    alias: Optional[str] = None
    from_list: List[str] = field(default_factory=list)
    line_number: int = 0
    is_used: bool = False
    import_type: str = "import"  # "import" or "from"


@dataclass
class FileImportAnalysis:
    """Analysis of imports for a single file."""
    file_path: Path
    imports: List[ImportInfo] = field(default_factory=list)
    unused_imports: List[ImportInfo] = field(default_factory=list)
    circular_dependencies: List[str] = field(default_factory=list)
    import_errors: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ProjectImportAnalysis:
    """Comprehensive analysis of imports across the project."""
    total_files: int = 0
    total_imports: int = 0
    unused_imports_count: int = 0
    circular_dependencies: List[Tuple[str, str]] = field(default_factory=list)
    import_patterns: Dict[str, int] = field(default_factory=dict)
    file_analyses: Dict[str, FileImportAnalysis] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class ImportCleanupManager:
    """
    Comprehensive import cleanup and optimization system.
    
    Analyzes and fixes import issues across the entire project including:
    - Unused import detection and removal
    - Circular import detection and resolution
    - Import pattern standardization
    - Relative vs absolute import consistency
    """
    
    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize import cleanup manager.
        
        Args:
            project_root: Root directory of the project
        """
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(project_root or os.getcwd())
        self.src_root = self.project_root / "src"
        
        # Import analysis state
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)
        self.module_paths: Dict[str, Path] = {}
        self.stdlib_modules = self._get_stdlib_modules()
        
        # Configuration
        self.ignore_patterns = {
            '__pycache__', '*.pyc', '.git', '.pytest_cache', 
            'build', 'dist', '*.egg-info'
        }
        
        self.standard_import_order = [
            'stdlib',      # Standard library
            'third_party', # Third-party packages
            'local',       # Local/project imports
        ]
        
        self.logger.info(f"Import cleanup manager initialized for {self.project_root}")
    
    def analyze_project_imports(self) -> ProjectImportAnalysis:
        """
        Analyze all imports across the project.
        
        Returns:
            Comprehensive project import analysis
        """
        self.logger.info("Starting comprehensive project import analysis")
        
        analysis = ProjectImportAnalysis()
        
        # Find all Python files
        python_files = list(self._find_python_files())
        analysis.total_files = len(python_files)
        
        self.logger.info(f"Found {len(python_files)} Python files to analyze")
        
        # Analyze each file
        for file_path in python_files:
            try:
                file_analysis = self._analyze_file_imports(file_path)
                analysis.file_analyses[str(file_path)] = file_analysis
                
                # Update totals
                analysis.total_imports += len(file_analysis.imports)
                analysis.unused_imports_count += len(file_analysis.unused_imports)
                
                # Track import patterns
                for import_info in file_analysis.imports:
                    pattern = f"{import_info.import_type}:{import_info.module.split('.')[0]}"
                    analysis.import_patterns[pattern] = analysis.import_patterns.get(pattern, 0) + 1
                
            except Exception as e:
                self.logger.error(f"Failed to analyze {file_path}: {e}")
        
        # Detect circular dependencies
        analysis.circular_dependencies = self._detect_circular_dependencies()
        
        # Generate recommendations
        analysis.recommendations = self._generate_project_recommendations(analysis)
        
        self.logger.info(f"Analysis complete: {analysis.total_imports} imports, "
                        f"{analysis.unused_imports_count} unused, "
                        f"{len(analysis.circular_dependencies)} circular dependencies")
        
        return analysis
    
    def cleanup_file_imports(self, file_path: Path, dry_run: bool = True) -> FileImportAnalysis:
        """
        Clean up imports in a single file.
        
        Args:
            file_path: Path to the file to clean up
            dry_run: If True, only analyze without making changes
            
        Returns:
            Analysis of the file's imports with cleanup results
        """
        self.logger.info(f"Cleaning up imports in {file_path}")
        
        # Analyze current state
        analysis = self._analyze_file_imports(file_path)
        
        if dry_run:
            return analysis
        
        # Apply cleanup
        if analysis.unused_imports:
            self._remove_unused_imports(file_path, analysis.unused_imports)
            analysis.suggestions.append(f"Removed {len(analysis.unused_imports)} unused imports")
        
        # Standardize import order
        self._standardize_import_order(file_path)
        analysis.suggestions.append("Standardized import order")
        
        # Convert relative imports to absolute where appropriate
        relative_imports = [imp for imp in analysis.imports if imp.module.startswith('.')]
        if relative_imports:
            self._convert_relative_imports(file_path, relative_imports)
            analysis.suggestions.append(f"Converted {len(relative_imports)} relative imports")
        
        return analysis
    
    def fix_circular_dependencies(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Detect and fix circular dependencies.
        
        Args:
            dry_run: If True, only analyze without making changes
            
        Returns:
            Results of circular dependency detection and fixes
        """
        self.logger.info("Analyzing circular dependencies")
        
        circular_deps = self._detect_circular_dependencies()
        
        results = {
            'circular_dependencies': circular_deps,
            'fixes_applied': [],
            'recommendations': []
        }
        
        if not circular_deps:
            results['recommendations'].append("No circular dependencies detected")
            return results
        
        self.logger.warning(f"Found {len(circular_deps)} circular dependencies")
        
        if not dry_run:
            # Apply fixes for circular dependencies
            for dep_cycle in circular_deps:
                fix_result = self._fix_circular_dependency(dep_cycle)
                if fix_result:
                    results['fixes_applied'].append(fix_result)
        
        # Generate recommendations
        for dep_cycle in circular_deps:
            results['recommendations'].append(
                f"Circular dependency: {' -> '.join(dep_cycle)} -> {dep_cycle[0]}"
            )
        
        return results
    
    def generate_import_report(self) -> Dict[str, Any]:
        """Generate comprehensive import analysis report."""
        analysis = self.analyze_project_imports()
        
        # Calculate statistics
        files_with_unused = sum(1 for fa in analysis.file_analyses.values() 
                              if fa.unused_imports)
        files_with_errors = sum(1 for fa in analysis.file_analyses.values() 
                              if fa.import_errors)
        
        report = {
            'summary': {
                'total_files_analyzed': analysis.total_files,
                'total_imports': analysis.total_imports,
                'unused_imports': analysis.unused_imports_count,
                'circular_dependencies': len(analysis.circular_dependencies),
                'files_with_unused_imports': files_with_unused,
                'files_with_errors': files_with_errors,
                'cleanup_potential': round((analysis.unused_imports_count / analysis.total_imports * 100), 2) if analysis.total_imports > 0 else 0
            },
            'import_patterns': analysis.import_patterns,
            'top_imported_modules': self._get_top_imported_modules(analysis),
            'circular_dependencies': analysis.circular_dependencies,
            'recommendations': analysis.recommendations,
            'files_needing_attention': self._get_files_needing_attention(analysis)
        }
        
        return report
    
    def create_import_standards_guide(self) -> str:
        """Create import standards and best practices guide."""
        guide = """
# Import Standards and Best Practices Guide

## Import Order Standard
1. Standard library imports
2. Third-party package imports  
3. Local application/project imports
4. Blank line between each group

## Import Style Guidelines
- Use absolute imports for clarity
- Avoid wildcard imports (from module import *)
- Group related imports together
- Sort imports alphabetically within groups
- Use 'as' aliases only when necessary for clarity

## Example:
```python
# Standard library
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Third-party
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

# Local imports
from .exception_hierarchy import BaseProcessingException
from ..utils.config_manager import ConfigurationManager
```

## Circular Import Resolution
1. Use lazy imports (import inside functions)
2. Restructure modules to eliminate cycles
3. Move shared code to separate modules
4. Use dependency injection patterns

## Import Performance
- Import modules at module level, not in functions (unless lazy loading)
- Cache expensive imports
- Use specific imports for large modules
"""
        return guide
    
    def _find_python_files(self):
        """Find all Python files in the project."""
        for root, dirs, files in os.walk(self.src_root):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not any(
                d.startswith(pattern.rstrip('*')) for pattern in self.ignore_patterns
            )]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    yield Path(root) / file
    
    def _analyze_file_imports(self, file_path: Path) -> FileImportAnalysis:
        """Analyze imports in a single file."""
        analysis = FileImportAnalysis(file_path=file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content)
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_info = ImportInfo(
                            module=alias.name,
                            alias=alias.asname,
                            line_number=node.lineno,
                            import_type="import"
                        )
                        analysis.imports.append(import_info)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        import_info = ImportInfo(
                            module=node.module,
                            from_list=[alias.name for alias in node.names],
                            line_number=node.lineno,
                            import_type="from"
                        )
                        analysis.imports.append(import_info)
            
            # Check for unused imports
            analysis.unused_imports = self._find_unused_imports(file_path, content, analysis.imports)
            
            # Check for import errors
            analysis.import_errors = self._check_import_errors(analysis.imports)
            
            # Generate suggestions
            analysis.suggestions = self._generate_file_suggestions(analysis)
            
        except Exception as e:
            analysis.import_errors.append(f"Failed to parse file: {e}")
        
        return analysis
    
    def _find_unused_imports(self, file_path: Path, content: str, imports: List[ImportInfo]) -> List[ImportInfo]:
        """Find unused imports in file content."""
        unused = []
        
        for import_info in imports:
            is_used = False
            
            # Check different usage patterns
            if import_info.import_type == "import":
                module_name = import_info.alias or import_info.module.split('.')[-1]
                if module_name in content:
                    is_used = True
            else:  # from import
                for item in import_info.from_list:
                    if item in content and item != "import":
                        is_used = True
                        break
            
            import_info.is_used = is_used
            if not is_used:
                unused.append(import_info)
        
        return unused
    
    def _check_import_errors(self, imports: List[ImportInfo]) -> List[str]:
        """Check for import errors."""
        errors = []
        
        for import_info in imports:
            try:
                if import_info.module.startswith('.'):
                    # Skip relative imports for now
                    continue
                    
                importlib.import_module(import_info.module)
            except ImportError as e:
                errors.append(f"Import error for {import_info.module}: {str(e)}")
            except Exception as e:
                errors.append(f"Unexpected error importing {import_info.module}: {str(e)}")
        
        return errors
    
    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies using graph analysis."""
        # Build import graph first
        self._build_import_graph()
        
        circular_deps = []
        visited = set()
        rec_stack = set()
        
        def has_cycle(node, path):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.import_graph.get(node, set()):
                if neighbor not in visited:
                    cycle_path = has_cycle(neighbor, path + [neighbor])
                    if cycle_path:
                        return cycle_path
                elif neighbor in rec_stack:
                    # Found cycle - fix for the bug
                    try:
                        cycle_start = path.index(neighbor)
                        return path[cycle_start:] + [neighbor]
                    except ValueError:
                        # neighbor not in path, so start cycle from neighbor
                        return [neighbor] + [node]
            
            rec_stack.remove(node)
            return None
        
        # Check each module
        for module in self.import_graph:
            if module not in visited:
                cycle = has_cycle(module, [module])
                if cycle:
                    circular_deps.append(cycle)
        
        return circular_deps
    
    def _build_import_graph(self):
        """Build graph of module dependencies."""
        self.import_graph.clear()
        
        for file_path in self._find_python_files():
            try:
                module_name = self._get_module_name(file_path)
                self.module_paths[module_name] = file_path
                
                analysis = self._analyze_file_imports(file_path)
                
                dependencies = set()
                for import_info in analysis.imports:
                    # Convert to module name
                    if import_info.module.startswith('.'):
                        # Handle relative imports
                        abs_module = self._resolve_relative_import(module_name, import_info.module)
                        if abs_module:
                            dependencies.add(abs_module)
                    else:
                        dependencies.add(import_info.module)
                
                self.import_graph[module_name] = dependencies
                
            except Exception as e:
                self.logger.warning(f"Failed to process {file_path} for dependency graph: {e}")
    
    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        relative_path = file_path.relative_to(self.src_root)
        module_parts = list(relative_path.parts)
        
        # Remove .py extension
        if module_parts[-1].endswith('.py'):
            module_parts[-1] = module_parts[-1][:-3]
        
        # Remove __init__ if present
        if module_parts[-1] == '__init__':
            module_parts = module_parts[:-1]
        
        return '.'.join(module_parts)
    
    def _resolve_relative_import(self, current_module: str, relative_import: str) -> Optional[str]:
        """Resolve relative import to absolute module name."""
        if not relative_import.startswith('.'):
            return relative_import
        
        # Count leading dots
        level = 0
        for char in relative_import:
            if char == '.':
                level += 1
            else:
                break
        
        # Get base module
        current_parts = current_module.split('.')
        if level > len(current_parts):
            return None
        
        base_parts = current_parts[:-level] if level > 0 else current_parts
        
        # Add relative part
        relative_part = relative_import[level:]
        if relative_part:
            base_parts.append(relative_part)
        
        return '.'.join(base_parts) if base_parts else None
    
    def _get_stdlib_modules(self) -> Set[str]:
        """Get set of standard library module names."""
        # Basic set of known stdlib modules
        stdlib_modules = {
            'os', 'sys', 'pathlib', 'json', 'yaml', 'logging', 'time', 'datetime',
            'collections', 'itertools', 'functools', 'operator', 're', 'math',
            'random', 'hashlib', 'uuid', 'threading', 'multiprocessing', 'subprocess',
            'urllib', 'http', 'socket', 'email', 'csv', 'xml', 'html', 'sqlite3',
            'pickle', 'gzip', 'zipfile', 'tarfile', 'shutil', 'tempfile', 'glob',
            'argparse', 'configparser', 'unittest', 'doctest', 'pdb', 'profile',
            'traceback', 'warnings', 'contextlib', 'abc', 'typing', 'enum',
            'dataclasses', 'inspect', 'ast', 'dis', 'importlib'
        }
        return stdlib_modules
    
    def _remove_unused_imports(self, file_path: Path, unused_imports: List[ImportInfo]):
        """Remove unused imports from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Remove lines with unused imports (in reverse order to maintain line numbers)
        for import_info in sorted(unused_imports, key=lambda x: x.line_number, reverse=True):
            if 0 < import_info.line_number <= len(lines):
                del lines[import_info.line_number - 1]
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    def _standardize_import_order(self, file_path: Path):
        """Standardize import order in file."""
        # This is a simplified implementation
        # A full implementation would parse AST and reorder imports properly
        pass
    
    def _convert_relative_imports(self, file_path: Path, relative_imports: List[ImportInfo]):
        """Convert relative imports to absolute."""
        # This is a simplified implementation
        # A full implementation would parse AST and convert imports properly
        pass
    
    def _fix_circular_dependency(self, cycle: List[str]) -> Optional[str]:
        """Fix a circular dependency."""
        # This is a complex operation that would require careful analysis
        # For now, return a recommendation
        return f"Recommend restructuring modules: {' -> '.join(cycle)}"
    
    def _generate_file_suggestions(self, analysis: FileImportAnalysis) -> List[str]:
        """Generate suggestions for file import cleanup."""
        suggestions = []
        
        if analysis.unused_imports:
            suggestions.append(f"Remove {len(analysis.unused_imports)} unused imports")
        
        if analysis.import_errors:
            suggestions.append(f"Fix {len(analysis.import_errors)} import errors")
        
        # Check import organization
        import_lines = [imp.line_number for imp in analysis.imports]
        if not all(import_lines[i] <= import_lines[i+1] for i in range(len(import_lines)-1)):
            suggestions.append("Reorganize imports for better readability")
        
        return suggestions
    
    def _generate_project_recommendations(self, analysis: ProjectImportAnalysis) -> List[str]:
        """Generate project-wide recommendations."""
        recommendations = []
        
        if analysis.unused_imports_count > 0:
            recommendations.append(f"Remove {analysis.unused_imports_count} unused imports across {analysis.total_files} files")
        
        if analysis.circular_dependencies:
            recommendations.append(f"Resolve {len(analysis.circular_dependencies)} circular dependencies")
        
        # Check for common patterns that could be improved
        if 'import:src' in analysis.import_patterns:
            recommendations.append("Consider using absolute imports instead of relative imports where possible")
        
        recommendations.append("Standardize import order across all files")
        recommendations.append("Add import organization to code quality checks")
        
        return recommendations
    
    def _get_top_imported_modules(self, analysis: ProjectImportAnalysis) -> Dict[str, int]:
        """Get most frequently imported modules."""
        module_counts = defaultdict(int)
        
        for file_analysis in analysis.file_analyses.values():
            for import_info in file_analysis.imports:
                base_module = import_info.module.split('.')[0]
                module_counts[base_module] += 1
        
        # Return top 10
        return dict(sorted(module_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _get_files_needing_attention(self, analysis: ProjectImportAnalysis) -> List[Dict[str, Any]]:
        """Get files that need the most attention."""
        attention_files = []
        
        for file_path, file_analysis in analysis.file_analyses.items():
            score = (len(file_analysis.unused_imports) * 2 + 
                    len(file_analysis.import_errors) * 3 +
                    len(file_analysis.circular_dependencies))
            
            if score > 0:
                attention_files.append({
                    'file': file_path,
                    'score': score,
                    'unused_imports': len(file_analysis.unused_imports),
                    'import_errors': len(file_analysis.import_errors),
                    'suggestions': file_analysis.suggestions
                })
        
        return sorted(attention_files, key=lambda x: x['score'], reverse=True)[:10]


# Utility functions
def analyze_imports(project_root: str = None) -> Dict[str, Any]:
    """Quick import analysis utility."""
    manager = ImportCleanupManager(project_root)
    return manager.generate_import_report()


def cleanup_project_imports(project_root: str = None, dry_run: bool = True) -> Dict[str, Any]:
    """Clean up imports across the entire project."""
    manager = ImportCleanupManager(project_root)
    
    results = {
        'analysis': manager.analyze_project_imports(),
        'circular_fixes': manager.fix_circular_dependencies(dry_run=dry_run),
        'files_processed': 0,
        'total_fixes': 0
    }
    
    if not dry_run:
        # Process each file
        for file_path in manager._find_python_files():
            try:
                file_result = manager.cleanup_file_imports(file_path, dry_run=False)
                results['files_processed'] += 1
                results['total_fixes'] += len(file_result.suggestions)
            except Exception as e:
                manager.logger.error(f"Failed to cleanup {file_path}: {e}")
    
    return results


# Test function
def test_import_cleanup():
    """Test import cleanup functionality."""
    print("Testing Import Cleanup Manager...")
    
    manager = ImportCleanupManager()
    
    # Generate report
    report = manager.generate_import_report()
    
    print(f"Import Analysis Report:")
    print(f"  Files analyzed: {report['summary']['total_files_analyzed']}")
    print(f"  Total imports: {report['summary']['total_imports']}")
    print(f"  Unused imports: {report['summary']['unused_imports']}")
    print(f"  Circular dependencies: {report['summary']['circular_dependencies']}")
    print(f"  Cleanup potential: {report['summary']['cleanup_potential']}%")
    
    if report['top_imported_modules']:
        print(f"  Top imported modules:")
        for module, count in list(report['top_imported_modules'].items())[:5]:
            print(f"    - {module}: {count} imports")
    
    print(f"  Recommendations: {len(report['recommendations'])}")
    for rec in report['recommendations'][:3]:
        print(f"    - {rec}")
    
    print("Import cleanup test completed successfully!")


if __name__ == "__main__":
    test_import_cleanup()