"""
Import Standardization Utility for Story 5.3 Task 2.

This module provides tools to standardize import patterns across the codebase
according to PEP 8 and project conventions.
"""

import re
import ast
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass


@dataclass
class ImportStandardizationResult:
    """Result of import standardization operation."""
    file_path: str
    original_imports: List[str]
    standardized_imports: List[str]
    changes_made: int
    issues_found: List[str]
    compliance_score: float


class ImportStandardizer:
    """
    Import standardization utility that enforces consistent import patterns.
    
    Standards enforced:
    1. PEP 8 import order: standard library, third-party, local imports
    2. Alphabetical sorting within each group
    3. Consistent import styles (prefer absolute imports)
    4. Remove duplicate imports
    5. Group similar imports
    """
    
    def __init__(self):
        self.standard_library_modules = {
            'os', 'sys', 'time', 'datetime', 'json', 'csv', 'pickle', 're', 
            'math', 'string', 'logging', 'threading', 'multiprocessing',
            'pathlib', 'collections', 'functools', 'itertools', 'uuid',
            'hashlib', 'subprocess', 'tempfile', 'shutil', 'inspect',
            'traceback', 'statistics', 'asyncio', 'urllib', 'io',
            'copy', 'ast', 'importlib', 'contextlib', 'abc',
            'weakref', 'signal', 'cProfile', 'pstats', 'gc', 'tracemalloc',
            'unicodedata', 'random'
        }
        
        self.third_party_modules = {
            'numpy', 'pandas', 'scipy', 'click', 'tqdm', 'yaml', 'pysrt',
            'fuzzywuzzy', 'httpx', 'websockets', 'pydantic', 'psutil',
            'inltk', 'indicnlp', 'transformers', 'torch', 'sanskrit_parser',
            'indic_transliteration', 'chardet', 'Levenshtein', 'structlog',
            'mcp'
        }
        
        self.project_modules = {
            'utils', 'post_processors', 'sanskrit_hindi_identifier',
            'contextual_modeling', 'scripture_processing', 'monitoring',
            'ner_module', 'qa_module', 'enhancement_integration',
            'research_integration', 'src'
        }
    
    def standardize_file_imports(self, file_path: Path) -> ImportStandardizationResult:
        """Standardize imports in a single Python file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_imports = self._extract_imports(content)
        standardized_imports = self._standardize_import_list(original_imports)
        
        # Calculate changes and compliance
        changes_made = len([i for i, (orig, std) in enumerate(zip(original_imports, standardized_imports)) if orig != std])
        issues_found = self._identify_issues(original_imports)
        compliance_score = self._calculate_compliance_score(original_imports, issues_found)
        
        return ImportStandardizationResult(
            file_path=str(file_path),
            original_imports=original_imports,
            standardized_imports=standardized_imports,
            changes_made=changes_made,
            issues_found=issues_found,
            compliance_score=compliance_score
        )
    
    def standardize_project_imports(self, src_dir: Path = None) -> Dict[str, ImportStandardizationResult]:
        """Standardize imports across the entire project."""
        if src_dir is None:
            src_dir = Path('src')
        
        results = {}
        python_files = list(src_dir.rglob('*.py'))
        
        for py_file in python_files:
            if py_file.name != '__init__.py':  # Skip __init__.py files
                try:
                    result = self.standardize_file_imports(py_file)
                    results[str(py_file)] = result
                except Exception as e:
                    # Create error result
                    results[str(py_file)] = ImportStandardizationResult(
                        file_path=str(py_file),
                        original_imports=[],
                        standardized_imports=[],
                        changes_made=0,
                        issues_found=[f"Error processing file: {e}"],
                        compliance_score=0.0
                    )
        
        return results
    
    def apply_import_standardization(self, file_path: Path, backup: bool = True) -> bool:
        """Apply import standardization to a file, modifying it in place."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if backup:
                backup_path = file_path.with_suffix(f'{file_path.suffix}.backup')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Extract and standardize imports
            lines = content.split('\n')
            import_section_start, import_section_end = self._find_import_section(lines)
            
            if import_section_start is not None and import_section_end is not None:
                original_imports = lines[import_section_start:import_section_end + 1]
                standardized_imports = self._standardize_import_list(original_imports)
                
                # Replace import section
                new_lines = (
                    lines[:import_section_start] + 
                    standardized_imports + 
                    lines[import_section_end + 1:]
                )
                
                new_content = '\n'.join(new_lines)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Error applying standardization to {file_path}: {e}")
            return False
    
    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements from file content."""
        try:
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = [alias.name + (f" as {alias.asname}" if alias.asname else "") for alias in node.names]
                    if len(names) == 1:
                        imports.append(f"from {module} import {names[0]}")
                    else:
                        imports.append(f"from {module} import {', '.join(names)}")
            
            return imports
            
        except SyntaxError:
            # Fallback to regex-based extraction for files with syntax issues
            return self._extract_imports_regex(content)
    
    def _extract_imports_regex(self, content: str) -> List[str]:
        """Fallback regex-based import extraction."""
        import_pattern = re.compile(r'^(import\s+\S+|from\s+\S+\s+import\s+.+)$', re.MULTILINE)
        return import_pattern.findall(content)
    
    def _standardize_import_list(self, imports: List[str]) -> List[str]:
        """Standardize a list of import statements."""
        # Categorize imports
        stdlib_imports = []
        third_party_imports = []
        local_imports = []
        
        for imp in imports:
            category = self._categorize_import(imp)
            if category == 'stdlib':
                stdlib_imports.append(imp)
            elif category == 'third_party':
                third_party_imports.append(imp)
            else:
                local_imports.append(imp)
        
        # Sort each category
        stdlib_imports.sort()
        third_party_imports.sort()
        local_imports.sort()
        
        # Combine with blank lines between groups
        result = []
        if stdlib_imports:
            result.extend(stdlib_imports)
            if third_party_imports or local_imports:
                result.append("")
        
        if third_party_imports:
            result.extend(third_party_imports)
            if local_imports:
                result.append("")
        
        if local_imports:
            result.extend(local_imports)
        
        return result
    
    def _categorize_import(self, import_stmt: str) -> str:
        """Categorize an import statement as stdlib, third_party, or local."""
        # Extract the base module name
        if import_stmt.startswith('import '):
            module = import_stmt.split()[1].split('.')[0]
        elif import_stmt.startswith('from '):
            module_part = import_stmt.split()[1]
            if module_part == '.':
                return 'local'  # Relative import
            module = module_part.split('.')[0]
        else:
            return 'local'
        
        if module in self.standard_library_modules:
            return 'stdlib'
        elif module in self.third_party_modules:
            return 'third_party'
        else:
            return 'local'
    
    def _identify_issues(self, imports: List[str]) -> List[str]:
        """Identify import-related issues."""
        issues = []
        
        # Check for duplicates
        unique_imports = set()
        for imp in imports:
            if imp in unique_imports:
                issues.append(f"Duplicate import: {imp}")
            unique_imports.add(imp)
        
        # Check for relative imports (could be flagged based on project policy)
        for imp in imports:
            if imp.startswith('from .'):
                issues.append(f"Relative import found: {imp}")
        
        return issues
    
    def _calculate_compliance_score(self, imports: List[str], issues: List[str]) -> float:
        """Calculate import compliance score (0.0 to 1.0)."""
        if not imports:
            return 1.0
        
        # Basic compliance scoring
        score = 1.0
        
        # Deduct for issues
        score -= len(issues) * 0.1
        
        # Check if imports are properly sorted (basic check)
        standardized = self._standardize_import_list(imports)
        if imports != standardized:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _find_import_section(self, lines: List[str]) -> Tuple[int, int]:
        """Find the start and end of the import section."""
        import_start = None
        import_end = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')) and not stripped.startswith('#'):
                if import_start is None:
                    import_start = i
                import_end = i
            elif import_start is not None and stripped and not stripped.startswith('#'):
                # Non-import, non-empty, non-comment line after imports
                break
        
        return import_start, import_end
    
    def generate_standardization_report(self, results: Dict[str, ImportStandardizationResult]) -> str:
        """Generate a comprehensive import standardization report."""
        total_files = len(results)
        total_changes = sum(r.changes_made for r in results.values())
        avg_compliance = sum(r.compliance_score for r in results.values()) / total_files if total_files > 0 else 0
        
        files_with_issues = [r for r in results.values() if r.issues_found]
        
        report = [
            "# Import Standardization Report",
            "",
            f"## Summary",
            f"- Total files analyzed: {total_files}",
            f"- Total changes needed: {total_changes}",
            f"- Average compliance score: {avg_compliance:.2f}",
            f"- Files with issues: {len(files_with_issues)}",
            "",
            "## Files Requiring Attention",
        ]
        
        for result in sorted(files_with_issues, key=lambda x: x.compliance_score):
            report.append(f"### {result.file_path}")
            report.append(f"- Compliance score: {result.compliance_score:.2f}")
            report.append(f"- Changes needed: {result.changes_made}")
            if result.issues_found:
                report.append("- Issues:")
                for issue in result.issues_found:
                    report.append(f"  - {issue}")
            report.append("")
        
        return "\n".join(report)