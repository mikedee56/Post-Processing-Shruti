"""
Security Scanner for Production Security Monitoring
Implements vulnerability scanning and security best practices validation
"""

import os
import re
import ast
import json
import hashlib
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from enum import Enum
import subprocess
import tempfile


class VulnerabilityLevel(Enum):
    """Vulnerability severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityFinding:
    """Security vulnerability finding"""
    finding_id: str
    title: str
    description: str
    level: VulnerabilityLevel
    file_path: Optional[str]
    line_number: Optional[int]
    code_snippet: Optional[str]
    recommendation: str
    cwe_id: Optional[str] = None
    cve_id: Optional[str] = None


@dataclass
class SecurityScanResult:
    """Complete security scan results"""
    scan_id: str
    timestamp: datetime
    findings: List[SecurityFinding]
    scan_duration: float
    files_scanned: int
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary of findings by severity"""
        summary = {level.value: 0 for level in VulnerabilityLevel}
        for finding in self.findings:
            summary[finding.level.value] += 1
        return summary
    
    def has_critical_findings(self) -> bool:
        """Check if scan has critical findings"""
        return any(f.level == VulnerabilityLevel.CRITICAL for f in self.findings)


class CodeSecurityScanner:
    """Static code analysis security scanner"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Security patterns to detect
        self.security_patterns = {
            # SQL Injection patterns
            'sql_injection': [
                r'execute\s*\(\s*["\'].*%.*["\'].*\)',
                r'cursor\.execute\s*\(\s*["\'].*\+.*["\']',
                r'query\s*=.*["\'].*%.*["\']',
            ],
            
            # Command injection patterns
            'command_injection': [
                r'os\.system\s*\([^)]*\+',
                r'subprocess\.(run|call|Popen)\s*\([^)]*\+',
                r'eval\s*\(',
                r'exec\s*\(',
            ],
            
            # Hardcoded secrets
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']{8,}["\']',
                r'secret\s*=\s*["\'][^"\']{16,}["\']',
                r'api_key\s*=\s*["\'][^"\']{20,}["\']',
                r'token\s*=\s*["\'][^"\']{20,}["\']',
            ],
            
            # Insecure random
            'weak_random': [
                r'random\.random\s*\(',
                r'random\.randint\s*\(',
                r'random\.choice\s*\(',
            ],
            
            # Insecure SSL/TLS
            'insecure_ssl': [
                r'ssl_verify\s*=\s*False',
                r'verify\s*=\s*False',
                r'CERT_NONE',
            ],
            
            # Debug mode in production
            'debug_mode': [
                r'debug\s*=\s*True',
                r'DEBUG\s*=\s*True',
            ],
        }
    
    def scan_file(self, file_path: Path) -> List[SecurityFinding]:
        """Scan a single file for security issues"""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check patterns
            for category, patterns in self.security_patterns.items():
                for pattern in patterns:
                    matches = list(re.finditer(pattern, content, re.IGNORECASE))
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        finding = self._create_finding(
                            category, pattern, file_path, line_num, match.group()
                        )
                        findings.append(finding)
            
            # AST-based analysis for Python files
            if file_path.suffix == '.py':
                findings.extend(self._analyze_python_ast(file_path, content))
            
        except Exception as e:
            self.logger.error(f"Error scanning file {file_path}: {e}")
        
        return findings
    
    def _analyze_python_ast(self, file_path: Path, content: str) -> List[SecurityFinding]:
        """Analyze Python file using AST for deeper security analysis"""
        findings = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'compile']:
                            finding = SecurityFinding(
                                finding_id=f"ast_{hashlib.md5(str(node).encode()).hexdigest()[:8]}",
                                title="Dangerous Function Call",
                                description=f"Use of {node.func.id} function detected",
                                level=VulnerabilityLevel.HIGH,
                                file_path=str(file_path),
                                line_number=node.lineno,
                                code_snippet=ast.get_source_segment(content, node),
                                recommendation=f"Avoid using {node.func.id}. Use safer alternatives.",
                                cwe_id="CWE-94"
                            )
                            findings.append(finding)
                
                # Check for insecure imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ['pickle', 'cPickle']:
                            finding = SecurityFinding(
                                finding_id=f"ast_{hashlib.md5(str(node).encode()).hexdigest()[:8]}",
                                title="Insecure Import",
                                description=f"Import of {alias.name} module detected",
                                level=VulnerabilityLevel.MEDIUM,
                                file_path=str(file_path),
                                line_number=node.lineno,
                                code_snippet=f"import {alias.name}",
                                recommendation="Use json or other secure serialization methods instead of pickle",
                                cwe_id="CWE-502"
                            )
                            findings.append(finding)
        
        except SyntaxError:
            # File has syntax errors, skip AST analysis
            pass
        except Exception as e:
            self.logger.error(f"AST analysis error for {file_path}: {e}")
        
        return findings
    
    def _create_finding(self, category: str, pattern: str, file_path: Path, 
                       line_num: int, code_snippet: str) -> SecurityFinding:
        """Create security finding from pattern match"""
        
        finding_templates = {
            'sql_injection': {
                'title': 'SQL Injection Vulnerability',
                'description': 'Potential SQL injection vulnerability detected',
                'level': VulnerabilityLevel.HIGH,
                'recommendation': 'Use parameterized queries or ORM to prevent SQL injection',
                'cwe_id': 'CWE-89'
            },
            'command_injection': {
                'title': 'Command Injection Vulnerability', 
                'description': 'Potential command injection vulnerability detected',
                'level': VulnerabilityLevel.HIGH,
                'recommendation': 'Use subprocess with shell=False and validate inputs',
                'cwe_id': 'CWE-78'
            },
            'hardcoded_secrets': {
                'title': 'Hardcoded Secrets',
                'description': 'Hardcoded password or secret detected',
                'level': VulnerabilityLevel.CRITICAL,
                'recommendation': 'Use environment variables or secure secret management',
                'cwe_id': 'CWE-798'
            },
            'weak_random': {
                'title': 'Weak Random Number Generation',
                'description': 'Use of cryptographically weak random number generator',
                'level': VulnerabilityLevel.MEDIUM,
                'recommendation': 'Use secrets module for cryptographic purposes',
                'cwe_id': 'CWE-338'
            },
            'insecure_ssl': {
                'title': 'Insecure SSL/TLS Configuration',
                'description': 'SSL/TLS certificate verification disabled',
                'level': VulnerabilityLevel.HIGH,
                'recommendation': 'Enable SSL certificate verification',
                'cwe_id': 'CWE-295'
            },
            'debug_mode': {
                'title': 'Debug Mode Enabled',
                'description': 'Debug mode enabled in production code',
                'level': VulnerabilityLevel.MEDIUM,
                'recommendation': 'Disable debug mode in production',
                'cwe_id': 'CWE-489'
            }
        }
        
        template = finding_templates.get(category, {
            'title': 'Security Issue',
            'description': 'Potential security issue detected',
            'level': VulnerabilityLevel.MEDIUM,
            'recommendation': 'Review code for security implications',
            'cwe_id': None
        })
        
        return SecurityFinding(
            finding_id=hashlib.md5(f"{file_path}{line_num}{pattern}".encode()).hexdigest()[:12],
            title=template['title'],
            description=template['description'],
            level=template['level'],
            file_path=str(file_path),
            line_number=line_num,
            code_snippet=code_snippet.strip()[:200],
            recommendation=template['recommendation'],
            cwe_id=template['cwe_id']
        )


class DependencyScanner:
    """Scanner for vulnerable dependencies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def scan_requirements(self, requirements_file: Path) -> List[SecurityFinding]:
        """Scan requirements file for known vulnerabilities using safety"""
        findings = []
        
        try:
            # Use safety to check for known vulnerabilities
            result = subprocess.run(
                ['safety', 'check', '-r', str(requirements_file), '--json'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0 and result.stdout:
                vulnerabilities = json.loads(result.stdout)
                
                for vuln in vulnerabilities:
                    finding = SecurityFinding(
                        finding_id=f"dep_{vuln.get('id', 'unknown')}",
                        title=f"Vulnerable Dependency: {vuln.get('package', 'unknown')}",
                        description=vuln.get('advisory', 'Known vulnerability in dependency'),
                        level=self._map_safety_severity(vuln.get('severity', 'medium')),
                        file_path=str(requirements_file),
                        line_number=None,
                        code_snippet=f"{vuln.get('package', 'unknown')}=={vuln.get('installed_version', 'unknown')}",
                        recommendation=f"Update to version {vuln.get('safe_version', 'latest')} or higher",
                        cve_id=vuln.get('cve')
                    )
                    findings.append(finding)
                    
        except FileNotFoundError:
            self.logger.warning("Safety tool not found. Install with: pip install safety")
        except subprocess.TimeoutExpired:
            self.logger.error("Safety check timed out")
        except Exception as e:
            self.logger.error(f"Error running safety check: {e}")
        
        return findings
    
    def _map_safety_severity(self, severity: str) -> VulnerabilityLevel:
        """Map safety severity to our vulnerability levels"""
        mapping = {
            'low': VulnerabilityLevel.LOW,
            'medium': VulnerabilityLevel.MEDIUM,
            'high': VulnerabilityLevel.HIGH,
            'critical': VulnerabilityLevel.CRITICAL
        }
        return mapping.get(severity.lower(), VulnerabilityLevel.MEDIUM)


class SecurityScanner:
    """Main security scanner orchestrating all security checks"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        
        self.code_scanner = CodeSecurityScanner()
        self.dependency_scanner = DependencyScanner()
        
        # Files and directories to scan
        self.scan_patterns = ['*.py', '*.yaml', '*.yml', '*.json', '*.sh']
        self.exclude_patterns = ['.git', '__pycache__', '.venv', 'node_modules', '.pytest_cache']
    
    def run_full_scan(self) -> SecurityScanResult:
        """Run complete security scan"""
        start_time = datetime.now()
        scan_id = f"scan_{int(start_time.timestamp())}"
        
        self.logger.info(f"Starting security scan {scan_id}")
        
        all_findings = []
        files_scanned = 0
        
        # Scan source code files
        for pattern in self.scan_patterns:
            for file_path in self.project_root.rglob(pattern):
                if self._should_scan_file(file_path):
                    findings = self.code_scanner.scan_file(file_path)
                    all_findings.extend(findings)
                    files_scanned += 1
        
        # Scan dependencies
        requirements_files = list(self.project_root.glob('requirements*.txt'))
        for req_file in requirements_files:
            findings = self.dependency_scanner.scan_requirements(req_file)
            all_findings.extend(findings)
        
        scan_duration = (datetime.now() - start_time).total_seconds()
        
        result = SecurityScanResult(
            scan_id=scan_id,
            timestamp=start_time,
            findings=all_findings,
            scan_duration=scan_duration,
            files_scanned=files_scanned
        )
        
        self.logger.info(f"Security scan {scan_id} completed: {len(all_findings)} findings in {scan_duration:.2f}s")
        return result
    
    def _should_scan_file(self, file_path: Path) -> bool:
        """Check if file should be scanned"""
        for exclude in self.exclude_patterns:
            if exclude in file_path.parts:
                return False
        
        # Skip very large files
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
                return False
        except OSError:
            return False
        
        return True
    
    def generate_report(self, scan_result: SecurityScanResult, output_file: Path):
        """Generate security scan report"""
        report = {
            'scan_id': scan_result.scan_id,
            'timestamp': scan_result.timestamp.isoformat(),
            'summary': scan_result.get_summary(),
            'scan_duration': scan_result.scan_duration,
            'files_scanned': scan_result.files_scanned,
            'findings': [
                {
                    'id': f.finding_id,
                    'title': f.title,
                    'description': f.description,
                    'level': f.level.value,
                    'file': f.file_path,
                    'line': f.line_number,
                    'code': f.code_snippet,
                    'recommendation': f.recommendation,
                    'cwe_id': f.cwe_id,
                    'cve_id': f.cve_id
                }
                for f in scan_result.findings
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Security report written to {output_file}")


def run_security_scan(project_root: str, output_file: str = None) -> SecurityScanResult:
    """Run security scan and optionally generate report"""
    scanner = SecurityScanner(Path(project_root))
    result = scanner.run_full_scan()
    
    if output_file:
        scanner.generate_report(result, Path(output_file))
    
    return result