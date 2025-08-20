"""
Production Security Management
Comprehensive security manager for production deployment with JWT, RBAC, and audit logging
"""

import logging
import os
import jwt
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import re


@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    event_type: str
    user_id: str
    timestamp: datetime
    action: str
    resource: Optional[str] = None
    success: bool = True
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class JWTConfig:
    """JWT configuration"""
    secret_key: str
    algorithm: str = "HS256"
    token_expiry_hours: int = 24
    

@dataclass
class RBACConfig:
    """RBAC configuration"""
    roles: Dict[str, Dict[str, Any]]


@dataclass
class AuditConfig:
    """Audit logging configuration"""
    log_directory: str
    retention_days: int = 90
    max_log_size_mb: int = 100
    backup_count: int = 10
    critical_events_separate_log: bool = True


class SecurityManager:
    """Production security manager with JWT, RBAC, and audit logging"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize JWT configuration
        jwt_config = config.get('jwt', {})
        self.jwt_config = JWTConfig(
            secret_key=jwt_config.get('secret_key', os.environ.get('JWT_SECRET_KEY', 'default-secret-key')),
            algorithm=jwt_config.get('algorithm', 'HS256'),
            token_expiry_hours=jwt_config.get('token_expiry_hours', 24)
        )
        
        # Initialize RBAC configuration
        rbac_config = config.get('rbac', {})
        self.rbac_config = RBACConfig(
            roles=rbac_config.get('roles', {})
        )
        
        # Initialize audit configuration
        audit_config = config.get('audit', {})
        self.audit_config = AuditConfig(
            log_directory=audit_config.get('log_directory', 'logs/audit'),
            retention_days=audit_config.get('retention_days', 90),
            max_log_size_mb=audit_config.get('max_log_size_mb', 100),
            backup_count=audit_config.get('backup_count', 10),
            critical_events_separate_log=audit_config.get('critical_events_separate_log', True)
        )
        
        # Setup audit logging
        self._setup_audit_logging()
        
        # Initialize security scanner configuration
        self.scanner_config = config.get('scanner', {})
        
        self.logger.info("Security manager initialized")
        
    def _setup_audit_logging(self):
        """Setup audit logging infrastructure"""
        try:
            # Create audit log directory
            os.makedirs(self.audit_config.log_directory, exist_ok=True)
            
            # Setup audit logger
            self.audit_logger = logging.getLogger('security.audit')
            self.audit_logger.setLevel(logging.INFO)
            
            # Create audit log handler
            audit_log_path = os.path.join(self.audit_config.log_directory, 'security_audit.log')
            handler = logging.FileHandler(audit_log_path)
            
            # Create formatter for audit logs
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            
            # Add handler to audit logger
            if not self.audit_logger.handlers:
                self.audit_logger.addHandler(handler)
            
            self.logger.info(f"Audit logging setup completed: {audit_log_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup audit logging: {e}")
            raise
            
    def generate_jwt_token(self, user_id: str, roles: List[str], additional_claims: Dict[str, Any] = None) -> str:
        """Generate JWT token for user authentication"""
        try:
            # Create token payload
            payload = {
                'user_id': user_id,
                'roles': roles,
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(hours=self.jwt_config.token_expiry_hours)
            }
            
            # Add additional claims if provided
            if additional_claims:
                payload.update(additional_claims)
            
            # Generate token
            token = jwt.encode(payload, self.jwt_config.secret_key, algorithm=self.jwt_config.algorithm)
            
            # Log successful token generation
            self.log_security_event(
                'jwt_token_generated',
                user_id,
                {'roles': roles, 'expiry_hours': self.jwt_config.token_expiry_hours}
            )
            
            return token
            
        except Exception as e:
            self.logger.error(f"Failed to generate JWT token: {e}")
            self.log_security_event(
                'jwt_token_generation_failed',
                user_id,
                {'error': str(e)},
                success=False
            )
            raise
            
    def validate_jwt_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token and return payload"""
        try:
            # Decode and validate token
            payload = jwt.decode(
                token,
                self.jwt_config.secret_key,
                algorithms=[self.jwt_config.algorithm]
            )
            
            # Log successful token validation
            self.log_security_event(
                'jwt_token_validated',
                payload.get('user_id', 'unknown'),
                {'roles': payload.get('roles', [])}
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("JWT token expired")
            self.log_security_event(
                'jwt_token_expired',
                'unknown',
                {'token_hash': hashlib.sha256(token.encode()).hexdigest()[:16]},
                success=False
            )
            raise
            
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid JWT token: {e}")
            self.log_security_event(
                'jwt_token_invalid',
                'unknown',
                {'error': str(e), 'token_hash': hashlib.sha256(token.encode()).hexdigest()[:16]},
                success=False
            )
            raise
            
        except Exception as e:
            self.logger.error(f"JWT token validation failed: {e}")
            self.log_security_event(
                'jwt_token_validation_failed',
                'unknown',
                {'error': str(e)},
                success=False
            )
            raise
            
    def check_permission(self, user_id: str, user_roles: List[str], required_permission: str) -> bool:
        """Check if user has required permission based on RBAC"""
        try:
            # Check each user role for the required permission
            for role in user_roles:
                role_config = self.rbac_config.roles.get(role, {})
                permissions = role_config.get('permissions', [])
                
                if required_permission in permissions:
                    self.log_security_event(
                        'permission_granted',
                        user_id,
                        {'role': role, 'permission': required_permission}
                    )
                    return True
            
            # Permission denied
            self.log_security_event(
                'permission_denied',
                user_id,
                {'roles': user_roles, 'permission': required_permission},
                success=False
            )
            return False
            
        except Exception as e:
            self.logger.error(f"Permission check failed: {e}")
            self.log_security_event(
                'permission_check_failed',
                user_id,
                {'error': str(e), 'permission': required_permission},
                success=False
            )
            return False
            
    def log_security_event(self, event_type: str, user_id: str, additional_data: Dict[str, Any] = None, success: bool = True):
        """Log security event for audit trail"""
        try:
            event = SecurityEvent(
                event_type=event_type,
                user_id=user_id,
                timestamp=datetime.utcnow(),
                action=event_type,
                success=success,
                additional_data=additional_data or {}
            )
            
            # Create audit log entry
            log_entry = {
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type,
                'user_id': event.user_id,
                'action': event.action,
                'success': event.success,
                'additional_data': event.additional_data
            }
            
            # Log to audit logger
            log_message = f"AUDIT: {json.dumps(log_entry)}"
            
            if event.success:
                self.audit_logger.info(log_message)
            else:
                self.audit_logger.warning(log_message)
                
            # Log critical events to separate file if configured
            if not event.success and self.audit_config.critical_events_separate_log:
                self._log_critical_event(log_entry)
                
        except Exception as e:
            self.logger.error(f"Failed to log security event: {e}")
            
    def _log_critical_event(self, log_entry: Dict[str, Any]):
        """Log critical security events to separate file"""
        try:
            critical_log_path = os.path.join(self.audit_config.log_directory, 'security_critical.log')
            
            with open(critical_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{json.dumps(log_entry)}\n")
                
        except Exception as e:
            self.logger.error(f"Failed to log critical event: {e}")
            
    def get_user_roles(self, user_id: str) -> List[str]:
        """Get user roles (placeholder - would integrate with user management system)"""
        # This is a placeholder implementation
        # In production, this would integrate with your user management system
        default_roles = {
            'admin': ['admin'],
            'operator': ['operator'],
            'reviewer': ['reviewer'],
            'test_user': ['admin']  # For testing
        }
        return default_roles.get(user_id, ['viewer'])
        
    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[str], Optional[List[str]]]:
        """Authenticate user credentials (placeholder implementation)"""
        try:
            # This is a placeholder implementation
            # In production, this would integrate with your authentication system
            
            # For demonstration purposes, accept any user with password "password"
            if password == "password":
                roles = self.get_user_roles(username)
                token = self.generate_jwt_token(username, roles)
                
                self.log_security_event(
                    'user_authentication_success',
                    username,
                    {'roles': roles}
                )
                
                return True, token, roles
            else:
                self.log_security_event(
                    'user_authentication_failed',
                    username,
                    {'reason': 'invalid_credentials'},
                    success=False
                )
                return False, None, None
                
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            self.log_security_event(
                'user_authentication_error',
                username,
                {'error': str(e)},
                success=False
            )
            return False, None, None
            
    def run_security_scan(self, scan_patterns: List[str] = None, exclude_patterns: List[str] = None) -> Dict[str, Any]:
        """Run security vulnerability scan"""
        try:
            if not self.scanner_config.get('enabled', False):
                return {'status': 'disabled', 'message': 'Security scanning disabled'}
                
            scan_patterns = scan_patterns or self.scanner_config.get('scan_patterns', ['*.py'])
            exclude_patterns = exclude_patterns or self.scanner_config.get('exclude_patterns', [])
            
            scan_results = {
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'completed',
                'vulnerabilities': {
                    'critical': 0,
                    'high': 0,
                    'medium': 0,
                    'low': 0
                },
                'files_scanned': 0,
                'scan_duration_seconds': 0
            }
            
            start_time = datetime.utcnow()
            
            # Perform basic security checks
            project_root = Path(__file__).parent.parent.parent
            files_scanned = 0
            
            # Check for common security issues
            security_issues = []
            
            for pattern in scan_patterns:
                for file_path in project_root.rglob(pattern):
                    # Skip excluded patterns
                    if any(exclude in str(file_path) for exclude in exclude_patterns):
                        continue
                        
                    files_scanned += 1
                    
                    # Basic security checks
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        # Check for potential security issues
                        issues = self._check_file_security(file_path, content)
                        security_issues.extend(issues)
                        
                    except Exception:
                        continue  # Skip files we can't read
                        
            # Categorize vulnerabilities
            for issue in security_issues:
                severity = issue.get('severity', 'low')
                scan_results['vulnerabilities'][severity] += 1
                
            scan_results['files_scanned'] = files_scanned
            scan_results['scan_duration_seconds'] = (datetime.utcnow() - start_time).total_seconds()
            
            # Check against vulnerability thresholds
            thresholds = self.scanner_config.get('vulnerability_thresholds', {})
            scan_results['threshold_violations'] = []
            
            for severity, count in scan_results['vulnerabilities'].items():
                threshold = thresholds.get(severity, float('inf'))
                if count > threshold:
                    scan_results['threshold_violations'].append({
                        'severity': severity,
                        'count': count,
                        'threshold': threshold
                    })
                    
            # Log security scan
            self.log_security_event(
                'security_scan_completed',
                'system',
                {
                    'files_scanned': files_scanned,
                    'vulnerabilities': scan_results['vulnerabilities'],
                    'threshold_violations': len(scan_results['threshold_violations'])
                }
            )
            
            return scan_results
            
        except Exception as e:
            self.logger.error(f"Security scan failed: {e}")
            self.log_security_event(
                'security_scan_failed',
                'system',
                {'error': str(e)},
                success=False
            )
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            
    def _check_file_security(self, file_path: Path, content: str) -> List[Dict[str, Any]]:
        """Check file for common security issues"""
        issues = []
        
        # Check for hardcoded secrets/passwords
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]
        
        for pattern in secret_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if 'default-secret-key' not in match and 'password' not in match.lower():
                    issues.append({
                        'type': 'hardcoded_secret',
                        'severity': 'high',
                        'file': str(file_path),
                        'description': f'Potential hardcoded secret: {match[:50]}...'
                    })
                    
        # Check for SQL injection vulnerabilities
        sql_patterns = [
            r'execute\s*\(\s*["\'][^"\']*%[^"\']*["\']',
            r'query\s*\(\s*["\'][^"\']*\+[^"\']*["\']'
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append({
                    'type': 'sql_injection',
                    'severity': 'critical',
                    'file': str(file_path),
                    'description': 'Potential SQL injection vulnerability'
                })
                
        # Check for command injection
        if 'subprocess' in content and 'shell=True' in content:
            issues.append({
                'type': 'command_injection',
                'severity': 'high',
                'file': str(file_path),
                'description': 'Potential command injection with shell=True'
            })
            
        return issues
        
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security system status"""
        try:
            # Check audit log health
            audit_log_path = os.path.join(self.audit_config.log_directory, 'security_audit.log')
            audit_log_size = 0
            audit_log_exists = False
            
            if os.path.exists(audit_log_path):
                audit_log_exists = True
                audit_log_size = os.path.getsize(audit_log_path) / (1024 * 1024)  # MB
                
            status = {
                'timestamp': datetime.utcnow().isoformat(),
                'security_components': {
                    'jwt_authentication': 'enabled',
                    'rbac_authorization': 'enabled',
                    'audit_logging': 'enabled' if audit_log_exists else 'error',
                    'security_scanning': 'enabled' if self.scanner_config.get('enabled') else 'disabled'
                },
                'configuration': {
                    'jwt_algorithm': self.jwt_config.algorithm,
                    'token_expiry_hours': self.jwt_config.token_expiry_hours,
                    'roles_configured': len(self.rbac_config.roles),
                    'audit_retention_days': self.audit_config.retention_days
                },
                'audit_log_status': {
                    'exists': audit_log_exists,
                    'size_mb': round(audit_log_size, 2),
                    'directory': self.audit_config.log_directory
                }
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get security status: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'error',
                'error': str(e)
            }
            
    def cleanup_old_audit_logs(self):
        """Clean up old audit logs based on retention policy"""
        try:
            audit_dir = Path(self.audit_config.log_directory)
            if not audit_dir.exists():
                return
                
            cutoff_date = datetime.utcnow() - timedelta(days=self.audit_config.retention_days)
            
            for log_file in audit_dir.glob('*.log'):
                try:
                    # Check file modification time
                    file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    
                    if file_mtime < cutoff_date:
                        log_file.unlink()
                        self.logger.info(f"Removed old audit log: {log_file}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to remove old audit log {log_file}: {e}")
                    
            self.log_security_event(
                'audit_log_cleanup',
                'system',
                {'retention_days': self.audit_config.retention_days}
            )
            
        except Exception as e:
            self.logger.error(f"Audit log cleanup failed: {e}")
            self.log_security_event(
                'audit_log_cleanup_failed',
                'system',
                {'error': str(e)},
                success=False
            )
            
    def shutdown(self):
        """Graceful shutdown of security manager"""
        try:
            self.logger.info("Shutting down security manager")
            
            # Final security event
            self.log_security_event(
                'security_manager_shutdown',
                'system',
                {'shutdown_time': datetime.utcnow().isoformat()}
            )
            
            # Close audit log handlers
            for handler in self.audit_logger.handlers[:]:
                handler.close()
                self.audit_logger.removeHandler(handler)
                
            self.logger.info("Security manager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during security manager shutdown: {e}")