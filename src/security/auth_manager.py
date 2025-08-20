"""
Authentication and Authorization Manager for Production Security
Implements JWT-based authentication with role-based access control (RBAC)
"""

import jwt
import bcrypt
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from functools import wraps
from flask import request, jsonify

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles for RBAC system"""
    ADMIN = "admin"
    OPERATOR = "operator"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


class Permission(Enum):
    """System permissions"""
    PROCESS_FILES = "process_files"
    VIEW_FILES = "view_files"
    DELETE_FILES = "delete_files"
    MANAGE_USERS = "manage_users"
    VIEW_METRICS = "view_metrics"
    MANAGE_CONFIG = "manage_config"


@dataclass
class User:
    """User data model"""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: List[Permission]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True


class AuthenticationManager:
    """JWT-based authentication manager"""
    
    def __init__(self, secret_key: str, token_expiry_hours: int = 24):
        self.secret_key = secret_key
        self.token_expiry_hours = token_expiry_hours
        self.algorithm = "HS256"
        
        # Role-permission mapping
        self.role_permissions = {
            UserRole.ADMIN: list(Permission),
            UserRole.OPERATOR: [
                Permission.PROCESS_FILES,
                Permission.VIEW_FILES,
                Permission.VIEW_METRICS
            ],
            UserRole.REVIEWER: [
                Permission.VIEW_FILES,
                Permission.VIEW_METRICS
            ],
            UserRole.VIEWER: [
                Permission.VIEW_FILES
            ]
        }
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def generate_token(self, user: User) -> str:
        """Generate JWT token for user"""
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'role': user.role.value,
            'permissions': [p.value for p in user.permissions],
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def create_user(self, username: str, email: str, password: str, role: UserRole) -> User:
        """Create new user with role-based permissions"""
        user_id = f"user_{datetime.utcnow().timestamp()}"
        permissions = self.role_permissions.get(role, [])
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            permissions=permissions,
            created_at=datetime.utcnow()
        )
        
        # In production, this would save to database
        logger.info(f"Created user {username} with role {role.value}")
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user credentials"""
        # In production, this would query database
        # For now, return a mock user for demonstration
        if username == "admin" and password == "admin123":
            return User(
                user_id="admin_user",
                username="admin",
                email="admin@example.com",
                role=UserRole.ADMIN,
                permissions=self.role_permissions[UserRole.ADMIN],
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow()
            )
        return None


class AuthorizationManager:
    """Role-based access control manager"""
    
    def __init__(self, auth_manager: AuthenticationManager):
        self.auth_manager = auth_manager
    
    def check_permission(self, user_permissions: List[str], required_permission: Permission) -> bool:
        """Check if user has required permission"""
        return required_permission.value in user_permissions
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Extract token from Authorization header
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    return jsonify({'error': 'Missing or invalid authorization header'}), 401
                
                token = auth_header.split(' ')[1]
                payload = self.auth_manager.verify_token(token)
                
                if not payload:
                    return jsonify({'error': 'Invalid or expired token'}), 401
                
                user_permissions = payload.get('permissions', [])
                if not self.check_permission(user_permissions, permission):
                    return jsonify({'error': 'Insufficient permissions'}), 403
                
                # Add user info to request context
                request.user_info = payload
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def require_role(self, required_role: UserRole):
        """Decorator to require specific role"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    return jsonify({'error': 'Missing or invalid authorization header'}), 401
                
                token = auth_header.split(' ')[1]
                payload = self.auth_manager.verify_token(token)
                
                if not payload:
                    return jsonify({'error': 'Invalid or expired token'}), 401
                
                user_role = payload.get('role')
                if user_role != required_role.value:
                    return jsonify({'error': f'Role {required_role.value} required'}), 403
                
                request.user_info = payload
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator


class SecurityService:
    """Main security service integrating authentication and authorization"""
    
    def __init__(self, secret_key: str, token_expiry_hours: int = 24):
        self.auth_manager = AuthenticationManager(secret_key, token_expiry_hours)
        self.authorization_manager = AuthorizationManager(self.auth_manager)
    
    def login(self, username: str, password: str) -> Optional[Tuple[str, User]]:
        """Login user and return token"""
        user = self.auth_manager.authenticate_user(username, password)
        if user:
            token = self.auth_manager.generate_token(user)
            logger.info(f"User {username} logged in successfully")
            return token, user
        
        logger.warning(f"Failed login attempt for username: {username}")
        return None
    
    def validate_request(self, token: str) -> Optional[Dict]:
        """Validate request token"""
        return self.auth_manager.verify_token(token)
    
    def get_user_permissions(self, token: str) -> List[str]:
        """Get user permissions from token"""
        payload = self.auth_manager.verify_token(token)
        if payload:
            return payload.get('permissions', [])
        return []
    
    def has_permission(self, token: str, permission: Permission) -> bool:
        """Check if user has specific permission"""
        user_permissions = self.get_user_permissions(token)
        return permission.value in user_permissions


# Global security service instance
security_service = None


def initialize_security(secret_key: str, token_expiry_hours: int = 24) -> SecurityService:
    """Initialize global security service"""
    global security_service
    security_service = SecurityService(secret_key, token_expiry_hours)
    logger.info("Security service initialized")
    return security_service


def get_security_service() -> SecurityService:
    """Get global security service instance"""
    if security_service is None:
        raise RuntimeError("Security service not initialized. Call initialize_security() first.")
    return security_service