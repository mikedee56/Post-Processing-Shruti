"""
Epic 4 - Story 4.2: Version Control & Documentation
Version management system with semantic versioning for the Sanskrit processing system

This module provides:
- Semantic version management (MAJOR.MINOR.PATCH)
- Automated version bumping with Git integration
- Version-aware configuration management
- Compatibility checking for lexicons and models
- Release history tracking
"""

import os
import re
import json
import logging
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import semver
import hashlib


@dataclass
class Version:
    """Represents a semantic version."""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation of version."""
        version_str = f"{self.major}.{self.minor}.{self.patch}"
        
        if self.prerelease:
            version_str += f"-{self.prerelease}"
        
        if self.build:
            version_str += f"+{self.build}"
        
        return version_str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_string(cls, version_str: str) -> 'Version':
        """Parse version from string."""
        try:
            parsed = semver.VersionInfo.parse(version_str)
            return cls(
                major=parsed.major,
                minor=parsed.minor,
                patch=parsed.patch,
                prerelease=parsed.prerelease,
                build=parsed.build
            )
        except Exception as e:
            raise ValueError(f"Invalid version string '{version_str}': {e}")
    
    def compare(self, other: 'Version') -> int:
        """
        Compare with another version.
        
        Returns:
            -1 if self < other, 0 if self == other, 1 if self > other
        """
        self_ver = semver.VersionInfo(
            self.major, self.minor, self.patch, 
            self.prerelease, self.build
        )
        other_ver = semver.VersionInfo(
            other.major, other.minor, other.patch,
            other.prerelease, other.build
        )
        
        return self_ver.compare(other_ver)
    
    def is_compatible_with(self, other: 'Version') -> bool:
        """
        Check if this version is compatible with another.
        Compatible if major versions match and this version >= other.
        """
        if self.major != other.major:
            return False
        
        return self.compare(other) >= 0


@dataclass
class ReleaseInfo:
    """Information about a software release."""
    version: Version
    timestamp: datetime
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_tag: Optional[str] = None
    changelog: Optional[str] = None
    breaking_changes: List[str] = None
    features: List[str] = None
    bug_fixes: List[str] = None
    dependencies: Dict[str, str] = None
    
    def __post_init__(self):
        if self.breaking_changes is None:
            self.breaking_changes = []
        if self.features is None:
            self.features = []
        if self.bug_fixes is None:
            self.bug_fixes = []
        if self.dependencies is None:
            self.dependencies = {}


@dataclass  
class ComponentVersion:
    """Version information for a specific component."""
    name: str
    version: Version
    checksum: str
    last_updated: datetime
    compatibility_requirements: Dict[str, str] = None
    
    def __post_init__(self):
        if self.compatibility_requirements is None:
            self.compatibility_requirements = {}


class VersionManager:
    """
    Manages semantic versioning for the Sanskrit processing system.
    
    Features:
    - Semantic version parsing and comparison
    - Automated version bumping
    - Git integration for tagging and history
    - Component version tracking (lexicons, models)
    - Compatibility validation
    - Release history management
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration files
        self.version_file = self.project_root / "version.json"
        self.release_history_file = self.project_root / "release_history.json"
        self.component_versions_file = self.project_root / "component_versions.json"
        
        # Current version
        self._current_version: Optional[Version] = None
        self._release_history: List[ReleaseInfo] = []
        self._component_versions: Dict[str, ComponentVersion] = {}
        
        # Load existing data
        self._load_version_data()
    
    def _load_version_data(self):
        """Load version data from files."""
        # Load current version
        if self.version_file.exists():
            try:
                with open(self.version_file, 'r') as f:
                    version_data = json.load(f)
                self._current_version = Version(**version_data)
            except Exception as e:
                self.logger.warning(f"Could not load version file: {e}")
        
        # Load release history
        if self.release_history_file.exists():
            try:
                with open(self.release_history_file, 'r') as f:
                    history_data = json.load(f)
                
                self._release_history = []
                for item in history_data:
                    release = ReleaseInfo(
                        version=Version(**item['version']),
                        timestamp=datetime.fromisoformat(item['timestamp']),
                        **{k: v for k, v in item.items() if k not in ['version', 'timestamp']}
                    )
                    self._release_history.append(release)
                    
            except Exception as e:
                self.logger.warning(f"Could not load release history: {e}")
        
        # Load component versions
        if self.component_versions_file.exists():
            try:
                with open(self.component_versions_file, 'r') as f:
                    components_data = json.load(f)
                
                self._component_versions = {}
                for name, data in components_data.items():
                    component = ComponentVersion(
                        name=name,
                        version=Version(**data['version']),
                        checksum=data['checksum'],
                        last_updated=datetime.fromisoformat(data['last_updated']),
                        compatibility_requirements=data.get('compatibility_requirements', {})
                    )
                    self._component_versions[name] = component
                    
            except Exception as e:
                self.logger.warning(f"Could not load component versions: {e}")
    
    def _save_version_data(self):
        """Save version data to files."""
        try:
            # Save current version
            if self._current_version:
                with open(self.version_file, 'w') as f:
                    json.dump(self._current_version.to_dict(), f, indent=2)
            
            # Save release history
            history_data = []
            for release in self._release_history:
                release_dict = asdict(release)
                release_dict['version'] = release.version.to_dict()
                release_dict['timestamp'] = release.timestamp.isoformat()
                history_data.append(release_dict)
            
            with open(self.release_history_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
            
            # Save component versions
            components_data = {}
            for name, component in self._component_versions.items():
                components_data[name] = {
                    'version': component.version.to_dict(),
                    'checksum': component.checksum,
                    'last_updated': component.last_updated.isoformat(),
                    'compatibility_requirements': component.compatibility_requirements
                }
            
            with open(self.component_versions_file, 'w') as f:
                json.dump(components_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save version data: {e}")
    
    def get_current_version(self) -> Version:
        """Get the current project version."""
        if self._current_version is None:
            # Initialize with default version
            self._current_version = Version(0, 1, 0)
            self._save_version_data()
        
        return self._current_version
    
    def set_version(self, version: Version, release_info: ReleaseInfo = None):
        """
        Set the current version.
        
        Args:
            version: New version to set
            release_info: Optional release information
        """
        old_version = self._current_version
        self._current_version = version
        
        # Add to release history
        if release_info:
            self._release_history.append(release_info)
        else:
            release = ReleaseInfo(
                version=version,
                timestamp=datetime.utcnow(),
                git_commit=self._get_git_commit(),
                git_branch=self._get_git_branch()
            )
            self._release_history.append(release)
        
        self._save_version_data()
        
        self.logger.info(f"Version updated from {old_version} to {version}")
    
    def bump_version(
        self, 
        bump_type: str, 
        prerelease: str = None,
        create_git_tag: bool = True
    ) -> Version:
        """
        Bump the version according to semantic versioning rules.
        
        Args:
            bump_type: 'major', 'minor', 'patch', or 'prerelease'
            prerelease: Optional prerelease identifier
            create_git_tag: Whether to create a Git tag
            
        Returns:
            New version
        """
        current = self.get_current_version()
        
        if bump_type == 'major':
            new_version = Version(current.major + 1, 0, 0, prerelease)
        elif bump_type == 'minor':
            new_version = Version(current.major, current.minor + 1, 0, prerelease)
        elif bump_type == 'patch':
            new_version = Version(current.major, current.minor, current.patch + 1, prerelease)
        elif bump_type == 'prerelease':
            if current.prerelease:
                # Increment prerelease number
                match = re.match(r'(.+?)(\d+)$', current.prerelease)
                if match:
                    try:
                        prefix, number = match.groups()
                    except AttributeError:
                        prefix, number = "", "1"
                    new_prerelease = f"{prefix}{int(number) + 1}"
                else:
                    new_prerelease = f"{current.prerelease}.1"
            else:
                new_prerelease = prerelease or "rc.1"
            
            new_version = Version(
                current.major, current.minor, current.patch, 
                new_prerelease
            )
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")
        
        # Create release info
        release_info = ReleaseInfo(
            version=new_version,
            timestamp=datetime.utcnow(),
            git_commit=self._get_git_commit(),
            git_branch=self._get_git_branch(),
            changelog=f"Version bumped from {current} to {new_version}"
        )
        
        # Set new version
        self.set_version(new_version, release_info)
        
        # Create Git tag if requested
        if create_git_tag:
            self._create_git_tag(str(new_version))
        
        return new_version
    
    def register_component(
        self,
        name: str,
        file_path: str,
        version: Version = None,
        compatibility_requirements: Dict[str, str] = None
    ) -> ComponentVersion:
        """
        Register a component (lexicon, model, etc.) with version tracking.
        
        Args:
            name: Component name
            file_path: Path to the component file
            version: Component version (auto-detected if not provided)
            compatibility_requirements: Version requirements for compatibility
            
        Returns:
            ComponentVersion object
        """
        # Calculate file checksum
        checksum = self._calculate_file_checksum(file_path)
        
        # Auto-detect version if not provided
        if version is None:
            version = self._detect_component_version(name, checksum)
        
        component = ComponentVersion(
            name=name,
            version=version,
            checksum=checksum,
            last_updated=datetime.utcnow(),
            compatibility_requirements=compatibility_requirements or {}
        )
        
        self._component_versions[name] = component
        self._save_version_data()
        
        self.logger.info(f"Registered component {name} version {version}")
        return component
    
    def get_component_version(self, name: str) -> Optional[ComponentVersion]:
        """Get version information for a component."""
        return self._component_versions.get(name)
    
    def validate_component_compatibility(self, name: str) -> bool:
        """
        Validate that a component is compatible with the current system.
        
        Args:
            name: Component name
            
        Returns:
            True if compatible
        """
        component = self._component_versions.get(name)
        if not component:
            self.logger.warning(f"Component {name} not registered")
            return False
        
        current_version = self.get_current_version()
        
        # Check compatibility requirements
        for req_component, req_version_str in component.compatibility_requirements.items():
            if req_component == 'system':
                req_version = Version.from_string(req_version_str)
                if not current_version.is_compatible_with(req_version):
                    self.logger.error(
                        f"Component {name} requires system version {req_version_str}, "
                        f"but current version is {current_version}"
                    )
                    return False
            
            elif req_component in self._component_versions:
                req_version = Version.from_string(req_version_str)
                actual_component = self._component_versions[req_component]
                
                if not actual_component.version.is_compatible_with(req_version):
                    self.logger.error(
                        f"Component {name} requires {req_component} version {req_version_str}, "
                        f"but current version is {actual_component.version}"
                    )
                    return False
        
        return True
    
    def get_release_history(self, limit: int = None) -> List[ReleaseInfo]:
        """
        Get release history, optionally limited to recent releases.
        
        Args:
            limit: Maximum number of releases to return
            
        Returns:
            List of ReleaseInfo objects, most recent first
        """
        sorted_history = sorted(
            self._release_history,
            key=lambda r: r.timestamp,
            reverse=True
        )
        
        if limit:
            return sorted_history[:limit]
        
        return sorted_history
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of a file."""
        hasher = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        
        except Exception as e:
            self.logger.warning(f"Could not calculate checksum for {file_path}: {e}")
            return "unknown"
    
    def _detect_component_version(self, name: str, checksum: str) -> Version:
        """Auto-detect component version based on changes."""
        existing = self._component_versions.get(name)
        
        if existing is None:
            # First version
            return Version(1, 0, 0)
        
        if existing.checksum != checksum:
            # File changed, bump patch version
            return Version(
                existing.version.major,
                existing.version.minor,
                existing.version.patch + 1
            )
        
        # No change
        return existing.version
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current Git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception:
            return None
    
    def _get_git_branch(self) -> Optional[str]:
        """Get current Git branch name."""
        try:
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception:
            return None
    
    def _create_git_tag(self, tag_name: str):
        """Create a Git tag for the version."""
        try:
            subprocess.run(
                ['git', 'tag', '-a', f'v{tag_name}', '-m', f'Version {tag_name}'],
                cwd=self.project_root,
                check=True
            )
            self.logger.info(f"Created Git tag v{tag_name}")
        
        except Exception as e:
            self.logger.warning(f"Could not create Git tag: {e}")
    
    def generate_changelog(self, from_version: Version = None, to_version: Version = None) -> str:
        """
        Generate changelog between two versions.
        
        Args:
            from_version: Starting version (defaults to previous release)
            to_version: Ending version (defaults to current version)
            
        Returns:
            Formatted changelog string
        """
        if to_version is None:
            to_version = self.get_current_version()
        
        if from_version is None:
            # Get previous version from history
            history = self.get_release_history(limit=2)
            if len(history) >= 2:
                from_version = history[1].version
            else:
                from_version = Version(0, 0, 0)
        
        changelog_lines = [
            f"# Changelog from {from_version} to {to_version}",
            f"Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            ""
        ]
        
        # Find releases in range
        relevant_releases = [
            r for r in self._release_history
            if r.version.compare(from_version) > 0 and r.version.compare(to_version) <= 0
        ]
        
        relevant_releases.sort(key=lambda r: r.timestamp, reverse=True)
        
        for release in relevant_releases:
            changelog_lines.extend([
                f"## Version {release.version} ({release.timestamp.strftime('%Y-%m-%d')})",
                ""
            ])
            
            if release.breaking_changes:
                changelog_lines.append("### Breaking Changes")
                for change in release.breaking_changes:
                    changelog_lines.append(f"- {change}")
                changelog_lines.append("")
            
            if release.features:
                changelog_lines.append("### Features")
                for feature in release.features:
                    changelog_lines.append(f"- {feature}")
                changelog_lines.append("")
            
            if release.bug_fixes:
                changelog_lines.append("### Bug Fixes")
                for fix in release.bug_fixes:
                    changelog_lines.append(f"- {fix}")
                changelog_lines.append("")
            
            if release.changelog:
                changelog_lines.extend([
                    "### Additional Notes",
                    release.changelog,
                    ""
                ])
        
        return "\n".join(changelog_lines)
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get comprehensive version information."""
        current_version = self.get_current_version()
        
        return {
            'current_version': str(current_version),
            'version_details': current_version.to_dict(),
            'git_commit': self._get_git_commit(),
            'git_branch': self._get_git_branch(),
            'component_count': len(self._component_versions),
            'components': {
                name: str(comp.version) 
                for name, comp in self._component_versions.items()
            },
            'release_count': len(self._release_history),
            'latest_release': self._release_history[-1].timestamp.isoformat() if self._release_history else None
        }