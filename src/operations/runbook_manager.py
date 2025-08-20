"""
Runbook Management System
Operational runbooks, procedures, and knowledge base management
"""

import logging
import json
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class RunbookCategory(Enum):
    """Runbook categories"""
    DEPLOYMENT = "deployment"
    TROUBLESHOOTING = "troubleshooting"
    SCALING = "scaling"
    BACKUP_RECOVERY = "backup_recovery"
    SECURITY = "security"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"
    INCIDENT_RESPONSE = "incident_response"


class RunbookStatus(Enum):
    """Runbook status"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"


@dataclass
class RunbookStep:
    """Individual runbook step"""
    step_number: int
    title: str
    description: str
    command: Optional[str] = None
    expected_output: Optional[str] = None
    notes: Optional[str] = None
    estimated_time_minutes: Optional[int] = None


@dataclass
class Runbook:
    """Operational runbook"""
    id: str
    title: str
    description: str
    category: RunbookCategory
    status: RunbookStatus
    version: str
    created_at: datetime
    updated_at: datetime
    author: str
    reviewer: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    steps: List[RunbookStep] = field(default_factory=list)
    rollback_steps: List[RunbookStep] = field(default_factory=list)
    troubleshooting_tips: List[str] = field(default_factory=list)
    related_runbooks: List[str] = field(default_factory=list)
    external_links: Dict[str, str] = field(default_factory=dict)
    estimated_total_time_minutes: int = 30


@dataclass
class RunbookExecution:
    """Runbook execution record"""
    id: str
    runbook_id: str
    executor: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "in_progress"  # in_progress, completed, failed, cancelled
    completed_steps: List[int] = field(default_factory=list)
    notes: str = ""
    success: Optional[bool] = None


class RunbookManager:
    """Production runbook management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Runbook storage
        self.runbooks: Dict[str, Runbook] = {}
        self.executions: Dict[str, RunbookExecution] = {}
        
        # Configuration
        self.runbook_storage_dir = config.get('storage_dir', 'runbooks')
        self.wiki_base_url = config.get('wiki_base_url', '')
        
        # Load runbooks from configuration
        self._load_runbooks_from_config()
        
        # Ensure storage directory exists
        os.makedirs(self.runbook_storage_dir, exist_ok=True)
        
    def _load_runbooks_from_config(self):
        """Load runbook URLs and basic info from configuration"""
        runbook_urls = self.config
        
        # Create basic runbook entries from URLs
        for category, url in runbook_urls.items():
            if isinstance(url, str):
                runbook_id = f"wiki_{category}"
                
                # Map category names to enum values
                category_mapping = {
                    'deployment': RunbookCategory.DEPLOYMENT,
                    'troubleshooting': RunbookCategory.TROUBLESHOOTING,
                    'scaling': RunbookCategory.SCALING,
                    'backup_recovery': RunbookCategory.BACKUP_RECOVERY,
                    'database_migration': RunbookCategory.DEPLOYMENT,
                }
                
                runbook_category = category_mapping.get(category, RunbookCategory.DEPLOYMENT)
                
                runbook = Runbook(
                    id=runbook_id,
                    title=f"{category.replace('_', ' ').title()} Runbook",
                    description=f"External runbook for {category.replace('_', ' ')} procedures",
                    category=runbook_category,
                    status=RunbookStatus.ACTIVE,
                    version="1.0",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    author="operations",
                    external_links={"wiki": url}
                )
                
                self.runbooks[runbook_id] = runbook
                
    def create_runbook(self,
                      title: str,
                      description: str,
                      category: RunbookCategory,
                      author: str,
                      steps: List[Dict[str, Any]],
                      tags: Optional[List[str]] = None,
                      prerequisites: Optional[List[str]] = None) -> str:
        """Create new runbook"""
        
        runbook_id = f"rb_{int(datetime.utcnow().timestamp())}"
        
        # Convert step dictionaries to RunbookStep objects
        runbook_steps = []
        for i, step_data in enumerate(steps, 1):
            step = RunbookStep(
                step_number=i,
                title=step_data['title'],
                description=step_data['description'],
                command=step_data.get('command'),
                expected_output=step_data.get('expected_output'),
                notes=step_data.get('notes'),
                estimated_time_minutes=step_data.get('estimated_time_minutes', 5)
            )
            runbook_steps.append(step)
            
        # Calculate total estimated time
        total_time = sum(step.estimated_time_minutes or 5 for step in runbook_steps)
        
        runbook = Runbook(
            id=runbook_id,
            title=title,
            description=description,
            category=category,
            status=RunbookStatus.DRAFT,
            version="1.0",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            author=author,
            tags=tags or [],
            prerequisites=prerequisites or [],
            steps=runbook_steps,
            estimated_total_time_minutes=total_time
        )
        
        self.runbooks[runbook_id] = runbook
        
        # Save to storage
        self._save_runbook_to_file(runbook)
        
        self.logger.info(
            f"Runbook created: {runbook_id}",
            title=title,
            category=category.value,
            steps=len(runbook_steps)
        )
        
        return runbook_id
        
    def update_runbook(self,
                      runbook_id: str,
                      updates: Dict[str, Any]) -> bool:
        """Update existing runbook"""
        
        if runbook_id not in self.runbooks:
            return False
            
        runbook = self.runbooks[runbook_id]
        
        # Update allowed fields
        updatable_fields = [
            'title', 'description', 'tags', 'prerequisites',
            'troubleshooting_tips', 'related_runbooks', 'external_links'
        ]
        
        for field, value in updates.items():
            if field in updatable_fields:
                setattr(runbook, field, value)
                
        # Update steps if provided
        if 'steps' in updates:
            runbook_steps = []
            for i, step_data in enumerate(updates['steps'], 1):
                step = RunbookStep(
                    step_number=i,
                    title=step_data['title'],
                    description=step_data['description'],
                    command=step_data.get('command'),
                    expected_output=step_data.get('expected_output'),
                    notes=step_data.get('notes'),
                    estimated_time_minutes=step_data.get('estimated_time_minutes', 5)
                )
                runbook_steps.append(step)
                
            runbook.steps = runbook_steps
            runbook.estimated_total_time_minutes = sum(step.estimated_time_minutes or 5 for step in runbook_steps)
            
        # Increment version
        version_parts = runbook.version.split('.')
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        runbook.version = '.'.join(version_parts)
        
        runbook.updated_at = datetime.utcnow()
        
        # Save to storage
        self._save_runbook_to_file(runbook)
        
        self.logger.info(f"Runbook updated: {runbook_id}", version=runbook.version)
        return True
        
    def approve_runbook(self, runbook_id: str, reviewer: str) -> bool:
        """Approve runbook for active use"""
        
        if runbook_id not in self.runbooks:
            return False
            
        runbook = self.runbooks[runbook_id]
        runbook.status = RunbookStatus.ACTIVE
        runbook.reviewer = reviewer
        runbook.updated_at = datetime.utcnow()
        
        self._save_runbook_to_file(runbook)
        
        self.logger.info(f"Runbook approved: {runbook_id}", reviewer=reviewer)
        return True
        
    def deprecate_runbook(self, runbook_id: str, reason: str = "") -> bool:
        """Deprecate runbook"""
        
        if runbook_id not in self.runbooks:
            return False
            
        runbook = self.runbooks[runbook_id]
        runbook.status = RunbookStatus.DEPRECATED
        runbook.description += f"\n\nDEPRECATED: {reason}"
        runbook.updated_at = datetime.utcnow()
        
        self._save_runbook_to_file(runbook)
        
        self.logger.info(f"Runbook deprecated: {runbook_id}", reason=reason)
        return True
        
    def start_runbook_execution(self,
                               runbook_id: str,
                               executor: str) -> Optional[str]:
        """Start runbook execution"""
        
        if runbook_id not in self.runbooks:
            return None
            
        runbook = self.runbooks[runbook_id]
        if runbook.status != RunbookStatus.ACTIVE:
            return None
            
        execution_id = f"exec_{int(datetime.utcnow().timestamp())}"
        
        execution = RunbookExecution(
            id=execution_id,
            runbook_id=runbook_id,
            executor=executor,
            started_at=datetime.utcnow()
        )
        
        self.executions[execution_id] = execution
        
        self.logger.info(
            f"Runbook execution started: {execution_id}",
            runbook_id=runbook_id,
            executor=executor
        )
        
        return execution_id
        
    def complete_execution_step(self,
                               execution_id: str,
                               step_number: int,
                               success: bool = True,
                               notes: str = "") -> bool:
        """Complete a step in runbook execution"""
        
        if execution_id not in self.executions:
            return False
            
        execution = self.executions[execution_id]
        
        if success:
            execution.completed_steps.append(step_number)
            
        if notes:
            execution.notes += f"Step {step_number}: {notes}\n"
            
        # Check if all steps are completed
        runbook = self.runbooks[execution.runbook_id]
        if len(execution.completed_steps) >= len(runbook.steps):
            execution.status = "completed"
            execution.completed_at = datetime.utcnow()
            execution.success = True
            
        self.logger.info(
            f"Execution step completed: {execution_id}",
            step=step_number,
            success=success
        )
        
        return True
        
    def complete_runbook_execution(self,
                                  execution_id: str,
                                  success: bool = True,
                                  final_notes: str = "") -> bool:
        """Complete runbook execution"""
        
        if execution_id not in self.executions:
            return False
            
        execution = self.executions[execution_id]
        execution.status = "completed" if success else "failed"
        execution.completed_at = datetime.utcnow()
        execution.success = success
        
        if final_notes:
            execution.notes += f"Final notes: {final_notes}\n"
            
        self.logger.info(
            f"Runbook execution completed: {execution_id}",
            success=success,
            duration_minutes=(execution.completed_at - execution.started_at).total_seconds() / 60
        )
        
        return True
        
    def search_runbooks(self,
                       query: str,
                       category: Optional[RunbookCategory] = None,
                       status: Optional[RunbookStatus] = None) -> List[Runbook]:
        """Search runbooks by query, category, and status"""
        
        results = []
        query_lower = query.lower()
        
        for runbook in self.runbooks.values():
            # Apply filters
            if category and runbook.category != category:
                continue
            if status and runbook.status != status:
                continue
                
            # Search in title, description, and tags
            if (query_lower in runbook.title.lower() or
                query_lower in runbook.description.lower() or
                any(query_lower in tag.lower() for tag in runbook.tags)):
                results.append(runbook)
                
        return results
        
    def get_runbooks_by_category(self, category: RunbookCategory) -> List[Runbook]:
        """Get all runbooks in a category"""
        return [rb for rb in self.runbooks.values() if rb.category == category]
        
    def get_runbook_execution_history(self,
                                    runbook_id: str,
                                    limit: int = 10) -> List[RunbookExecution]:
        """Get execution history for a runbook"""
        
        executions = [
            ex for ex in self.executions.values()
            if ex.runbook_id == runbook_id
        ]
        
        # Sort by started_at (newest first)
        executions.sort(key=lambda x: x.started_at, reverse=True)
        return executions[:limit]
        
    def _save_runbook_to_file(self, runbook: Runbook):
        """Save runbook to file storage"""
        try:
            file_path = os.path.join(self.runbook_storage_dir, f"{runbook.id}.json")
            
            # Convert runbook to dictionary for JSON serialization
            runbook_data = {
                'id': runbook.id,
                'title': runbook.title,
                'description': runbook.description,
                'category': runbook.category.value,
                'status': runbook.status.value,
                'version': runbook.version,
                'created_at': runbook.created_at.isoformat(),
                'updated_at': runbook.updated_at.isoformat(),
                'author': runbook.author,
                'reviewer': runbook.reviewer,
                'tags': runbook.tags,
                'prerequisites': runbook.prerequisites,
                'steps': [
                    {
                        'step_number': step.step_number,
                        'title': step.title,
                        'description': step.description,
                        'command': step.command,
                        'expected_output': step.expected_output,
                        'notes': step.notes,
                        'estimated_time_minutes': step.estimated_time_minutes
                    }
                    for step in runbook.steps
                ],
                'rollback_steps': [
                    {
                        'step_number': step.step_number,
                        'title': step.title,
                        'description': step.description,
                        'command': step.command,
                        'expected_output': step.expected_output,
                        'notes': step.notes,
                        'estimated_time_minutes': step.estimated_time_minutes
                    }
                    for step in runbook.rollback_steps
                ],
                'troubleshooting_tips': runbook.troubleshooting_tips,
                'related_runbooks': runbook.related_runbooks,
                'external_links': runbook.external_links,
                'estimated_total_time_minutes': runbook.estimated_total_time_minutes
            }
            
            with open(file_path, 'w') as f:
                json.dump(runbook_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save runbook {runbook.id}", exception=e)
            
    def get_runbook_metrics(self) -> Dict[str, Any]:
        """Get runbook usage and effectiveness metrics"""
        
        total_runbooks = len(self.runbooks)
        active_runbooks = len([rb for rb in self.runbooks.values() if rb.status == RunbookStatus.ACTIVE])
        total_executions = len(self.executions)
        
        # Calculate success rate
        completed_executions = [ex for ex in self.executions.values() if ex.status == "completed"]
        successful_executions = [ex for ex in completed_executions if ex.success]
        
        success_rate = 0.0
        if completed_executions:
            success_rate = len(successful_executions) / len(completed_executions) * 100
            
        # Calculate average execution time
        avg_execution_time = 0.0
        if completed_executions:
            execution_times = []
            for ex in completed_executions:
                if ex.completed_at:
                    duration = (ex.completed_at - ex.started_at).total_seconds() / 60
                    execution_times.append(duration)
                    
            if execution_times:
                avg_execution_time = sum(execution_times) / len(execution_times)
                
        # Runbooks by category
        category_counts = {}
        for runbook in self.runbooks.values():
            category = runbook.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
            
        return {
            'total_runbooks': total_runbooks,
            'active_runbooks': active_runbooks,
            'deprecated_runbooks': len([rb for rb in self.runbooks.values() if rb.status == RunbookStatus.DEPRECATED]),
            'total_executions': total_executions,
            'success_rate_percent': success_rate,
            'average_execution_time_minutes': avg_execution_time,
            'runbooks_by_category': category_counts,
            'most_executed_runbooks': self._get_most_executed_runbooks(5)
        }
        
    def _get_most_executed_runbooks(self, limit: int) -> List[Dict[str, Any]]:
        """Get most frequently executed runbooks"""
        execution_counts = {}
        
        for execution in self.executions.values():
            runbook_id = execution.runbook_id
            execution_counts[runbook_id] = execution_counts.get(runbook_id, 0) + 1
            
        # Sort by execution count
        sorted_runbooks = sorted(execution_counts.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for runbook_id, count in sorted_runbooks[:limit]:
            if runbook_id in self.runbooks:
                runbook = self.runbooks[runbook_id]
                result.append({
                    'runbook_id': runbook_id,
                    'title': runbook.title,
                    'execution_count': count,
                    'category': runbook.category.value
                })
                
        return result
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get runbook management health status"""
        
        metrics = self.get_runbook_metrics()
        
        # Check for issues
        issues = []
        
        # Check for low success rate
        if metrics['success_rate_percent'] < 80:
            issues.append(f"Low success rate: {metrics['success_rate_percent']:.1f}%")
            
        # Check for outdated runbooks (no executions in 90 days)
        ninety_days_ago = datetime.utcnow() - timedelta(days=90)
        stale_runbooks = 0
        
        for runbook in self.runbooks.values():
            if runbook.status == RunbookStatus.ACTIVE:
                recent_executions = [
                    ex for ex in self.executions.values()
                    if ex.runbook_id == runbook.id and ex.started_at > ninety_days_ago
                ]
                
                if not recent_executions:
                    stale_runbooks += 1
                    
        if stale_runbooks > 0:
            issues.append(f"{stale_runbooks} runbooks not executed in 90 days")
            
        health = "healthy" if not issues else "degraded"
        
        return {
            'status': health,
            'total_runbooks': metrics['total_runbooks'],
            'active_runbooks': metrics['active_runbooks'],
            'success_rate_percent': metrics['success_rate_percent'],
            'stale_runbooks': stale_runbooks,
            'issues': issues
        }
        
    def get_runbook(self, runbook_id: str) -> Optional[Runbook]:
        """Get runbook by ID"""
        return self.runbooks.get(runbook_id)
        
    def list_runbooks(self,
                     category: Optional[RunbookCategory] = None,
                     status: Optional[RunbookStatus] = None) -> List[Runbook]:
        """List runbooks with optional filters"""
        runbooks = list(self.runbooks.values())
        
        if category:
            runbooks = [rb for rb in runbooks if rb.category == category]
        if status:
            runbooks = [rb for rb in runbooks if rb.status == status]
            
        # Sort by updated_at (newest first)
        runbooks.sort(key=lambda x: x.updated_at, reverse=True)
        return runbooks


# Global runbook manager instance
_runbook_manager = None


def initialize_runbook_management(config: Dict[str, Any]) -> RunbookManager:
    """Initialize runbook management system"""
    global _runbook_manager
    _runbook_manager = RunbookManager(config)
    return _runbook_manager


def get_runbook_manager() -> Optional[RunbookManager]:
    """Get the global runbook manager instance"""
    return _runbook_manager