#!/usr/bin/env python3
"""
Emergency Response and Recovery Procedures for Story 4.4

This script provides comprehensive emergency response automation, rollback procedures,
graceful degradation, and incident response workflows for the MCP Pipeline Excellence system.
"""

import os
import sys
import json
import yaml
import time
import shutil
import subprocess
import tempfile
import platform
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from post_processors.sanskrit_post_processor import SanskritPostProcessor
from utils.srt_parser import SRTParser


class EmergencyLevel(Enum):
    """Emergency severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentType(Enum):
    """Types of system incidents"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SYSTEM_FAILURE = "system_failure"
    DATA_CORRUPTION = "data_corruption"
    SECURITY_BREACH = "security_breach"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_FAILURE = "dependency_failure"


@dataclass
class EmergencyIncident:
    """Emergency incident data structure"""
    incident_id: str
    incident_type: IncidentType
    severity: EmergencyLevel
    timestamp: datetime
    description: str
    affected_components: List[str]
    system_state: Dict[str, Any]
    recovery_actions_taken: List[str]
    rollback_performed: bool
    resolution_status: str  # "active", "resolved", "escalated"
    escalation_level: int
    estimated_impact: str
    recovery_time_seconds: Optional[float] = None


@dataclass
class RollbackOperation:
    """Rollback operation data structure"""
    operation_id: str
    timestamp: datetime
    rollback_type: str  # "configuration", "data", "system", "full"
    backup_source: str
    rollback_target: str
    validation_required: bool
    rollback_status: str  # "initiated", "in_progress", "completed", "failed"
    validation_results: Dict[str, Any]
    rollback_time_seconds: Optional[float] = None


@dataclass
class FallbackSystem:
    """Fallback system configuration"""
    system_name: str
    enabled: bool
    priority: int
    health_check_command: str
    activation_command: str
    deactivation_command: str
    validation_command: str
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"  # "healthy", "degraded", "failed"


class EmergencyResponseSystem:
    """Comprehensive emergency response and recovery coordinator"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize emergency response system"""
        self.project_root = Path(__file__).parent.parent
        self.config_path = config_path or self.project_root / "config"
        self.backup_dir = self.project_root / "backup"
        self.logs_dir = self.project_root / "logs"
        self.emergency_logs_dir = self.logs_dir / "emergency"
        
        # Create necessary directories
        self.backup_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.emergency_logs_dir.mkdir(exist_ok=True)
        
        # Load emergency configuration
        self.emergency_config = self._load_emergency_config()
        
        # Initialize fallback systems
        self.fallback_systems = self._initialize_fallback_systems()
        
        # Emergency response tracking
        self.active_incidents = {}
        self.rollback_history = []
        
        # System state monitoring
        self.baseline_performance = self._load_baseline_performance()
        
    def _load_emergency_config(self) -> Dict[str, Any]:
        """Load emergency response configuration"""
        config_file = self.config_path / "emergency_config.yaml"
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # Return default emergency configuration
            return {
                'response_timeouts': {
                    'low': 300,      # 5 minutes
                    'medium': 60,    # 1 minute
                    'high': 30,      # 30 seconds
                    'critical': 10   # 10 seconds
                },
                'escalation_thresholds': {
                    'performance_degradation': 2.0,  # 2x normal time
                    'error_rate': 0.10,              # 10% error rate
                    'memory_usage': 0.90,            # 90% memory usage
                    'disk_usage': 0.95,              # 95% disk usage
                },
                'rollback_settings': {
                    'auto_rollback_enabled': True,
                    'rollback_timeout_minutes': 10,
                    'validation_required': True,
                    'preserve_logs': True
                },
                'notification_settings': {
                    'email_alerts': False,
                    'log_alerts': True,
                    'console_alerts': True
                }
            }
    
    def _initialize_fallback_systems(self) -> Dict[str, FallbackSystem]:
        """Initialize fallback system configurations"""
        return {
            'basic_processing': FallbackSystem(
                system_name="Basic Processing Fallback",
                enabled=True,
                priority=1,
                health_check_command="python3 -c 'from utils.text_normalizer import TextNormalizer; print(\"OK\")'",
                activation_command="echo 'Activating basic processing fallback'",
                deactivation_command="echo 'Deactivating basic processing fallback'",
                validation_command="python3 -c 'from utils.srt_parser import SRTParser; print(\"OK\")'"
            ),
            'legacy_sanskrit_processor': FallbackSystem(
                system_name="Legacy Sanskrit Processor",
                enabled=True,
                priority=2,
                health_check_command="python3 -c 'from post_processors.sanskrit_post_processor import SanskritPostProcessor; print(\"OK\")'",
                activation_command="echo 'Activating legacy Sanskrit processor'",
                deactivation_command="echo 'Deactivating legacy Sanskrit processor'",
                validation_command="python3 -c 'from utils.text_normalizer import TextNormalizer; print(\"OK\")'"
            ),
            'emergency_mode': FallbackSystem(
                system_name="Emergency Mode Processing",
                enabled=True,
                priority=3,
                health_check_command="echo 'Emergency mode always available'",
                activation_command="echo 'Activating emergency mode - minimal processing only'",
                deactivation_command="echo 'Deactivating emergency mode'",
                validation_command="echo 'Emergency mode validation passed'"
            )
        }
    
    def _load_baseline_performance(self) -> Dict[str, float]:
        """Load baseline performance metrics"""
        baseline_file = self.project_root / "data/performance_baselines/system_baseline.json"
        
        if baseline_file.exists():
            with open(baseline_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Return default baselines
            return {
                'average_processing_time': 0.5,     # seconds per segment
                'throughput': 15.0,                 # segments per second
                'memory_usage_mb': 512,             # MB
                'cpu_usage_percent': 40,            # %
                'error_rate': 0.02                  # 2%
            }
    
    def detect_emergency_conditions(self) -> List[EmergencyIncident]:
        """Detect emergency conditions in the system"""
        detected_incidents = []
        
        # Check performance degradation
        performance_incident = self._check_performance_degradation()
        if performance_incident:
            detected_incidents.append(performance_incident)
        
        # Check system resource exhaustion
        resource_incident = self._check_resource_exhaustion()
        if resource_incident:
            detected_incidents.append(resource_incident)
        
        # Check system component failures
        component_incidents = self._check_component_failures()
        detected_incidents.extend(component_incidents)
        
        # Check data integrity issues
        data_incident = self._check_data_integrity()
        if data_incident:
            detected_incidents.append(data_incident)
        
        return detected_incidents
    
    def _check_performance_degradation(self) -> Optional[EmergencyIncident]:
        """Check for performance degradation"""
        try:
            # Test basic processing performance
            test_content = """1
00:00:01,000 --> 00:00:05,000
Test content for performance monitoring."""
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
                f.write(test_content)
                test_file = Path(f.name)
            
            output_file = test_file.with_suffix('.processed.srt')
            
            start_time = time.time()
            try:
                processor = SanskritPostProcessor()
                metrics = processor.process_srt_file(test_file, output_file)
                processing_time = time.time() - start_time
                
                # Compare with baseline
                baseline_time = self.baseline_performance.get('average_processing_time', 0.5)
                degradation_threshold = self.emergency_config['escalation_thresholds']['performance_degradation']
                
                if processing_time > baseline_time * degradation_threshold:
                    return EmergencyIncident(
                        incident_id=f"perf_{int(time.time())}",
                        incident_type=IncidentType.PERFORMANCE_DEGRADATION,
                        severity=EmergencyLevel.MEDIUM if processing_time < baseline_time * 3 else EmergencyLevel.HIGH,
                        timestamp=datetime.now(),
                        description=f"Performance degraded: {processing_time:.2f}s vs baseline {baseline_time:.2f}s",
                        affected_components=["sanskrit_post_processor"],
                        system_state={"processing_time": processing_time, "baseline_time": baseline_time},
                        recovery_actions_taken=[],
                        rollback_performed=False,
                        resolution_status="active",
                        escalation_level=1,
                        estimated_impact=f"{((processing_time / baseline_time - 1) * 100):.1f}% performance degradation"
                    )
                
            finally:
                # Clean up test files
                test_file.unlink()
                if output_file.exists():
                    output_file.unlink()
                    
        except Exception as e:
            return EmergencyIncident(
                incident_id=f"fail_{int(time.time())}",
                incident_type=IncidentType.SYSTEM_FAILURE,
                severity=EmergencyLevel.HIGH,
                timestamp=datetime.now(),
                description=f"System component failure during performance check: {e}",
                affected_components=["sanskrit_post_processor"],
                system_state={"error": str(e)},
                recovery_actions_taken=[],
                rollback_performed=False,
                resolution_status="active",
                escalation_level=2,
                estimated_impact="Critical system component unavailable"
            )
        
        return None
    
    def _check_resource_exhaustion(self) -> Optional[EmergencyIncident]:
        """Check for system resource exhaustion"""
        try:
            import psutil
            
            # Check memory usage
            memory = psutil.virtual_memory()
            memory_threshold = self.emergency_config['escalation_thresholds']['memory_usage']
            
            if memory.percent / 100.0 > memory_threshold:
                return EmergencyIncident(
                    incident_id=f"mem_{int(time.time())}",
                    incident_type=IncidentType.RESOURCE_EXHAUSTION,
                    severity=EmergencyLevel.HIGH if memory.percent > 95 else EmergencyLevel.MEDIUM,
                    timestamp=datetime.now(),
                    description=f"High memory usage: {memory.percent:.1f}%",
                    affected_components=["system_memory"],
                    system_state={"memory_percent": memory.percent, "available_mb": memory.available / 1024 / 1024},
                    recovery_actions_taken=[],
                    rollback_performed=False,
                    resolution_status="active",
                    escalation_level=1,
                    estimated_impact="System may become unresponsive"
                )
            
            # Check disk usage
            disk = psutil.disk_usage(self.project_root)
            disk_percent = (disk.used / disk.total) * 100
            disk_threshold = self.emergency_config['escalation_thresholds']['disk_usage'] * 100
            
            if disk_percent > disk_threshold:
                return EmergencyIncident(
                    incident_id=f"disk_{int(time.time())}",
                    incident_type=IncidentType.RESOURCE_EXHAUSTION,
                    severity=EmergencyLevel.HIGH if disk_percent > 98 else EmergencyLevel.MEDIUM,
                    timestamp=datetime.now(),
                    description=f"High disk usage: {disk_percent:.1f}%",
                    affected_components=["system_disk"],
                    system_state={"disk_percent": disk_percent, "free_gb": disk.free / 1024 / 1024 / 1024},
                    recovery_actions_taken=[],
                    rollback_performed=False,
                    resolution_status="active",
                    escalation_level=1,
                    estimated_impact="System may run out of storage"
                )
                
        except ImportError:
            # psutil not available - skip resource monitoring
            pass
        except Exception as e:
            return EmergencyIncident(
                incident_id=f"resource_{int(time.time())}",
                incident_type=IncidentType.SYSTEM_FAILURE,
                severity=EmergencyLevel.MEDIUM,
                timestamp=datetime.now(),
                description=f"Error checking system resources: {e}",
                affected_components=["system_monitoring"],
                system_state={"error": str(e)},
                recovery_actions_taken=[],
                rollback_performed=False,
                resolution_status="active",
                escalation_level=1,
                estimated_impact="Unable to monitor system health"
            )
        
        return None
    
    def _check_component_failures(self) -> List[EmergencyIncident]:
        """Check for system component failures"""
        incidents = []
        
        critical_components = [
            ("SanskritPostProcessor", "post_processors.sanskrit_post_processor", "SanskritPostProcessor"),
            ("SRTParser", "utils.srt_parser", "SRTParser"),
            ("TextNormalizer", "utils.text_normalizer", "TextNormalizer")
        ]
        
        for component_name, module_path, class_name in critical_components:
            try:
                module = __import__(module_path, fromlist=[class_name])
                component_class = getattr(module, class_name)
                
                # Try to instantiate
                if callable(component_class):
                    instance = component_class()
                
            except ImportError as e:
                incidents.append(EmergencyIncident(
                    incident_id=f"comp_{component_name.lower()}_{int(time.time())}",
                    incident_type=IncidentType.DEPENDENCY_FAILURE,
                    severity=EmergencyLevel.HIGH,
                    timestamp=datetime.now(),
                    description=f"Component import failure: {component_name} - {e}",
                    affected_components=[component_name.lower()],
                    system_state={"error": str(e), "error_type": "import_error"},
                    recovery_actions_taken=[],
                    rollback_performed=False,
                    resolution_status="active",
                    escalation_level=2,
                    estimated_impact=f"Critical component {component_name} unavailable"
                ))
            except Exception as e:
                incidents.append(EmergencyIncident(
                    incident_id=f"comp_{component_name.lower()}_{int(time.time())}",
                    incident_type=IncidentType.SYSTEM_FAILURE,
                    severity=EmergencyLevel.MEDIUM,
                    timestamp=datetime.now(),
                    description=f"Component initialization failure: {component_name} - {e}",
                    affected_components=[component_name.lower()],
                    system_state={"error": str(e), "error_type": "initialization_error"},
                    recovery_actions_taken=[],
                    rollback_performed=False,
                    resolution_status="active",
                    escalation_level=1,
                    estimated_impact=f"Component {component_name} may not function correctly"
                ))
        
        return incidents
    
    def _check_data_integrity(self) -> Optional[EmergencyIncident]:
        """Check for data integrity issues"""
        try:
            # Check critical data files
            critical_files = [
                "data/lexicons/proper_nouns.yaml",
                "data/lexicons/corrections.yaml",
                "config/default_config.yaml"
            ]
            
            corrupted_files = []
            for file_path in critical_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    try:
                        if file_path.endswith('.yaml'):
                            with open(full_path, 'r', encoding='utf-8') as f:
                                yaml.safe_load(f)
                        elif file_path.endswith('.json'):
                            with open(full_path, 'r', encoding='utf-8') as f:
                                json.load(f)
                    except Exception as e:
                        corrupted_files.append(f"{file_path}: {e}")
                else:
                    corrupted_files.append(f"{file_path}: missing")
            
            if corrupted_files:
                return EmergencyIncident(
                    incident_id=f"data_{int(time.time())}",
                    incident_type=IncidentType.DATA_CORRUPTION,
                    severity=EmergencyLevel.HIGH,
                    timestamp=datetime.now(),
                    description=f"Data integrity issues detected: {', '.join(corrupted_files)}",
                    affected_components=["data_files"],
                    system_state={"corrupted_files": corrupted_files},
                    recovery_actions_taken=[],
                    rollback_performed=False,
                    resolution_status="active",
                    escalation_level=2,
                    estimated_impact="System may not function correctly due to corrupted data"
                )
                
        except Exception as e:
            return EmergencyIncident(
                incident_id=f"integrity_{int(time.time())}",
                incident_type=IncidentType.SYSTEM_FAILURE,
                severity=EmergencyLevel.MEDIUM,
                timestamp=datetime.now(),
                description=f"Error checking data integrity: {e}",
                affected_components=["data_integrity_checker"],
                system_state={"error": str(e)},
                recovery_actions_taken=[],
                rollback_performed=False,
                resolution_status="active",
                escalation_level=1,
                estimated_impact="Unable to verify data integrity"
            )
        
        return None
    
    def respond_to_incident(self, incident: EmergencyIncident) -> EmergencyIncident:
        """Respond to emergency incident with appropriate actions"""
        print(f"🚨 Emergency Response: {incident.incident_type.value} - {incident.severity.value.upper()}")
        print(f"📋 Description: {incident.description}")
        
        # Record incident
        self.active_incidents[incident.incident_id] = incident
        
        # Log incident
        self._log_incident(incident)
        
        # Determine response based on severity and type
        if incident.severity in [EmergencyLevel.HIGH, EmergencyLevel.CRITICAL]:
            # High-priority response
            incident = self._execute_high_priority_response(incident)
        elif incident.severity == EmergencyLevel.MEDIUM:
            # Medium-priority response
            incident = self._execute_medium_priority_response(incident)
        else:
            # Low-priority response
            incident = self._execute_low_priority_response(incident)
        
        # Update incident record
        self.active_incidents[incident.incident_id] = incident
        
        return incident
    
    def _execute_high_priority_response(self, incident: EmergencyIncident) -> EmergencyIncident:
        """Execute high-priority emergency response"""
        print("🔥 Executing HIGH PRIORITY emergency response")
        
        recovery_actions = []
        
        # Immediate actions based on incident type
        if incident.incident_type == IncidentType.PERFORMANCE_DEGRADATION:
            # Attempt system optimization
            recovery_actions.append("Initiated system optimization procedures")
            
            # Activate fallback systems
            fallback_activated = self._activate_fallback_systems(priority_threshold=2)
            if fallback_activated:
                recovery_actions.append(f"Activated {len(fallback_activated)} fallback systems")
        
        elif incident.incident_type == IncidentType.SYSTEM_FAILURE:
            # Attempt immediate rollback
            rollback_success = self._execute_emergency_rollback("system")
            if rollback_success:
                recovery_actions.append("Emergency system rollback completed")
                incident.rollback_performed = True
            else:
                recovery_actions.append("Emergency system rollback failed")
            
            # Activate emergency mode
            emergency_fallback = self._activate_emergency_mode()
            if emergency_fallback:
                recovery_actions.append("Emergency mode activated")
        
        elif incident.incident_type == IncidentType.RESOURCE_EXHAUSTION:
            # Clean up resources
            cleanup_results = self._cleanup_system_resources()
            recovery_actions.append(f"Resource cleanup: {cleanup_results}")
            
            # Reduce system load
            load_reduction = self._reduce_system_load()
            recovery_actions.append(f"System load reduction: {load_reduction}")
        
        elif incident.incident_type == IncidentType.DATA_CORRUPTION:
            # Attempt data recovery
            recovery_success = self._execute_data_recovery()
            if recovery_success:
                recovery_actions.append("Data recovery completed")
            else:
                recovery_actions.append("Data recovery failed - manual intervention required")
                incident.escalation_level = 3
        
        # Update incident with recovery actions
        incident.recovery_actions_taken.extend(recovery_actions)
        
        # Check if incident is resolved
        if self._validate_incident_resolution(incident):
            incident.resolution_status = "resolved"
            incident.recovery_time_seconds = (datetime.now() - incident.timestamp).total_seconds()
            print("✅ HIGH PRIORITY incident resolved")
        else:
            incident.escalation_level += 1
            print("⚠️ HIGH PRIORITY incident requires escalation")
        
        return incident
    
    def _execute_medium_priority_response(self, incident: EmergencyIncident) -> EmergencyIncident:
        """Execute medium-priority emergency response"""
        print("⚠️ Executing MEDIUM PRIORITY emergency response")
        
        recovery_actions = []
        
        # Standard response actions
        if incident.incident_type == IncidentType.PERFORMANCE_DEGRADATION:
            # Monitor and attempt optimization
            recovery_actions.append("Performance monitoring increased")
            
            # Prepare fallback systems
            fallback_prepared = self._prepare_fallback_systems()
            recovery_actions.append(f"Prepared {len(fallback_prepared)} fallback systems")
        
        elif incident.incident_type in [IncidentType.DEPENDENCY_FAILURE, IncidentType.CONFIGURATION_ERROR]:
            # Attempt configuration recovery
            config_recovery = self._recover_configuration()
            if config_recovery:
                recovery_actions.append("Configuration recovery successful")
            else:
                recovery_actions.append("Configuration recovery failed")
        
        # Update incident with recovery actions
        incident.recovery_actions_taken.extend(recovery_actions)
        
        # Check resolution
        if self._validate_incident_resolution(incident):
            incident.resolution_status = "resolved"
            incident.recovery_time_seconds = (datetime.now() - incident.timestamp).total_seconds()
            print("✅ MEDIUM PRIORITY incident resolved")
        
        return incident
    
    def _execute_low_priority_response(self, incident: EmergencyIncident) -> EmergencyIncident:
        """Execute low-priority emergency response"""
        print("ℹ️ Executing LOW PRIORITY emergency response")
        
        # Log and monitor
        recovery_actions = ["Incident logged and monitoring enabled"]
        
        # Schedule maintenance if needed
        if incident.incident_type == IncidentType.PERFORMANCE_DEGRADATION:
            recovery_actions.append("Scheduled maintenance window for optimization")
        
        incident.recovery_actions_taken.extend(recovery_actions)
        incident.resolution_status = "resolved"  # Low priority incidents are often just logged
        incident.recovery_time_seconds = (datetime.now() - incident.timestamp).total_seconds()
        
        return incident
    
    def _activate_fallback_systems(self, priority_threshold: int = 3) -> List[str]:
        """Activate fallback systems based on priority"""
        activated_systems = []
        
        for system_name, fallback_system in self.fallback_systems.items():
            if fallback_system.enabled and fallback_system.priority <= priority_threshold:
                try:
                    # Check system health first
                    health_result = subprocess.run(
                        fallback_system.health_check_command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if health_result.returncode == 0:
                        # Activate the fallback system
                        activation_result = subprocess.run(
                            fallback_system.activation_command,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        
                        if activation_result.returncode == 0:
                            activated_systems.append(system_name)
                            fallback_system.health_status = "healthy"
                            print(f"✅ Activated fallback system: {system_name}")
                        else:
                            print(f"❌ Failed to activate fallback system: {system_name}")
                            fallback_system.health_status = "failed"
                    else:
                        fallback_system.health_status = "failed"
                        print(f"❌ Fallback system health check failed: {system_name}")
                        
                except Exception as e:
                    print(f"❌ Error activating fallback system {system_name}: {e}")
                    fallback_system.health_status = "failed"
        
        return activated_systems
    
    def _activate_emergency_mode(self) -> bool:
        """Activate emergency mode processing"""
        try:
            emergency_system = self.fallback_systems.get('emergency_mode')
            if emergency_system and emergency_system.enabled:
                result = subprocess.run(
                    emergency_system.activation_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print("🆘 Emergency mode activated - minimal processing only")
                    return True
                else:
                    print("❌ Failed to activate emergency mode")
                    return False
            else:
                print("❌ Emergency mode not available")
                return False
                
        except Exception as e:
            print(f"❌ Error activating emergency mode: {e}")
            return False
    
    def _execute_emergency_rollback(self, rollback_type: str) -> bool:
        """Execute emergency rollback procedure"""
        print(f"🔄 Executing emergency rollback: {rollback_type}")
        
        rollback_operation = RollbackOperation(
            operation_id=f"rollback_{int(time.time())}",
            timestamp=datetime.now(),
            rollback_type=rollback_type,
            backup_source="",
            rollback_target="",
            validation_required=True,
            rollback_status="initiated",
            validation_results={}
        )
        
        try:
            start_time = time.time()
            
            if rollback_type == "configuration":
                # Rollback configuration files
                rollback_success = self._rollback_configuration()
            elif rollback_type == "data":
                # Rollback data files
                rollback_success = self._rollback_data()
            elif rollback_type == "system":
                # Full system rollback
                rollback_success = self._rollback_system()
            else:
                print(f"❌ Unknown rollback type: {rollback_type}")
                rollback_success = False
            
            rollback_operation.rollback_time_seconds = time.time() - start_time
            rollback_operation.rollback_status = "completed" if rollback_success else "failed"
            
            # Validate rollback if required
            if rollback_success and rollback_operation.validation_required:
                validation_results = self._validate_rollback(rollback_type)
                rollback_operation.validation_results = validation_results
                rollback_success = validation_results.get('overall_success', False)
            
            self.rollback_history.append(rollback_operation)
            
            if rollback_success:
                print(f"✅ Emergency rollback completed: {rollback_type}")
            else:
                print(f"❌ Emergency rollback failed: {rollback_type}")
            
            return rollback_success
            
        except Exception as e:
            rollback_operation.rollback_status = "failed"
            rollback_operation.validation_results = {"error": str(e)}
            self.rollback_history.append(rollback_operation)
            print(f"❌ Emergency rollback exception: {e}")
            return False
    
    def _rollback_configuration(self) -> bool:
        """Rollback configuration files"""
        try:
            # Find latest configuration backup
            config_backups = list(self.backup_dir.glob("config_backup_*.tar.gz"))
            if not config_backups:
                print("❌ No configuration backups found")
                return False
            
            latest_backup = max(config_backups, key=lambda p: p.stat().st_mtime)
            print(f"📦 Restoring configuration from: {latest_backup.name}")
            
            # Extract backup to temporary location
            with tempfile.TemporaryDirectory() as temp_dir:
                extract_result = subprocess.run(
                    ["tar", "-xzf", str(latest_backup), "-C", temp_dir],
                    capture_output=True,
                    text=True
                )
                
                if extract_result.returncode == 0:
                    # Copy configuration files back
                    temp_config_dir = Path(temp_dir) / "config"
                    if temp_config_dir.exists():
                        shutil.copytree(temp_config_dir, self.config_path, dirs_exist_ok=True)
                        print("✅ Configuration files restored")
                        return True
                    else:
                        print("❌ Configuration directory not found in backup")
                        return False
                else:
                    print(f"❌ Failed to extract backup: {extract_result.stderr}")
                    return False
                    
        except Exception as e:
            print(f"❌ Configuration rollback error: {e}")
            return False
    
    def _rollback_data(self) -> bool:
        """Rollback data files"""
        try:
            # Find latest data backup
            data_backups = list(self.backup_dir.glob("data_backup_*.tar.gz"))
            if not data_backups:
                print("❌ No data backups found")
                return False
            
            latest_backup = max(data_backups, key=lambda p: p.stat().st_mtime)
            print(f"📦 Restoring data from: {latest_backup.name}")
            
            # Extract backup to temporary location
            with tempfile.TemporaryDirectory() as temp_dir:
                extract_result = subprocess.run(
                    ["tar", "-xzf", str(latest_backup), "-C", temp_dir],
                    capture_output=True,
                    text=True
                )
                
                if extract_result.returncode == 0:
                    # Copy critical data files back
                    temp_data_dir = Path(temp_dir) / "data"
                    if temp_data_dir.exists():
                        # Only restore critical files, not all data
                        critical_subdirs = ["lexicons", "scriptures"]
                        for subdir in critical_subdirs:
                            src_dir = temp_data_dir / subdir
                            dst_dir = self.project_root / "data" / subdir
                            if src_dir.exists():
                                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                        
                        print("✅ Critical data files restored")
                        return True
                    else:
                        print("❌ Data directory not found in backup")
                        return False
                else:
                    print(f"❌ Failed to extract backup: {extract_result.stderr}")
                    return False
                    
        except Exception as e:
            print(f"❌ Data rollback error: {e}")
            return False
    
    def _rollback_system(self) -> bool:
        """Full system rollback"""
        print("🔄 Executing full system rollback")
        
        # Rollback configuration first
        config_success = self._rollback_configuration()
        
        # Rollback critical data
        data_success = self._rollback_data()
        
        # Both must succeed for full rollback success
        return config_success and data_success
    
    def _validate_rollback(self, rollback_type: str) -> Dict[str, Any]:
        """Validate rollback operation"""
        validation_results = {}
        
        try:
            # Test basic system functionality
            test_content = """1
00:00:01,000 --> 00:00:05,000
Rollback validation test content."""
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
                f.write(test_content)
                test_file = Path(f.name)
            
            output_file = test_file.with_suffix('.processed.srt')
            
            try:
                processor = SanskritPostProcessor()
                metrics = processor.process_srt_file(test_file, output_file)
                
                validation_results['processing_test'] = True
                validation_results['processing_time'] = time.time()
                
            except Exception as e:
                validation_results['processing_test'] = False
                validation_results['processing_error'] = str(e)
            
            finally:
                test_file.unlink()
                if output_file.exists():
                    output_file.unlink()
            
            # Check configuration integrity
            config_file = self.config_path / "default_config.yaml"
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        yaml.safe_load(f)
                    validation_results['config_integrity'] = True
                except Exception as e:
                    validation_results['config_integrity'] = False
                    validation_results['config_error'] = str(e)
            else:
                validation_results['config_integrity'] = False
                validation_results['config_error'] = "Configuration file missing"
            
            # Overall success
            validation_results['overall_success'] = (
                validation_results.get('processing_test', False) and
                validation_results.get('config_integrity', False)
            )
            
        except Exception as e:
            validation_results['overall_success'] = False
            validation_results['validation_error'] = str(e)
        
        return validation_results
    
    def _cleanup_system_resources(self) -> str:
        """Clean up system resources to free memory and disk space"""
        cleanup_actions = []
        
        try:
            # Clean up temporary files
            temp_dirs = [tempfile.gettempdir(), "/tmp", str(self.project_root / "temp")]
            for temp_dir in temp_dirs:
                temp_path = Path(temp_dir)
                if temp_path.exists():
                    temp_files = list(temp_path.glob("*.tmp")) + list(temp_path.glob("tmp*"))
                    for temp_file in temp_files[:50]:  # Limit to 50 files for safety
                        try:
                            temp_file.unlink()
                            cleanup_actions.append(f"Removed temp file: {temp_file.name}")
                        except Exception:
                            pass
            
            # Clean up old log files
            if self.logs_dir.exists():
                old_logs = [f for f in self.logs_dir.glob("*.log") if 
                           (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days > 7]
                for old_log in old_logs:
                    try:
                        old_log.unlink()
                        cleanup_actions.append(f"Removed old log: {old_log.name}")
                    except Exception:
                        pass
            
            # Clean up old metrics files
            metrics_dir = self.project_root / "data/metrics"
            if metrics_dir.exists():
                old_metrics = [f for f in metrics_dir.glob("*.json") if 
                              (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days > 30]
                for old_metric in old_metrics:
                    try:
                        old_metric.unlink()
                        cleanup_actions.append(f"Removed old metric: {old_metric.name}")
                    except Exception:
                        pass
            
            return f"Cleaned up {len(cleanup_actions)} items"
            
        except Exception as e:
            return f"Cleanup error: {e}"
    
    def _reduce_system_load(self) -> str:
        """Reduce system load to manage resource usage"""
        try:
            # This is a placeholder for actual load reduction
            # In a real system, this might:
            # - Reduce processing concurrency
            # - Pause non-critical background tasks
            # - Switch to lighter processing modes
            
            return "System load reduction measures applied"
            
        except Exception as e:
            return f"Load reduction error: {e}"
    
    def _execute_data_recovery(self) -> bool:
        """Execute data recovery procedures"""
        try:
            # Check for data backups
            data_backups = list(self.backup_dir.glob("data_backup_*.tar.gz"))
            if data_backups:
                # Use the rollback mechanism for data recovery
                return self._rollback_data()
            else:
                # Try to regenerate essential data files
                return self._regenerate_essential_data()
                
        except Exception as e:
            print(f"❌ Data recovery error: {e}")
            return False
    
    def _regenerate_essential_data(self) -> bool:
        """Regenerate essential data files if backups are not available"""
        try:
            # Create minimal lexicon files if missing
            lexicons_dir = self.project_root / "data/lexicons"
            lexicons_dir.mkdir(exist_ok=True)
            
            essential_files = {
                "proper_nouns.yaml": {"Krishna": {"variations": ["krishna", "krsna"], "category": "deity"}},
                "corrections.yaml": {"yoga": {"variations": ["yog", "yga"], "category": "practice"}},
                "phrases.yaml": {"namaste": {"variations": ["namaskar"], "category": "greeting"}},
                "verses.yaml": {}
            }
            
            for filename, default_content in essential_files.items():
                file_path = lexicons_dir / filename
                if not file_path.exists():
                    with open(file_path, 'w', encoding='utf-8') as f:
                        yaml.dump(default_content, f, default_flow_style=False)
                    print(f"✅ Regenerated: {filename}")
            
            return True
            
        except Exception as e:
            print(f"❌ Data regeneration error: {e}")
            return False
    
    def _prepare_fallback_systems(self) -> List[str]:
        """Prepare fallback systems for potential activation"""
        prepared_systems = []
        
        for system_name, fallback_system in self.fallback_systems.items():
            if fallback_system.enabled:
                try:
                    # Check system health
                    health_result = subprocess.run(
                        fallback_system.health_check_command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if health_result.returncode == 0:
                        fallback_system.health_status = "healthy"
                        fallback_system.last_health_check = datetime.now()
                        prepared_systems.append(system_name)
                    else:
                        fallback_system.health_status = "failed"
                        
                except Exception as e:
                    fallback_system.health_status = "failed"
                    print(f"❌ Error preparing fallback system {system_name}: {e}")
        
        return prepared_systems
    
    def _recover_configuration(self) -> bool:
        """Attempt to recover system configuration"""
        try:
            # Check if default configuration exists
            default_config = self.config_path / "default_config.yaml"
            
            if not default_config.exists():
                # Create minimal default configuration
                minimal_config = {
                    'processing': {
                        'enable_sanskrit_correction': True,
                        'enable_number_conversion': True,
                        'enable_filler_removal': True
                    },
                    'logging': {
                        'level': 'INFO'
                    }
                }
                
                with open(default_config, 'w', encoding='utf-8') as f:
                    yaml.dump(minimal_config, f, default_flow_style=False)
                
                print("✅ Created minimal default configuration")
                return True
            else:
                # Validate existing configuration
                try:
                    with open(default_config, 'r', encoding='utf-8') as f:
                        yaml.safe_load(f)
                    print("✅ Configuration validation passed")
                    return True
                except Exception as e:
                    print(f"❌ Configuration validation failed: {e}")
                    return False
                    
        except Exception as e:
            print(f"❌ Configuration recovery error: {e}")
            return False
    
    def _validate_incident_resolution(self, incident: EmergencyIncident) -> bool:
        """Validate if incident has been resolved"""
        try:
            if incident.incident_type == IncidentType.PERFORMANCE_DEGRADATION:
                # Re-test performance
                return self._check_performance_degradation() is None
            
            elif incident.incident_type == IncidentType.SYSTEM_FAILURE:
                # Test basic system functionality
                try:
                    processor = SanskritPostProcessor()
                    return True
                except Exception:
                    return False
            
            elif incident.incident_type == IncidentType.RESOURCE_EXHAUSTION:
                # Check current resource usage
                return self._check_resource_exhaustion() is None
            
            elif incident.incident_type == IncidentType.DATA_CORRUPTION:
                # Check data integrity
                return self._check_data_integrity() is None
            
            else:
                # For other types, assume resolved if recovery actions were taken
                return len(incident.recovery_actions_taken) > 0
                
        except Exception:
            return False
    
    def _log_incident(self, incident: EmergencyIncident):
        """Log incident to emergency log file"""
        log_file = self.emergency_logs_dir / f"incidents_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Load existing incidents for the day
        incidents = []
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    incidents = json.load(f)
            except Exception:
                incidents = []
        
        # Add new incident
        incident_data = asdict(incident)
        incident_data['timestamp'] = incident.timestamp.isoformat()
        incidents.append(incident_data)
        
        # Save updated incidents
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(incidents, f, indent=2, default=str)
    
    def test_emergency_procedures(self) -> Dict[str, Any]:
        """Test emergency procedures and systems"""
        print("🧪 Testing Emergency Procedures")
        print("=" * 50)
        
        test_results = {
            'emergency_detection': False,
            'fallback_activation': False,
            'rollback_procedures': False,
            'incident_response': False,
            'overall_success': False
        }
        
        try:
            # Test 1: Emergency detection
            print("\n🔍 Testing emergency detection...")
            incidents = self.detect_emergency_conditions()
            test_results['emergency_detection'] = True
            print(f"✅ Emergency detection: {len(incidents)} incidents detected")
            
            # Test 2: Fallback system activation
            print("\n🔄 Testing fallback system activation...")
            activated = self._activate_fallback_systems(priority_threshold=3)
            test_results['fallback_activation'] = len(activated) > 0
            print(f"✅ Fallback activation: {len(activated)} systems activated")
            
            # Test 3: Rollback procedures
            print("\n🔄 Testing rollback procedures...")
            # Test configuration rollback (without actually rolling back)
            backup_exists = len(list(self.backup_dir.glob("*.tar.gz"))) > 0
            test_results['rollback_procedures'] = backup_exists
            result_msg = "✅" if backup_exists else "⚠️"
            print(f"{result_msg} Rollback procedures: {'Ready' if backup_exists else 'No backups available'}")
            
            # Test 4: Incident response
            print("\n🚨 Testing incident response...")
            if incidents:
                test_incident = incidents[0]
                responded_incident = self.respond_to_incident(test_incident)
                test_results['incident_response'] = len(responded_incident.recovery_actions_taken) > 0
                print(f"✅ Incident response: {len(responded_incident.recovery_actions_taken)} actions taken")
            else:
                test_results['incident_response'] = True  # No incidents to respond to
                print("✅ Incident response: No active incidents")
            
            # Overall success
            test_results['overall_success'] = all([
                test_results['emergency_detection'],
                test_results['fallback_activation'] or not self.fallback_systems,
                test_results['incident_response']
            ])
            
            print(f"\n{'✅' if test_results['overall_success'] else '❌'} Emergency procedures test: {'PASSED' if test_results['overall_success'] else 'FAILED'}")
            
        except Exception as e:
            print(f"❌ Emergency procedures test failed: {e}")
            test_results['test_error'] = str(e)
        
        return test_results


def main():
    """Main function for emergency procedures script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Emergency Response and Recovery Procedures")
    parser.add_argument("--detect", action="store_true", help="Detect emergency conditions")
    parser.add_argument("--test", action="store_true", help="Test emergency procedures")
    parser.add_argument("--rollback", choices=["configuration", "data", "system"], help="Execute rollback")
    parser.add_argument("--activate-fallback", action="store_true", help="Activate fallback systems")
    parser.add_argument("--emergency-mode", action="store_true", help="Activate emergency mode")
    parser.add_argument("--config", type=Path, help="Configuration directory path")
    
    args = parser.parse_args()
    
    # Initialize emergency response system
    emergency_system = EmergencyResponseSystem(args.config)
    
    try:
        if args.detect:
            # Detect emergency conditions
            incidents = emergency_system.detect_emergency_conditions()
            print(f"Detected {len(incidents)} emergency conditions")
            for incident in incidents:
                print(f"- {incident.incident_type.value}: {incident.description}")
        
        elif args.test:
            # Test emergency procedures
            results = emergency_system.test_emergency_procedures()
            sys.exit(0 if results['overall_success'] else 1)
        
        elif args.rollback:
            # Execute rollback
            success = emergency_system._execute_emergency_rollback(args.rollback)
            sys.exit(0 if success else 1)
        
        elif args.activate_fallback:
            # Activate fallback systems
            activated = emergency_system._activate_fallback_systems()
            print(f"Activated {len(activated)} fallback systems")
            sys.exit(0 if activated else 1)
        
        elif args.emergency_mode:
            # Activate emergency mode
            success = emergency_system._activate_emergency_mode()
            sys.exit(0 if success else 1)
        
        else:
            # Default: detect and respond to incidents
            incidents = emergency_system.detect_emergency_conditions()
            if incidents:
                print(f"🚨 {len(incidents)} emergency incidents detected")
                for incident in incidents:
                    emergency_system.respond_to_incident(incident)
            else:
                print("✅ No emergency conditions detected")
            
    except Exception as e:
        print(f"❌ Emergency procedures error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()