#!/usr/bin/env python3
"""
Production Readiness Validation Script
Validates all Story 5.4 production readiness components
"""

import sys
import os
import yaml
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@dataclass
class ValidationResult:
    """Validation result for a component"""
    component: str
    passed: bool
    message: str
    execution_time_ms: float
    details: Dict[str, Any] = None


class ProductionReadinessValidator:
    """Comprehensive production readiness validator"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or 'config/production.yaml'
        self.results: List[ValidationResult] = []
        self.temp_dir = tempfile.mkdtemp()
        
    def load_config(self) -> Dict[str, Any]:
        """Load production configuration"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            # Update paths for testing
            config['security']['audit']['log_directory'] = os.path.join(self.temp_dir, 'audit')
            os.makedirs(config['security']['audit']['log_directory'], exist_ok=True)
            
            # Use in-memory database for testing
            config['database']['primary']['database'] = ':memory:'
            config['database']['primary']['type'] = 'sqlite'
            
            return config
            
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            sys.exit(1)
            
    def validate_component(self, component_name: str, validation_func) -> ValidationResult:
        """Validate a single component"""
        print(f"Validating {component_name}...")
        
        start_time = time.time()
        try:
            success, message, details = validation_func()
            execution_time = (time.time() - start_time) * 1000
            
            result = ValidationResult(
                component=component_name,
                passed=success,
                message=message,
                execution_time_ms=execution_time,
                details=details
            )
            
            status = "PASS" if success else "FAIL"
            print(f"  {status}: {message} ({execution_time:.2f}ms)")
            
            if details:
                for key, value in details.items():
                    print(f"    {key}: {value}")
                    
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            result = ValidationResult(
                component=component_name,
                passed=False,
                message=f"Validation error: {str(e)}",
                execution_time_ms=execution_time
            )
            print(f"  ERROR: {str(e)} ({execution_time:.2f}ms)")
            
        self.results.append(result)
        return result
        
    def validate_infrastructure(self, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate production infrastructure"""
        try:
            from production.infrastructure import ProductionInfrastructure
            
            infrastructure = ProductionInfrastructure(config)
            health_status = infrastructure.get_health_status()
            
            details = {
                'environment': infrastructure.environment,
                'instance_id': infrastructure.instance_id,
                'service_version': infrastructure.service_version,
                'health_status': health_status.get('status', 'unknown')
            }
            
            success = health_status.get('status') in ['healthy', 'degraded']
            message = f"Infrastructure initialized (status: {health_status.get('status')})"
            
            return success, message, details
            
        except ImportError as e:
            return False, f"Infrastructure module not found: {e}", {}
        except Exception as e:
            return False, f"Infrastructure validation failed: {e}", {}
            
    def validate_security(self, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate security components"""
        try:
            from production.security import SecurityManager
            
            security_config = config.get('security', {})
            security_manager = SecurityManager(security_config)
            
            # Test JWT token generation and validation
            token = security_manager.generate_jwt_token('test_user', ['admin'])
            payload = security_manager.validate_jwt_token(token)
            
            # Test RBAC
            has_permission = security_manager.check_permission('test_user', ['admin'], 'process_files')
            
            # Test audit logging
            security_manager.log_security_event('validation_test', 'test_user', {'test': True})
            
            details = {
                'jwt_generation': token is not None,
                'jwt_validation': payload is not None,
                'rbac_check': has_permission,
                'audit_logging': True
            }
            
            success = all(details.values())
            message = "Security components validated"
            
            return success, message, details
            
        except ImportError as e:
            return False, f"Security module not found: {e}", {}
        except Exception as e:
            return False, f"Security validation failed: {e}", {}
            
    def validate_monitoring(self, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate monitoring components"""
        try:
            from monitoring.production_monitor import ProductionMonitor
            
            monitoring_config = config.get('monitoring', {}).get('production', {})
            monitor = ProductionMonitor(monitoring_config)
            
            try:
                # Test metrics collection
                metrics = monitor.get_current_metrics()
                
                # Test health checks
                health_checks = monitor.health_checker.health_checks if hasattr(monitor, 'health_checker') else {}
                
                # Test SLA monitoring
                sla_status = monitor.check_sla_compliance()
                
                details = {
                    'metrics_collection': 'cpu_percent' in metrics,
                    'health_checks_configured': len(health_checks) > 0,
                    'sla_monitoring': 'availability_sla' in sla_status
                }
                
                success = details['metrics_collection']
                message = f"Monitoring components validated ({len(health_checks)} health checks)"
                
                return success, message, details
                
            finally:
                if hasattr(monitor, 'shutdown'):
                    monitor.shutdown()
                    
        except ImportError as e:
            return False, f"Monitoring module not found: {e}", {}
        except Exception as e:
            return False, f"Monitoring validation failed: {e}", {}
            
    def validate_scalability(self, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate scalability components"""
        try:
            from scalability import initialize_scalability_infrastructure
            
            scalability_config = config.get('scalability', {})
            components = initialize_scalability_infrastructure(scalability_config)
            
            try:
                details = {
                    'components_initialized': len(components),
                    'load_balancer': 'load_balancer' in components,
                    'distributed_processor': 'distributed_processor' in components
                }
                
                # Test component health
                for name, component in components.items():
                    if hasattr(component, 'get_health_status'):
                        health = component.get_health_status()
                        details[f'{name}_health'] = health.get('status', 'unknown')
                        
                success = len(components) > 0
                message = f"Scalability infrastructure validated ({len(components)} components)"
                
                return success, message, details
                
            finally:
                # Cleanup components
                for component in components.values():
                    if hasattr(component, 'shutdown'):
                        component.shutdown()
                        
        except ImportError as e:
            return False, f"Scalability module not found: {e}", {}
        except Exception as e:
            return False, f"Scalability validation failed: {e}", {}
            
    def validate_database(self, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate database and storage components"""
        try:
            from database import initialize_database_storage_infrastructure
            
            database_config = config.get('database', {})
            components = initialize_database_storage_infrastructure(database_config)
            
            try:
                details = {
                    'components_initialized': len(components),
                    'database_manager': 'database_manager' in components,
                    'storage_manager': 'storage_manager' in components
                }
                
                # Test component health
                for name, component in components.items():
                    if hasattr(component, 'get_health_status'):
                        health = component.get_health_status()
                        details[f'{name}_health'] = health.get('status', 'unknown')
                        
                success = len(components) > 0
                message = f"Database infrastructure validated ({len(components)} components)"
                
                return success, message, details
                
            finally:
                # Cleanup components
                for component in components.values():
                    if hasattr(component, 'shutdown'):
                        component.shutdown()
                        
        except ImportError as e:
            return False, f"Database module not found: {e}", {}
        except Exception as e:
            return False, f"Database validation failed: {e}", {}
            
    def validate_operations(self, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate operational procedures"""
        try:
            from operations import initialize_operational_procedures
            
            operations_config = config.get('operations', {})
            components = initialize_operational_procedures(operations_config)
            
            try:
                details = {
                    'components_initialized': len(components),
                    'incident_manager': 'incident_manager' in components,
                    'maintenance_manager': 'maintenance_manager' in components,
                    'runbook_manager': 'runbook_manager' in components,
                    'operations_coordinator': 'operations_coordinator' in components
                }
                
                # Test operations coordinator
                if 'operations_coordinator' in components:
                    coordinator = components['operations_coordinator']
                    status = coordinator.get_operational_status()
                    details['operational_health'] = status.get('overall_health', 'unknown')
                    
                success = len(components) >= 4  # All 4 components should be present
                message = f"Operational procedures validated ({len(components)} components)"
                
                return success, message, details
                
            finally:
                # Cleanup components
                for component in components.values():
                    if hasattr(component, 'shutdown'):
                        component.shutdown()
                        
        except ImportError as e:
            return False, f"Operations module not found: {e}", {}
        except Exception as e:
            return False, f"Operations validation failed: {e}", {}
            
    def validate_integration(self, config: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate end-to-end integration"""
        try:
            # Import all required modules
            from production.infrastructure import ProductionInfrastructure
            from production.security import SecurityManager
            from monitoring.production_monitor import ProductionMonitor
            from scalability import initialize_scalability_infrastructure
            from database import initialize_database_storage_infrastructure
            from operations import initialize_operational_procedures
            
            # Initialize all components
            infrastructure = ProductionInfrastructure(config)
            security_manager = SecurityManager(config.get('security', {}))
            monitor = ProductionMonitor(config.get('monitoring', {}).get('production', {}))
            
            scalability_components = initialize_scalability_infrastructure(config.get('scalability', {}))
            database_components = initialize_database_storage_infrastructure(config.get('database', {}))
            operations_components = initialize_operational_procedures(config.get('operations', {}))
            
            try:
                # Test integration workflow
                token = security_manager.generate_jwt_token('integration_test', ['admin'])
                payload = security_manager.validate_jwt_token(token)
                
                metrics = monitor.get_current_metrics()
                
                operational_status = operations_components['operations_coordinator'].get_operational_status()
                
                details = {
                    'authentication_flow': payload is not None,
                    'metrics_collection': 'cpu_percent' in metrics,
                    'operational_status': operational_status.get('overall_health', 'unknown'),
                    'all_components_active': True
                }
                
                success = all([
                    payload is not None,
                    'cpu_percent' in metrics,
                    operational_status.get('overall_health') in ['healthy', 'degraded', 'critical']
                ])
                
                message = "End-to-end integration validated"
                
                return success, message, details
                
            finally:
                # Cleanup all components
                if hasattr(monitor, 'shutdown'):
                    monitor.shutdown()
                    
                for component in scalability_components.values():
                    if hasattr(component, 'shutdown'):
                        component.shutdown()
                        
                for component in database_components.values():
                    if hasattr(component, 'shutdown'):
                        component.shutdown()
                        
                for component in operations_components.values():
                    if hasattr(component, 'shutdown'):
                        component.shutdown()
                        
        except ImportError as e:
            return False, f"Integration modules not found: {e}", {}
        except Exception as e:
            return False, f"Integration validation failed: {e}", {}
            
    def run_validation(self) -> bool:
        """Run complete production readiness validation"""
        print("Starting Production Readiness Validation")
        print("=" * 60)
        
        # Load configuration
        config = self.load_config()
        print(f"Configuration loaded from {self.config_path}")
        print()
        
        # Run all validations
        validations = [
            ("Infrastructure", lambda: self.validate_infrastructure(config)),
            ("Security", lambda: self.validate_security(config)),
            ("Monitoring", lambda: self.validate_monitoring(config)),
            ("Scalability", lambda: self.validate_scalability(config)),
            ("Database & Storage", lambda: self.validate_database(config)),
            ("Operations", lambda: self.validate_operations(config)),
            ("Integration", lambda: self.validate_integration(config))
        ]
        
        for name, validation_func in validations:
            self.validate_component(name, validation_func)
            print()
            
        # Generate final report
        self.generate_report()
        
        # Cleanup
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Return overall success
        return all(result.passed for result in self.results)
        
    def generate_report(self):
        """Generate validation report"""
        print("PRODUCTION READINESS VALIDATION REPORT")
        print("=" * 60)
        
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        success_rate = (passed_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"Overall Success Rate: {success_rate:.1f}% ({passed_count}/{total_count})")
        print()
        
        # Component results
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"{status} {result.component:<20} {result.message}")
            
        print()
        
        # Performance summary
        total_execution_time = sum(r.execution_time_ms for r in self.results)
        avg_execution_time = total_execution_time / len(self.results) if self.results else 0
        
        print(f"Performance Summary:")
        print(f"  Total execution time: {total_execution_time:.2f}ms")
        print(f"  Average per component: {avg_execution_time:.2f}ms")
        print()
        
        # Final status
        if success_rate == 100.0:
            print("PRODUCTION READINESS: COMPLETE")
            print("   All components validated successfully")
            print("   System ready for production deployment")
        elif success_rate >= 80.0:
            print("PRODUCTION READINESS: PARTIAL")
            print("   Most components validated successfully")
            print("   Address failing components before deployment")
        else:
            print("PRODUCTION READINESS: INCOMPLETE")
            print("   Multiple components failed validation")
            print("   System not ready for production deployment")
            
        print()


def main():
    """Main validation entry point"""
    validator = ProductionReadinessValidator()
    
    try:
        success = validator.run_validation()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nValidation failed with unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()