"""
Story 5.4 Production Readiness Enhancement - Integration Testing
Comprehensive integration tests for production readiness components
"""

import pytest
import tempfile
import os
import yaml
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Import all production readiness components
from src.production.infrastructure import ProductionInfrastructure
from src.production.security import SecurityManager
from src.monitoring.production_monitor import ProductionMonitor
from src.scalability import initialize_scalability_infrastructure
from src.database import initialize_database_storage_infrastructure
from src.operations import initialize_operational_procedures


class TestProductionReadinessIntegration:
    """Integration tests for Story 5.4 production readiness components"""
    
    @pytest.fixture
    def production_config(self):
        """Production configuration for testing"""
        return {
            'environment': 'test',
            'instance_id': 'test-instance-01',
            'service_version': '2.0.0',
            
            # Security configuration
            'security': {
                'jwt': {
                    'secret_key': 'test-secret-key',
                    'token_expiry_hours': 24,
                    'algorithm': 'HS256'
                },
                'rbac': {
                    'roles': {
                        'admin': {
                            'permissions': ['process_files', 'view_files', 'manage_users'],
                            'description': 'Full system access'
                        }
                    }
                },
                'audit': {
                    'log_directory': '/tmp/test_audit',
                    'retention_days': 7,
                    'max_log_size_mb': 10
                },
                'scanner': {
                    'enabled': False  # Disable for testing
                }
            },
            
            # Monitoring configuration
            'monitoring': {
                'production': {
                    'environment': 'test',
                    'instance_id': 'test-instance-01',
                    'health_checks': {
                        'api_endpoint': {
                            'type': 'http',
                            'url': 'http://localhost:8080/health',
                            'timeout': 5,
                            'expected_status': 200,
                            'interval_seconds': 30
                        }
                    },
                    'alerting': {
                        'slack_webhook': 'http://test-webhook.example.com',
                        'email_config': {
                            'smtp_server': 'smtp.example.com',
                            'smtp_port': 587,
                            'username': 'test@example.com',
                            'password': 'test-password',
                            'from_address': 'test@example.com',
                            'to_addresses': ['ops@example.com']
                        }
                    },
                    'sla_targets': {
                        'availability': 99.9,
                        'response_time_ms': 1000,
                        'error_rate': 0.01
                    }
                }
            },
            
            # Database configuration
            'database': {
                'primary': {
                    'type': 'sqlite',  # Use SQLite for testing
                    'database': ':memory:',
                    'pool_size': 5,
                    'max_overflow': 10
                },
                'redis': {
                    'host': 'localhost',
                    'port': 6379,
                    'db': 1,
                    'max_connections': 10
                }
            },
            
            # Scalability configuration
            'scalability': {
                'load_balancing': {
                    'enabled': True,
                    'algorithm': 'round_robin',
                    'instances': [
                        {'host': 'test-host-1', 'port': 8080, 'weight': 1}
                    ]
                },
                'distributed_processing': {
                    'enabled': True,
                    'task_queue': {
                        'type': 'memory',  # Use memory queue for testing
                        'max_queue_size': 100
                    },
                    'worker_management': {
                        'enabled': True,
                        'min_workers': 1,
                        'max_workers': 2
                    }
                }
            },
            
            # Operations configuration
            'operations': {
                'incident_response': {
                    'escalation_matrix': [
                        {
                            'level': 1,
                            'response_time_minutes': 15,
                            'contacts': ['test@example.com']
                        }
                    ]
                },
                'maintenance': {
                    'weekly_window': {
                        'day': 'Sunday',
                        'start_time': '02:00',
                        'duration_hours': 4,
                        'timezone': 'UTC'
                    }
                },
                'runbooks': {
                    'deployment': 'https://test.example.com/runbooks/deployment'
                }
            }
        }
    
    @pytest.fixture
    def temp_directories(self):
        """Create temporary directories for testing"""
        temp_dir = tempfile.mkdtemp()
        directories = {
            'logs': os.path.join(temp_dir, 'logs'),
            'data': os.path.join(temp_dir, 'data'),
            'config': os.path.join(temp_dir, 'config'),
            'audit': os.path.join(temp_dir, 'audit')
        }
        
        for dir_path in directories.values():
            os.makedirs(dir_path, exist_ok=True)
            
        yield directories
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_infrastructure_initialization(self, production_config, temp_directories):
        """Test production infrastructure initialization"""
        # Initialize production infrastructure
        infrastructure = ProductionInfrastructure(production_config)
        
        # Verify infrastructure is properly initialized
        assert infrastructure.config == production_config
        assert infrastructure.environment == 'test'
        assert infrastructure.instance_id == 'test-instance-01'
        assert infrastructure.service_version == '2.0.0'
        
        # Test health status
        health_status = infrastructure.get_health_status()
        assert 'status' in health_status
        assert 'components' in health_status
        assert 'timestamp' in health_status
        
    def test_security_manager_integration(self, production_config, temp_directories):
        """Test security manager integration"""
        # Update config with temp directories
        security_config = production_config['security'].copy()
        security_config['audit']['log_directory'] = temp_directories['audit']
        
        # Initialize security manager
        security_manager = SecurityManager(security_config)
        
        # Test JWT token generation
        token = security_manager.generate_jwt_token('test_user', ['admin'])
        assert token is not None
        assert isinstance(token, str)
        
        # Test token validation
        payload = security_manager.validate_jwt_token(token)
        assert payload is not None
        assert payload['user'] == 'test_user'
        assert 'admin' in payload['roles']
        
        # Test RBAC permission check
        assert security_manager.check_permission('test_user', ['admin'], 'process_files')
        assert not security_manager.check_permission('test_user', ['viewer'], 'manage_users')
        
        # Test audit logging
        security_manager.log_security_event('test_event', 'test_user', {'action': 'test'})
        
        # Verify audit log was created
        audit_files = os.listdir(temp_directories['audit'])
        assert len(audit_files) > 0
        
    def test_production_monitor_integration(self, production_config):
        """Test production monitor integration"""
        # Initialize production monitor
        monitor = ProductionMonitor(production_config['monitoring']['production'])
        
        # Test health check registration
        health_checks = monitor.health_checker.health_checks
        assert 'api_endpoint' in health_checks
        
        # Test metrics collection
        monitor.collect_system_metrics()
        metrics = monitor.get_current_metrics()
        
        assert 'cpu_percent' in metrics
        assert 'memory_percent' in metrics
        assert 'timestamp' in metrics
        
        # Test SLA monitoring
        sla_status = monitor.check_sla_compliance()
        assert 'availability_sla' in sla_status
        assert 'response_time_sla' in sla_status
        assert 'error_rate_sla' in sla_status
        
        # Cleanup
        monitor.shutdown()
        
    def test_scalability_infrastructure_integration(self, production_config):
        """Test scalability infrastructure integration"""
        scalability_config = production_config['scalability']
        
        # Initialize scalability infrastructure
        components = initialize_scalability_infrastructure(scalability_config)
        
        # Verify components are initialized
        assert 'load_balancer' in components
        assert 'distributed_processor' in components
        
        # Test load balancer
        load_balancer = components['load_balancer']
        assert hasattr(load_balancer, 'get_next_instance')
        assert hasattr(load_balancer, 'get_health_status')
        
        # Test distributed processor
        distributed_processor = components['distributed_processor']
        assert hasattr(distributed_processor, 'submit_task')
        assert hasattr(distributed_processor, 'get_health_status')
        
        # Cleanup
        for component in components.values():
            if hasattr(component, 'shutdown'):
                component.shutdown()
                
    def test_database_storage_integration(self, production_config):
        """Test database and storage integration"""
        database_config = production_config['database']
        
        # Initialize database and storage infrastructure
        components = initialize_database_storage_infrastructure(database_config)
        
        # Verify components are initialized
        assert 'database_manager' in components
        assert 'storage_manager' in components
        
        # Test database manager
        database_manager = components['database_manager']
        assert hasattr(database_manager, 'execute_query')
        assert hasattr(database_manager, 'get_health_status')
        
        # Test storage manager
        storage_manager = components['storage_manager']
        assert hasattr(storage_manager, 'store_file')
        assert hasattr(storage_manager, 'get_health_status')
        
        # Cleanup
        for component in components.values():
            if hasattr(component, 'shutdown'):
                component.shutdown()
                
    def test_operational_procedures_integration(self, production_config):
        """Test operational procedures integration"""
        operations_config = production_config['operations']
        
        # Initialize operational procedures
        components = initialize_operational_procedures(operations_config)
        
        # Verify components are initialized
        assert 'incident_manager' in components
        assert 'maintenance_manager' in components
        assert 'runbook_manager' in components
        assert 'operations_coordinator' in components
        
        # Test incident manager
        incident_manager = components['incident_manager']
        incident_id = incident_manager.create_incident(
            'Test Incident',
            'Test incident description',
            incident_manager.IncidentSeverity.MEDIUM
        )
        assert incident_id is not None
        
        # Test maintenance manager
        maintenance_manager = components['maintenance_manager']
        maintenance_id = maintenance_manager.schedule_maintenance(
            'Test Maintenance',
            'Test maintenance description',
            maintenance_manager.MaintenanceType.SCHEDULED,
            datetime.utcnow() + timedelta(hours=1),
            60,
            ['test-service']
        )
        assert maintenance_id is not None
        
        # Test runbook manager
        runbook_manager = components['runbook_manager']
        runbook_id = runbook_manager.create_runbook(
            'Test Runbook',
            'Test runbook description',
            runbook_manager.RunbookCategory.MAINTENANCE,
            'test_author',
            [{'title': 'Step 1', 'description': 'Test step', 'command': 'echo test'}]
        )
        assert runbook_id is not None
        
        # Test operations coordinator
        operations_coordinator = components['operations_coordinator']
        operational_status = operations_coordinator.get_operational_status()
        assert 'overall_health' in operational_status
        assert 'components' in operational_status
        
        # Cleanup
        for component in components.values():
            if hasattr(component, 'shutdown'):
                component.shutdown()
                
    def test_end_to_end_production_workflow(self, production_config, temp_directories):
        """Test complete end-to-end production workflow"""
        # Update config with temp directories
        config = production_config.copy()
        config['security']['audit']['log_directory'] = temp_directories['audit']
        
        # Initialize all production components
        infrastructure = ProductionInfrastructure(config)
        security_manager = SecurityManager(config['security'])
        monitor = ProductionMonitor(config['monitoring']['production'])
        
        scalability_components = initialize_scalability_infrastructure(config['scalability'])
        database_components = initialize_database_storage_infrastructure(config['database'])
        operations_components = initialize_operational_procedures(config['operations'])
        
        try:
            # Simulate production workflow
            
            # 1. User authentication
            token = security_manager.generate_jwt_token('test_user', ['admin'])
            payload = security_manager.validate_jwt_token(token)
            assert payload is not None
            
            # 2. Permission check for file processing
            can_process = security_manager.check_permission('test_user', ['admin'], 'process_files')
            assert can_process
            
            # 3. Log security event
            security_manager.log_security_event('file_processing_started', 'test_user', {
                'file_count': 1,
                'processing_type': 'test'
            })
            
            # 4. Submit processing task to distributed system
            distributed_processor = scalability_components['distributed_processor']
            task_id = distributed_processor.submit_task('process_file', {
                'file_path': '/tmp/test.srt',
                'user': 'test_user'
            })
            assert task_id is not None
            
            # 5. Monitor system health during processing
            metrics = monitor.get_current_metrics()
            assert 'cpu_percent' in metrics
            
            # 6. Check SLA compliance
            sla_status = monitor.check_sla_compliance()
            assert sla_status['availability_sla']['compliant']
            
            # 7. Create operational incident if needed
            operations_coordinator = operations_components['operations_coordinator']
            if metrics['cpu_percent'] > 90:
                incident_id = operations_coordinator.create_operational_incident(
                    'High CPU Usage',
                    f"CPU usage at {metrics['cpu_percent']}%",
                    operations_coordinator.IncidentSeverity.MEDIUM
                )
            
            # 8. Get overall operational status
            operational_status = operations_coordinator.get_operational_status()
            assert operational_status['overall_health'] in ['healthy', 'degraded', 'critical']
            
            # 9. Store processing results
            storage_manager = database_components['storage_manager']
            file_id = storage_manager.store_file('/tmp/test_result.srt', {
                'original_file': '/tmp/test.srt',
                'processed_by': 'test_user',
                'processing_time': 1.5
            })
            
            # 10. Log completion
            security_manager.log_security_event('file_processing_completed', 'test_user', {
                'task_id': task_id,
                'file_id': file_id,
                'success': True
            })
            
            print("✅ End-to-end production workflow completed successfully")
            
        finally:
            # Cleanup all components
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
                    
    def test_production_readiness_validation(self, production_config):
        """Validate overall production readiness"""
        validation_results = {
            'infrastructure': False,
            'security': False,
            'monitoring': False,
            'scalability': False,
            'database': False,
            'operations': False
        }
        
        try:
            # Test infrastructure
            infrastructure = ProductionInfrastructure(production_config)
            health = infrastructure.get_health_status()
            validation_results['infrastructure'] = health['status'] in ['healthy', 'degraded']
            
            # Test security
            security_manager = SecurityManager(production_config['security'])
            token = security_manager.generate_jwt_token('test', ['admin'])
            validation_results['security'] = token is not None
            
            # Test monitoring
            monitor = ProductionMonitor(production_config['monitoring']['production'])
            metrics = monitor.get_current_metrics()
            validation_results['monitoring'] = 'cpu_percent' in metrics
            monitor.shutdown()
            
            # Test scalability
            scalability_components = initialize_scalability_infrastructure(production_config['scalability'])
            validation_results['scalability'] = len(scalability_components) > 0
            for component in scalability_components.values():
                if hasattr(component, 'shutdown'):
                    component.shutdown()
            
            # Test database
            database_components = initialize_database_storage_infrastructure(production_config['database'])
            validation_results['database'] = len(database_components) > 0
            for component in database_components.values():
                if hasattr(component, 'shutdown'):
                    component.shutdown()
            
            # Test operations
            operations_components = initialize_operational_procedures(production_config['operations'])
            validation_results['operations'] = len(operations_components) > 0
            for component in operations_components.values():
                if hasattr(component, 'shutdown'):
                    component.shutdown()
                    
        except Exception as e:
            pytest.fail(f"Production readiness validation failed: {e}")
        
        # Assert all components are ready
        for component, is_ready in validation_results.items():
            assert is_ready, f"{component} component failed validation"
            
        # Calculate overall readiness score
        total_components = len(validation_results)
        ready_components = sum(validation_results.values())
        readiness_score = (ready_components / total_components) * 100
        
        print(f"✅ Production Readiness Score: {readiness_score}%")
        assert readiness_score == 100.0, f"Production readiness incomplete: {readiness_score}%"
        
    def test_production_performance_validation(self, production_config):
        """Validate production performance requirements"""
        monitor = ProductionMonitor(production_config['monitoring']['production'])
        
        try:
            # Collect performance metrics multiple times
            performance_samples = []
            for i in range(5):
                start_time = time.time()
                metrics = monitor.get_current_metrics()
                processing_time = (time.time() - start_time) * 1000  # Convert to ms
                
                performance_samples.append({
                    'processing_time_ms': processing_time,
                    'cpu_percent': metrics.get('cpu_percent', 0),
                    'memory_percent': metrics.get('memory_percent', 0)
                })
                
                time.sleep(0.1)  # Short delay between samples
            
            # Calculate performance statistics
            avg_processing_time = sum(s['processing_time_ms'] for s in performance_samples) / len(performance_samples)
            max_processing_time = max(s['processing_time_ms'] for s in performance_samples)
            avg_cpu = sum(s['cpu_percent'] for s in performance_samples) / len(performance_samples)
            avg_memory = sum(s['memory_percent'] for s in performance_samples) / len(performance_samples)
            
            # Validate against SLA targets
            sla_targets = production_config['monitoring']['production']['sla_targets']
            
            assert avg_processing_time < sla_targets['response_time_ms'], f"Average response time {avg_processing_time}ms exceeds SLA {sla_targets['response_time_ms']}ms"
            assert max_processing_time < sla_targets['response_time_ms'] * 2, f"Max response time {max_processing_time}ms too high"
            assert avg_cpu < 80, f"Average CPU usage {avg_cpu}% too high"
            assert avg_memory < 85, f"Average memory usage {avg_memory}% too high"
            
            print(f"✅ Performance validation passed:")
            print(f"   Average response time: {avg_processing_time:.2f}ms")
            print(f"   Maximum response time: {max_processing_time:.2f}ms")
            print(f"   Average CPU usage: {avg_cpu:.1f}%")
            print(f"   Average memory usage: {avg_memory:.1f}%")
            
        finally:
            monitor.shutdown()


if __name__ == '__main__':
    """Run integration tests directly"""
    pytest.main([__file__, '-v', '--tb=short'])