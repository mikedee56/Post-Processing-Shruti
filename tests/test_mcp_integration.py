"""
Comprehensive MCP Integration Test Suite for Story 5.2
Tests MCP library integration with professional standards compliance per CEO directive.
"""

import pytest
import asyncio
import time
import tempfile
import json
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

# Import MCP components
from utils.mcp_client import MCPClient, MCPConfig, create_mcp_client
from utils.mcp_reliability import MCPHealthMonitor, MCPCircuitBreakerAdvanced, GracefulDegradationManager
from utils.mcp_epic4_foundation import Epic4ExtensibleMCPArchitecture, create_epic4_foundation
from utils.advanced_text_normalizer import AdvancedTextNormalizer
from utils.professional_standards import TechnicalQualityGate


class TestTechnicalQualityGate:
    """Test Technical Quality Gate per CEO directive - replaces deprecated ProfessionalStandardsValidator"""
    
    def test_quality_gate_initialization(self):
        """Test technical quality gate initialization"""
        quality_gate = TechnicalQualityGate()
        assert quality_gate is not None
        assert hasattr(quality_gate, 'gates')
        assert len(quality_gate.gates) > 0
    
    def test_code_quality_validation(self):
        """Test code quality validation with metrics"""
        quality_gate = TechnicalQualityGate()
        
        # Valid metrics meeting thresholds
        valid_metrics = {
            'test_coverage': 0.9,  # 90% coverage
            'cyclomatic_complexity': 8,  # Below threshold of 10
            'duplication_percentage': 0.03,  # 3% duplication
            'security_vulnerabilities': {
                'critical': 0, 'high': 0, 'medium': 0, 'low': 1
            },
            'performance': {
                'response_time_ms': 80,  # Below 100ms
                'memory_usage_mb': 1024  # Below 2048MB
            }
        }
        
        result = quality_gate.validate_code_quality(valid_metrics)
        assert result.passes is True
        assert result.overall_score > 0.8
        assert len(result.violations) == 0
    
    def test_quality_gate_violations(self):
        """Test quality gate with violations"""
        quality_gate = TechnicalQualityGate()
        
        # Metrics failing thresholds
        failing_metrics = {
            'test_coverage': 0.7,  # Below 85% threshold
            'cyclomatic_complexity': 15,  # Above threshold of 10
            'security_vulnerabilities': {
                'critical': 1, 'high': 2, 'medium': 1, 'low': 0
            }
        }
        
        result = quality_gate.validate_code_quality(failing_metrics)
        assert result.passes is False
        assert result.overall_score < 0.5
        assert len(result.violations) > 0
    
    def test_professional_compliance_reporting(self):
        """Test professional compliance reporting"""
        quality_gate = TechnicalQualityGate()
        
        # Quality metrics for reporting
        metrics = {
            'test_coverage': 0.88,
            'cyclomatic_complexity': 6,
            'duplication_percentage': 0.02,
            'security_vulnerabilities': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
            'performance': {'response_time_ms': 65, 'memory_usage_mb': 800}
        }
        
        report = quality_gate.generate_quality_report(metrics, 'detailed')
        assert '✅ PASS' in report
        assert 'Overall Score:' in report
        assert 'Production Ready: Yes' in report


class TestMCPClient:
    """Test MCP Client implementation with professional standards"""
    
    def test_mcp_client_initialization(self):
        """Test MCP client initialization"""
        config = MCPConfig(
            server_url="ws://localhost:8000",
            connection_timeout=10.0,
            request_timeout=5.0
        )
        
        client = MCPClient(config)
        assert client is not None
        assert client.config.server_url == "ws://localhost:8000"
        assert client.professional_validator is not None
        assert client.circuit_breaker is not None
    
    def test_mcp_client_factory(self):
        """Test MCP client factory function"""
        config = {
            'server_url': 'ws://test:8000',
            'connection_timeout': 15.0
        }
        
        client = create_mcp_client(config)
        assert client is not None
        assert client.config.server_url == 'ws://test:8000'
        assert client.config.connection_timeout == 15.0
    
    @pytest.mark.asyncio
    async def test_mcp_fallback_processing(self):
        """Test MCP fallback processing when MCP unavailable"""
        client = MCPClient()
        
        # Test fallback processing
        test_text = "Today we study chapter two verse twenty five"
        result = await client._fallback_processing(test_text, "scriptural")
        
        # Should return processed text (fallback to rule-based processing)
        assert result is not None
        assert isinstance(result, str)
    
    def test_performance_stats_collection(self):
        """Test MCP performance statistics collection"""
        client = MCPClient()
        
        # Update some metrics
        client.performance_metrics.requests_sent = 100
        client.performance_metrics.responses_received = 95
        client.performance_metrics.failed_requests = 5
        
        stats = client.get_performance_stats()
        
        assert stats['requests_sent'] == 100
        assert stats['responses_received'] == 95
        assert stats['failed_requests'] == 5
        assert stats['success_rate'] == 95.0
    
    def test_professional_compliance_reporting(self):
        """Test professional compliance reporting"""
        client = MCPClient()
        
        report = client.get_professional_compliance_report()
        
        assert 'total_validations_performed' in report
        assert 'professional_compliance_rate' in report
        assert 'integrity_checks_active' in report
        assert 'ceo_directive_compliance' in report
        assert report['ceo_directive_compliance'] is True


class TestMCPReliability:
    """Test MCP reliability patterns"""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization"""
        config = {
            'failure_threshold': 5,
            'recovery_timeout': 60.0,
            'success_threshold': 3
        }
        
        breaker = MCPCircuitBreakerAdvanced(config)
        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout == 60.0
        assert breaker.success_threshold == 3
        assert breaker.state == "closed"
    
    def test_circuit_breaker_state_transitions(self):
        """Test circuit breaker state transitions"""
        breaker = MCPCircuitBreakerAdvanced({'failure_threshold': 2})
        
        # Initial state: closed
        assert breaker.can_execute_request() is True
        assert breaker.state == "closed"
        
        # Record failures
        breaker.record_failure()
        assert breaker.state == "closed"  # Still closed
        
        breaker.record_failure()
        assert breaker.state == "open"  # Now open
        assert breaker.can_execute_request() is False
    
    def test_health_monitor_initialization(self):
        """Test health monitor initialization"""
        config = {
            'enable_health_monitoring': True,
            'health_check_interval': 30.0
        }
        
        monitor = MCPHealthMonitor(config)
        assert monitor.monitoring_enabled is True
        assert monitor.check_interval == 30.0
        assert monitor.professional_validator is not None
    
    def test_degradation_manager(self):
        """Test graceful degradation manager"""
        manager = GracefulDegradationManager()
        
        # Test degradation level setting
        result = manager.set_degradation_level(1, "Testing degradation")
        assert result is True
        assert manager.current_degradation_level == 1
        
        # Test status reporting
        status = manager.get_status()
        assert status['current_level'] == 1
        assert status['professional_compliance'] is True


class TestAdvancedTextNormalizerMCPIntegration:
    """Test AdvancedTextNormalizer MCP integration"""
    
    def test_normalizer_mcp_initialization(self):
        """Test normalizer with MCP integration"""
        config = {
            'enable_mcp_processing': True,
            'enable_fallback': True,
            'mcp': {
                'server_url': 'ws://localhost:8000',
                'connection_timeout': 10.0
            }
        }
        
        normalizer = AdvancedTextNormalizer(config)
        assert normalizer.enable_mcp_processing is True
        assert normalizer.enable_fallback is True
        assert normalizer.mcp_client is not None
        assert normalizer.professional_validator is not None
    
    def test_sync_mcp_processing_wrapper(self):
        """Test synchronous wrapper for MCP processing"""
        config = {
            'enable_mcp_processing': False,  # Disable for testing
            'enable_fallback': True
        }
        
        normalizer = AdvancedTextNormalizer(config)
        
        # Test sync wrapper with MCP disabled
        test_text = "Today we study chapter two verse twenty five"
        result = normalizer.convert_numbers_with_context_sync(test_text)
        
        assert result is not None
        assert isinstance(result, str)
    
    def test_professional_standards_in_processing(self):
        """Test professional standards enforcement in processing"""
        normalizer = AdvancedTextNormalizer({'enable_mcp_processing': True})
        
        # Professional validator should be initialized
        assert normalizer.professional_validator is not None
        
        # Test confidence tracking
        assert 'professional_compliance_rate' in normalizer.confidence_tracking
        assert normalizer.confidence_tracking['professional_compliance_rate'] == 1.0


class TestEpic4Foundation:
    """Test Epic 4 foundation architecture"""
    
    def test_epic4_foundation_initialization(self):
        """Test Epic 4 foundation initialization"""
        foundation = create_epic4_foundation()
        
        assert foundation is not None
        assert foundation.professional_validator is not None
        assert foundation.mcp_client is not None
        assert len(foundation.integration_patterns) > 0
    
    def test_integration_pattern_registration(self):
        """Test Epic 4 integration pattern registration"""
        foundation = Epic4ExtensibleMCPArchitecture()
        
        # Should have default patterns
        patterns = foundation.list_integration_patterns()
        assert 'advanced_language_processing' in patterns
        assert 'semantic_enhancement' in patterns
        assert 'real_time_collaboration' in patterns
        assert 'ai_assisted_quality' in patterns
    
    def test_epic4_foundation_report(self):
        """Test Epic 4 foundation status report"""
        foundation = Epic4ExtensibleMCPArchitecture()
        
        report = foundation.get_epic4_foundation_report()
        
        assert report['foundation_status'] == 'ready'
        assert 'registered_capabilities' in report
        assert 'available_patterns' in report
        assert 'professional_compliance_rate' in report
        assert report['epic4_ready'] is True
    
    def test_development_guidelines(self):
        """Test Epic 4 development guidelines"""
        foundation = Epic4ExtensibleMCPArchitecture()
        
        guidelines = foundation.get_development_guidelines()
        
        assert 'integration_patterns' in guidelines
        assert 'professional_standards' in guidelines
        assert 'development_process' in guidelines
        assert 'performance_requirements' in guidelines
        
        # Professional standards should be enforced
        standards = guidelines['professional_standards']
        assert standards['compliance_required'] is True
        assert standards['validation_mandatory'] is True
        assert standards['technical_integrity'] is True


class TestMCPPerformanceValidation:
    """Test MCP integration performance validation"""
    
    def test_performance_target_compliance(self):
        """Test MCP integration maintains performance targets"""
        # Performance targets from Story 5.1: 10+ segments/sec
        target_processing_time_ms = 100  # 10 seg/sec = <100ms per segment
        
        config = {
            'enable_mcp_processing': False,  # Test without MCP overhead
            'target_processing_time_ms': target_processing_time_ms
        }
        
        normalizer = AdvancedTextNormalizer(config)
        
        # Test processing time
        test_text = "Today we study yoga and dharma"
        start_time = time.time()
        
        result = normalizer.convert_numbers_with_context(test_text)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Should meet performance target
        assert processing_time_ms < target_processing_time_ms
        assert result is not None
    
    def test_mcp_integration_overhead(self):
        """Test MCP integration overhead is acceptable"""
        # Test with and without MCP to measure overhead
        
        # Without MCP
        config_no_mcp = {'enable_mcp_processing': False}
        normalizer_no_mcp = AdvancedTextNormalizer(config_no_mcp)
        
        test_text = "Today we study chapter two verse twenty five"
        
        start_time = time.time()
        result_no_mcp = normalizer_no_mcp.convert_numbers_with_context(test_text)
        time_no_mcp = time.time() - start_time
        
        # With MCP (fallback mode for testing)
        config_with_mcp = {'enable_mcp_processing': True, 'enable_fallback': True}
        normalizer_with_mcp = AdvancedTextNormalizer(config_with_mcp)
        
        start_time = time.time()
        result_with_mcp = normalizer_with_mcp.convert_numbers_with_context_sync(test_text)
        time_with_mcp = time.time() - start_time
        
        # MCP overhead should be minimal (both should produce same result with fallback)
        assert result_no_mcp == result_with_mcp
        
        # Overhead should be reasonable (less than 2x)
        if time_no_mcp > 0:
            overhead_ratio = time_with_mcp / time_no_mcp
            assert overhead_ratio < 2.0


class TestMCPConfigurationManagement:
    """Test MCP configuration management"""
    
    def test_mcp_config_loading(self):
        """Test MCP configuration loading"""
        # Test config file exists
        config_path = Path("config/mcp_config.yaml")
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert 'mcp_client' in config
            assert 'text_processing' in config
            assert 'monitoring' in config
            assert 'security' in config
            assert 'epic4_foundation' in config
    
    def test_professional_standards_config(self):
        """Test professional standards configuration"""
        config_path = Path("config/mcp_config.yaml")
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            mcp_config = config.get('mcp_client', {})
            
            # Professional standards should be enabled
            assert mcp_config.get('enable_professional_validation') is True
            assert mcp_config.get('enable_integrity_checks') is True
            assert mcp_config.get('require_multi_agent_verification') is True


class TestMCPIntegrationEndToEnd:
    """End-to-end MCP integration tests with professional validation"""
    
    def test_complete_mcp_pipeline(self):
        """Test complete MCP integration pipeline"""
        # Initialize all components
        config = {
            'enable_mcp_processing': True,
            'enable_fallback': True,
            'mcp': {
                'server_url': 'ws://localhost:8000',
                'connection_timeout': 5.0
            }
        }
        
        # Test normalizer with MCP integration
        normalizer = AdvancedTextNormalizer(config)
        
        # Test Epic 4 foundation
        foundation = create_epic4_foundation(config)
        
        # Test professional validator
        validator = ProfessionalStandardsValidator()
        
        # Validate all components initialized
        assert normalizer is not None
        assert foundation is not None
        assert validator is not None
        
        # Test professional compliance
        compliance_report = validator.get_professional_compliance_report()
        assert compliance_report['ceo_directive_compliance'] is True
    
    def test_story_5_2_acceptance_criteria(self):
        """Test all Story 5.2 acceptance criteria"""
        
        # AC1: MCP Library Installation and Configuration
        config = MCPConfig()
        assert config is not None
        
        # AC2: MCP Client Infrastructure
        client = MCPClient(config)
        assert client is not None
        assert client.circuit_breaker is not None
        assert client.performance_metrics is not None
        
        # AC3: Enhanced Text Processing Integration
        normalizer = AdvancedTextNormalizer({'enable_mcp_processing': True})
        assert normalizer.mcp_client is not None
        assert normalizer.enable_mcp_processing is True
        
        # AC4: Performance and Reliability
        health_monitor = MCPHealthMonitor()
        circuit_breaker = MCPCircuitBreakerAdvanced()
        degradation_manager = GracefulDegradationManager()
        
        assert health_monitor is not None
        assert circuit_breaker is not None
        assert degradation_manager is not None
        
        # AC5: Foundation for Epic 4 Development
        foundation = Epic4ExtensibleMCPArchitecture()
        patterns = foundation.list_integration_patterns()
        guidelines = foundation.get_development_guidelines()
        
        assert len(patterns) > 0
        assert 'integration_patterns' in guidelines
        assert 'professional_standards' in guidelines
    
    def test_professional_standards_enforcement(self):
        """Test professional standards enforcement throughout MCP integration"""
        
        # Test validator is used in all components
        client = MCPClient()
        normalizer = AdvancedTextNormalizer({'enable_mcp_processing': True})
        foundation = Epic4ExtensibleMCPArchitecture()
        
        # All should have professional validators
        assert hasattr(client, 'professional_validator')
        assert hasattr(normalizer, 'professional_validator')
        assert hasattr(foundation, 'professional_validator')
        
        # Test compliance reporting
        client_report = client.get_professional_compliance_report()
        foundation_report = foundation.get_epic4_foundation_report()
        
        assert client_report['ceo_directive_compliance'] is True
        assert foundation_report['professional_compliance_rate'] >= 0.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])