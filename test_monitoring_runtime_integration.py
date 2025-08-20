#!/usr/bin/env python3
"""
Runtime integration tests for monitoring system fixes.

Tests:
1. Structured logger can serialize enums to JSON
2. AlertSeverity enum has all required members
3. No reserved keyword collisions in logging
4. Full monitoring stack initialization
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def test_json_serialization():
    """Test that structured logger can serialize enums to JSON"""
    print("🧪 Testing JSON serialization of enums...")
    
    try:
        from monitoring.structured_logger import (
            StructuredLogEntry, LogLevel, LogCategory, LogContext
        )
        
        # Create a log entry with enums
        entry = StructuredLogEntry(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.ERROR,
            message="Test error message",
            logger_name="test_logger",
            category=LogCategory.ERROR,
            context=LogContext(correlation_id="test-123"),
            fields={"test_field": "test_value"}
        )
        
        # Test to_dict conversion
        entry_dict = entry.to_dict()
        assert isinstance(entry_dict, dict), "to_dict should return a dictionary"
        assert entry_dict['level'] == 'ERROR', f"Expected 'ERROR', got {entry_dict['level']}"
        assert entry_dict['category'] == 'error', f"Expected 'error', got {entry_dict['category']}"
        
        # Test JSON serialization
        json_str = entry.to_json()
        assert isinstance(json_str, str), "to_json should return a string"
        
        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed['level'] == 'ERROR', "JSON should contain enum value"
        assert parsed['category'] == 'error', "JSON should contain enum value"
        
        print("  ✅ JSON serialization test PASSED")
        return True
        
    except Exception as e:
        print(f"  ❌ JSON serialization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alert_severity_enum():
    """Test that AlertSeverity enum has all required members"""
    print("🧪 Testing AlertSeverity enum members...")
    
    try:
        from monitoring.dashboard_integration import AlertSeverity
        
        # Check all required members exist
        required_members = ['INFO', 'WARNING', 'ERROR', 'CRITICAL', 'EMERGENCY']
        
        for member in required_members:
            assert hasattr(AlertSeverity, member), f"AlertSeverity missing {member}"
            # Also check we can access the value
            value = getattr(AlertSeverity, member).value
            assert isinstance(value, str), f"AlertSeverity.{member}.value should be a string"
        
        # Test specific usage from production_monitor.py
        test_severities = [AlertSeverity.CRITICAL, AlertSeverity.ERROR]
        assert all(hasattr(s, 'value') for s in test_severities), "All severities should have value"
        
        # Test dictionary key usage
        escalation_rules = {
            AlertSeverity.INFO: ["slack"],
            AlertSeverity.WARNING: ["slack", "email"],
            AlertSeverity.ERROR: ["slack", "email", "webhook"],
            AlertSeverity.CRITICAL: ["slack", "email", "webhook", "sms"]
        }
        
        assert AlertSeverity.ERROR in escalation_rules, "ERROR should be a valid dictionary key"
        
        print("  ✅ AlertSeverity enum test PASSED")
        return True
        
    except Exception as e:
        print(f"  ❌ AlertSeverity enum test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reserved_keyword_collision():
    """Test that logging doesn't have reserved keyword collisions"""
    print("🧪 Testing reserved keyword collision fix...")
    
    try:
        from monitoring.structured_logger import StructuredLogger
        
        # Create a logger with temporary log file
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as tmp:
            log_file = tmp.name
        
        config = {
            'level': 'DEBUG',
            'log_file': log_file
        }
        
        logger = StructuredLogger("test_logger", config)
        
        # Test error logging with exception (previously caused exc_info collision)
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            # This should not cause KeyError for 'exc_info'
            logger.error("Test error with exception", exception=e)
        
        # Test critical logging with exception
        try:
            raise RuntimeError("Critical test exception")
        except RuntimeError as e:
            logger.critical("Test critical with exception", exception=e)
        
        # Clean up
        try:
            os.unlink(log_file)
        except:
            pass
        
        print("  ✅ Reserved keyword collision test PASSED")
        return True
        
    except KeyError as e:
        if 'exc_info' in str(e):
            print(f"  ❌ Reserved keyword collision still exists: {e}")
            return False
        raise
    except Exception as e:
        print(f"  ❌ Reserved keyword collision test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_monitoring_stack_initialization():
    """Test full monitoring stack initialization"""
    print("🧪 Testing full monitoring stack initialization...")
    
    try:
        # Import all monitoring components
        from monitoring.structured_logger import StructuredLogger, initialize_structured_logging
        from monitoring.dashboard_integration import DashboardIntegrationManager
        from monitoring.production_monitor import ProductionMonitor
        
        # Initialize structured logging
        log_config = {
            'level': 'INFO',
            'json_log_file': '/tmp/test_monitoring.json'
        }
        logger = initialize_structured_logging(log_config)
        assert logger is not None, "Structured logger should initialize"
        
        # Initialize dashboard integration
        dashboard_config = {
            'custom': {
                'update_interval_seconds': 60
            }
        }
        dashboard_manager = DashboardIntegrationManager(dashboard_config)
        assert dashboard_manager is not None, "Dashboard manager should initialize"
        
        # Initialize production monitor
        monitor_config = {
            'metrics_interval_seconds': 30,
            'health_check_interval_seconds': 10,
            'enable_alerting': True,
            'alert_channels': {
                'slack': {'webhook_url': 'https://example.com/slack'},
                'email': {'smtp_host': 'localhost', 'from_email': 'monitor@example.com'}
            }
        }
        
        production_monitor = ProductionMonitor(monitor_config)
        assert production_monitor is not None, "Production monitor should initialize"
        
        # Test that monitor can trigger alerts without errors
        from monitoring.dashboard_integration import AlertSeverity
        production_monitor.trigger_production_alert(
            alert_id="test_alert_001",
            title="Test Alert",
            description="Test alert message",
            severity=AlertSeverity.ERROR,  # This should work now
            component="test_component",
            metric_value=100.0,
            threshold=80.0
        )
        
        print("  ✅ Monitoring stack initialization test PASSED")
        return True
        
    except AttributeError as e:
        if 'ERROR' in str(e):
            print(f"  ❌ AlertSeverity.ERROR still missing: {e}")
            return False
        raise
    except Exception as e:
        print(f"  ❌ Monitoring stack initialization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_production_monitor_specific_issues():
    """Test specific issues in production_monitor.py"""
    print("🧪 Testing production_monitor.py specific fixes...")
    
    try:
        from monitoring.production_monitor import ProductionMonitor
        from monitoring.dashboard_integration import AlertSeverity
        
        config = {
            'enable_alerting': True,
            'alert_channels': {}
        }
        
        monitor = ProductionMonitor(config)
        
        # Test that AlertSeverity.LOW was replaced with INFO
        # The line that had AlertSeverity.LOW should now use AlertSeverity.INFO
        monitor.trigger_production_alert(
            alert_id="test_low_priority_001",
            title="Low Priority Alert",
            description="This used to use AlertSeverity.LOW",
            severity=AlertSeverity.INFO,  # Should work as replacement for LOW
            component="test",
            metric_value=50.0,
            threshold=60.0
        )
        
        # Test escalation rules with ERROR
        escalation = monitor.escalation_rules
        assert AlertSeverity.ERROR in escalation, "ERROR should be in escalation rules"
        assert AlertSeverity.CRITICAL in escalation, "CRITICAL should be in escalation rules"
        
        # Test severity filtering
        critical_alerts = [AlertSeverity.CRITICAL, AlertSeverity.ERROR]
        for severity in critical_alerts:
            assert hasattr(severity, 'value'), f"{severity} should have value attribute"
        
        print("  ✅ Production monitor specific fixes test PASSED")
        return True
        
    except Exception as e:
        print(f"  ❌ Production monitor specific fixes test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("MONITORING SYSTEM RUNTIME INTEGRATION TESTS")
    print("=" * 60)
    print()
    
    # Change to project root
    os.chdir(Path(__file__).parent)
    
    # Suppress verbose logging during tests
    logging.getLogger().setLevel(logging.ERROR)
    
    # Run tests
    results = {
        "JSON Serialization": test_json_serialization(),
        "AlertSeverity Enum": test_alert_severity_enum(),
        "Reserved Keyword Collision": test_reserved_keyword_collision(),
        "Stack Initialization": test_monitoring_stack_initialization(),
        "Production Monitor Fixes": test_production_monitor_specific_issues()
    }
    
    print()
    print("=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30} {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED! The monitoring system runtime issues are FIXED!")
        print()
        print("✅ JSON serialization of enums works correctly")
        print("✅ AlertSeverity enum has all required members including ERROR")
        print("✅ No reserved keyword collisions in logging")
        print("✅ Full monitoring stack initializes without errors")
        print()
        print("The system is now ready for production deployment.")
    else:
        print("❌ SOME TESTS FAILED. Please review the errors above.")
        print("The monitoring system still has runtime issues that need to be addressed.")
        sys.exit(1)


if __name__ == "__main__":
    main()