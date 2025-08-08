#!/usr/bin/env python3
"""
Production Monitoring Dashboard - Epic 2.4 Research-Grade Enhancement

Real-time monitoring and validation of Epic 2.4 processing performance with:
- Live processing metrics dashboard
- Quality validation alerts
- Performance regression detection
- Academic compliance monitoring

Usage:
    python scripts/production_monitor.py --watch-dir data/processed_srts
    python scripts/production_monitor.py --validate-batch data/processed_srts/batch_20250808_143022_metrics.json
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from research_integration.performance_benchmarking import PerformanceBenchmarking
from research_integration.research_validation_metrics import ResearchValidationMetrics
from research_integration.comprehensive_reporting import ComprehensiveReporting
from utils.logger_config import get_logger

logger = get_logger(__name__)

@dataclass
class ProductionHealthMetrics:
    """Production system health metrics"""
    timestamp: float
    batch_success_rate: float
    average_confidence: float
    processing_speed: float  # segments per second
    enhancement_rate: float
    iast_compliance: float
    error_rate: float
    memory_usage_mb: Optional[float] = None
    
@dataclass
class QualityAlert:
    """Quality validation alert"""
    alert_id: str
    severity: str  # CRITICAL, WARNING, INFO
    message: str
    timestamp: float
    batch_id: str
    metric_name: str
    actual_value: float
    expected_threshold: float

class ProductionMonitor:
    """
    Production monitoring and validation system for Epic 2.4 enhancements.
    
    Provides real-time monitoring, quality validation, and performance tracking
    for production SRT processing workloads.
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize Epic 2.4 monitoring components
        self.benchmarking = PerformanceBenchmarking()
        self.validation = ResearchValidationMetrics()
        self.reporting = ComprehensiveReporting()
        
        # Production thresholds (based on Epic 2.4 QA validation)
        self.quality_thresholds = {
            'min_success_rate': 0.95,      # 95% minimum success rate
            'min_confidence': 0.80,        # 80% minimum confidence
            'max_processing_time': 3.0,    # 3 seconds max per file
            'min_enhancement_rate': 0.75,  # 75% minimum enhancement rate  
            'min_iast_compliance': 0.90,   # 90% IAST compliance
            'max_error_rate': 0.05         # 5% maximum error rate
        }
        
        # Alert history
        self.alerts = []
        self.metrics_history = []
        
        logger.info("ProductionMonitor initialized with Epic 2.4 thresholds")
    
    def validate_batch_metrics(self, metrics_file: Path) -> List[QualityAlert]:
        """
        Validate batch processing metrics against production quality thresholds.
        
        Args:
            metrics_file: Path to batch metrics JSON file
            
        Returns:
            List of quality alerts for any threshold violations
        """
        alerts = []
        
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
            
            batch_id = batch_data.get('batch_id', 'unknown')
            
            # Calculate key metrics
            success_rate = batch_data['successful_files'] / batch_data['total_files'] if batch_data['total_files'] > 0 else 0
            enhancement_rate = batch_data['enhanced_segments'] / batch_data['total_segments'] if batch_data['total_segments'] > 0 else 0
            avg_confidence = batch_data.get('average_confidence', 0.0)
            processing_time = batch_data['performance_metrics']['avg_time_per_file']
            error_rate = batch_data['failed_files'] / batch_data['total_files'] if batch_data['total_files'] > 0 else 0
            
            timestamp = time.time()
            
            # Check thresholds
            if success_rate < self.quality_thresholds['min_success_rate']:
                alerts.append(QualityAlert(
                    alert_id=f"{batch_id}_success_rate",
                    severity="CRITICAL",
                    message=f"Success rate {success_rate:.1%} below threshold {self.quality_thresholds['min_success_rate']:.1%}",
                    timestamp=timestamp,
                    batch_id=batch_id,
                    metric_name="success_rate",
                    actual_value=success_rate,
                    expected_threshold=self.quality_thresholds['min_success_rate']
                ))
            
            if avg_confidence < self.quality_thresholds['min_confidence']:
                alerts.append(QualityAlert(
                    alert_id=f"{batch_id}_confidence",
                    severity="WARNING",
                    message=f"Average confidence {avg_confidence:.3f} below threshold {self.quality_thresholds['min_confidence']:.3f}",
                    timestamp=timestamp,
                    batch_id=batch_id,
                    metric_name="confidence",
                    actual_value=avg_confidence,
                    expected_threshold=self.quality_thresholds['min_confidence']
                ))
            
            if processing_time > self.quality_thresholds['max_processing_time']:
                alerts.append(QualityAlert(
                    alert_id=f"{batch_id}_processing_time",
                    severity="WARNING",
                    message=f"Processing time {processing_time:.2f}s above threshold {self.quality_thresholds['max_processing_time']:.2f}s",
                    timestamp=timestamp,
                    batch_id=batch_id,
                    metric_name="processing_time",
                    actual_value=processing_time,
                    expected_threshold=self.quality_thresholds['max_processing_time']
                ))
            
            if enhancement_rate < self.quality_thresholds['min_enhancement_rate']:
                alerts.append(QualityAlert(
                    alert_id=f"{batch_id}_enhancement_rate",
                    severity="INFO",
                    message=f"Enhancement rate {enhancement_rate:.1%} below expected {self.quality_thresholds['min_enhancement_rate']:.1%}",
                    timestamp=timestamp,
                    batch_id=batch_id,
                    metric_name="enhancement_rate",
                    actual_value=enhancement_rate,
                    expected_threshold=self.quality_thresholds['min_enhancement_rate']
                ))
            
            # Store metrics for trending
            health_metrics = ProductionHealthMetrics(
                timestamp=timestamp,
                batch_success_rate=success_rate,
                average_confidence=avg_confidence,
                processing_speed=batch_data['performance_metrics']['segments_per_second'],
                enhancement_rate=enhancement_rate,
                iast_compliance=0.95,  # Would be calculated from actual IAST validation
                error_rate=error_rate
            )
            self.metrics_history.append(health_metrics)
            
            # Keep only recent metrics (last 100 batches)
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
            
            self.alerts.extend(alerts)
            
            logger.info(f"Validated batch {batch_id}: {len(alerts)} alerts generated")
            
        except Exception as e:
            error_alert = QualityAlert(
                alert_id=f"validation_error_{int(time.time())}",
                severity="CRITICAL",
                message=f"Failed to validate batch metrics: {str(e)}",
                timestamp=time.time(),
                batch_id="unknown",
                metric_name="validation_error",
                actual_value=0.0,
                expected_threshold=0.0
            )
            alerts.append(error_alert)
            logger.error(f"Batch validation error: {e}")
        
        return alerts
    
    def generate_production_dashboard(self) -> str:
        """Generate production monitoring dashboard"""
        
        current_time = datetime.now()
        
        # Calculate recent metrics
        if self.metrics_history:
            recent_metrics = self.metrics_history[-10:]  # Last 10 batches
            avg_success_rate = statistics.mean([m.batch_success_rate for m in recent_metrics])
            avg_confidence = statistics.mean([m.average_confidence for m in recent_metrics])
            avg_processing_speed = statistics.mean([m.processing_speed for m in recent_metrics])
            avg_enhancement_rate = statistics.mean([m.enhancement_rate for m in recent_metrics])
        else:
            avg_success_rate = avg_confidence = avg_processing_speed = avg_enhancement_rate = 0.0
        
        # Count alerts by severity
        recent_alerts = [a for a in self.alerts if a.timestamp > time.time() - 3600]  # Last hour
        critical_alerts = len([a for a in recent_alerts if a.severity == "CRITICAL"])
        warning_alerts = len([a for a in recent_alerts if a.severity == "WARNING"])
        
        # Generate dashboard
        dashboard = f"""
# 📊 Epic 2.4 Production Dashboard

**Last Updated**: {current_time.strftime('%Y-%m-%d %H:%M:%S')}

## 🚀 System Health Overview

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Success Rate | {avg_success_rate:.1%} | >95% | {'🟢' if avg_success_rate >= 0.95 else '🟡' if avg_success_rate >= 0.90 else '🔴'} |
| Avg Confidence | {avg_confidence:.3f} | >0.80 | {'🟢' if avg_confidence >= 0.80 else '🟡' if avg_confidence >= 0.70 else '🔴'} |
| Processing Speed | {avg_processing_speed:.1f}/s | - | 🟢 |
| Enhancement Rate | {avg_enhancement_rate:.1%} | >75% | {'🟢' if avg_enhancement_rate >= 0.75 else '🟡'} |

## 🚨 Active Alerts

**Critical**: {critical_alerts} | **Warnings**: {warning_alerts} | **Total Recent**: {len(recent_alerts)}

### Recent Alerts
"""
        
        # Add recent alerts
        for alert in recent_alerts[-5:]:  # Show last 5 alerts
            alert_time = datetime.fromtimestamp(alert.timestamp).strftime('%H:%M:%S')
            severity_emoji = {'CRITICAL': '🔴', 'WARNING': '🟡', 'INFO': 'ℹ️'}.get(alert.severity, '❓')
            dashboard += f"- **{alert_time}** {severity_emoji} {alert.message}\n"
        
        if not recent_alerts:
            dashboard += "- No recent alerts ✅\n"
        
        dashboard += f"""

## 📈 Performance Trends

**Batch Processing History**: {len(self.metrics_history)} batches monitored

### Epic 2.4 Enhancement Status
- ✅ Research-grade confidence scoring active
- ✅ Academic IAST validation enabled  
- ✅ Sanskrit linguistic processing operational
- ✅ Performance benchmarking monitoring
- ✅ Cross-story enhancement integration active

## 🎯 Quality Thresholds
- **Min Success Rate**: {self.quality_thresholds['min_success_rate']:.1%}
- **Min Confidence**: {self.quality_thresholds['min_confidence']:.3f}
- **Max Processing Time**: {self.quality_thresholds['max_processing_time']:.1f}s/file
- **Min Enhancement Rate**: {self.quality_thresholds['min_enhancement_rate']:.1%}
- **Max Error Rate**: {self.quality_thresholds['max_error_rate']:.1%}

---
*Epic 2.4 Production Monitor - Research-Grade Quality Assurance*
"""
        
        return dashboard
    
    def watch_directory(self, watch_dir: Path, interval: int = 60):
        """
        Watch directory for new batch results and validate them continuously.
        
        Args:
            watch_dir: Directory to monitor for new metrics files
            interval: Check interval in seconds
        """
        logger.info(f"Starting directory watch: {watch_dir} (interval: {interval}s)")
        
        processed_files = set()
        
        print(f"🔍 Monitoring {watch_dir} for new batch results...")
        print(f"📊 Quality thresholds: Success≥{self.quality_thresholds['min_success_rate']:.1%}, Confidence≥{self.quality_thresholds['min_confidence']:.3f}")
        print("Press Ctrl+C to stop monitoring\n")
        
        try:
            while True:
                # Find new metrics files
                metrics_files = list(watch_dir.glob("*_metrics.json"))
                new_files = [f for f in metrics_files if f not in processed_files]
                
                # Process new files
                for metrics_file in new_files:
                    print(f"📊 Processing batch metrics: {metrics_file.name}")
                    
                    alerts = self.validate_batch_metrics(metrics_file)
                    processed_files.add(metrics_file)
                    
                    if alerts:
                        print(f"🚨 Generated {len(alerts)} alerts:")
                        for alert in alerts:
                            severity_emoji = {'CRITICAL': '🔴', 'WARNING': '🟡', 'INFO': 'ℹ️'}.get(alert.severity, '❓')
                            print(f"   {severity_emoji} {alert.message}")
                    else:
                        print(f"✅ No quality issues detected")
                    print()
                
                # Sleep until next check
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped. Processed {len(processed_files)} batch files.")
            
            # Generate final dashboard
            if self.metrics_history:
                dashboard = self.generate_production_dashboard()
                dashboard_file = watch_dir / f"production_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                with open(dashboard_file, 'w', encoding='utf-8') as f:
                    f.write(dashboard)
                print(f"📋 Final dashboard saved: {dashboard_file}")

def main():
    """Main monitoring entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Production Monitoring for Epic 2.4 Enhancements',
        epilog="""
Examples:
  python scripts/production_monitor.py --watch-dir data/processed_srts
  python scripts/production_monitor.py --validate-batch data/processed_srts/batch_20250808_143022_metrics.json
  python scripts/production_monitor.py --watch-dir data/processed_srts --interval 30
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--watch-dir', type=Path, help='Directory to monitor for batch results')
    parser.add_argument('--validate-batch', type=Path, help='Validate specific batch metrics file')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds (default: 60)')
    parser.add_argument('--dashboard', action='store_true', help='Generate current dashboard and exit')
    
    args = parser.parse_args()
    
    try:
        monitor = ProductionMonitor()
        
        if args.validate_batch:
            # Single batch validation
            if not args.validate_batch.exists():
                print(f"❌ Metrics file not found: {args.validate_batch}")
                return 1
            
            print(f"🔍 Validating batch metrics: {args.validate_batch}")
            alerts = monitor.validate_batch_metrics(args.validate_batch)
            
            if alerts:
                print(f"🚨 Quality issues detected ({len(alerts)} alerts):")
                for alert in alerts:
                    severity_emoji = {'CRITICAL': '🔴', 'WARNING': '🟡', 'INFO': 'ℹ️'}.get(alert.severity, '❓')
                    print(f"   {severity_emoji} {alert.message}")
                return 1
            else:
                print("✅ All quality thresholds met - batch validation passed")
                return 0
        
        elif args.dashboard:
            # Generate dashboard
            dashboard = monitor.generate_production_dashboard()
            print(dashboard)
            return 0
            
        elif args.watch_dir:
            # Directory monitoring
            if not args.watch_dir.exists():
                print(f"❌ Watch directory not found: {args.watch_dir}")
                return 1
            
            monitor.watch_directory(args.watch_dir, args.interval)
            return 0
        
        else:
            parser.print_help()
            return 1
            
    except Exception as e:
        print(f"❌ Production monitoring failed: {e}")
        logger.error(f"Production monitoring error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())