"""
Custom Dashboards Integration for Production Observability
Provides integration with Grafana, Prometheus, and custom dashboard systems
"""

import json
import logging
import requests
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class DashboardType(Enum):
    """Types of dashboard integrations"""
    GRAFANA = "grafana"
    PROMETHEUS = "prometheus"
    CUSTOM = "custom"
    ELASTICSEARCH = "elasticsearch"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    id: str
    title: str
    type: str  # graph, stat, table, etc.
    query: str
    position: Dict[str, int] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dashboard:
    """Dashboard configuration"""
    id: str
    title: str
    description: str
    tags: List[str] = field(default_factory=list)
    widgets: List[DashboardWidget] = field(default_factory=list)
    refresh_interval: str = "5m"
    time_range: str = "1h"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    query: str
    condition: str
    threshold: float
    severity: AlertSeverity
    duration: str = "5m"
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


class PrometheusIntegration:
    """Integration with Prometheus monitoring system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pushgateway_url = config.get('pushgateway_url')
        self.prometheus_url = config.get('prometheus_url')
        self.job_name = config.get('job_name', 'asr-processor')
        self.instance = config.get('instance', 'localhost:8080')
        
    def push_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Push metrics to Prometheus Pushgateway"""
        if not self.pushgateway_url:
            logger.warning("Pushgateway URL not configured")
            return False
        
        try:
            # Convert metrics to Prometheus format
            prometheus_metrics = self._format_for_prometheus(metrics)
            
            # Push to Pushgateway
            url = f"{self.pushgateway_url}/metrics/job/{self.job_name}/instance/{self.instance}"
            response = requests.post(
                url,
                data=prometheus_metrics,
                headers={'Content-Type': 'text/plain; version=0.0.4'},
                timeout=10
            )
            response.raise_for_status()
            
            logger.debug("Successfully pushed metrics to Prometheus")
            return True
            
        except Exception as e:
            logger.error(f"Failed to push metrics to Prometheus: {e}")
            return False
    
    def query_metrics(self, query: str, time_range: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Query metrics from Prometheus"""
        if not self.prometheus_url:
            logger.warning("Prometheus URL not configured")
            return None
        
        try:
            params = {'query': query}
            if time_range:
                params['time'] = time_range
            
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to query Prometheus: {e}")
            return None
    
    def _format_for_prometheus(self, metrics: Dict[str, Any]) -> str:
        """Format metrics in Prometheus exposition format"""
        lines = []
        
        for name, data in metrics.get('metrics', {}).items():
            value = data.get('current_value')
            if value is not None:
                # Sanitize metric name
                prom_name = name.replace('-', '_').replace('.', '_').lower()
                
                # Add help and type
                lines.append(f"# HELP {prom_name} {data.get('description', 'No description')}")
                lines.append(f"# TYPE {prom_name} gauge")
                
                # Add metric with labels
                labels = [
                    f'job="{self.job_name}"',
                    f'instance="{self.instance}"'
                ]
                label_str = '{' + ','.join(labels) + '}'
                lines.append(f"{prom_name}{label_str} {value}")
        
        return '\n'.join(lines) + '\n'


class GrafanaIntegration:
    """Integration with Grafana dashboard system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.grafana_url = config.get('grafana_url', '').rstrip('/')
        self.api_key = config.get('api_key')
        self.org_id = config.get('org_id', 1)
        
        # Session for API calls
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })
    
    def create_dashboard(self, dashboard: Dashboard) -> bool:
        """Create or update dashboard in Grafana"""
        if not self._check_connection():
            return False
        
        try:
            dashboard_json = self._build_dashboard_json(dashboard)
            
            response = self.session.post(
                f"{self.grafana_url}/api/dashboards/db",
                json={
                    'dashboard': dashboard_json,
                    'overwrite': True,
                    'message': f'Updated dashboard {dashboard.title}'
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Dashboard '{dashboard.title}' created/updated: {result.get('url')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            return False
    
    def create_alert_rule(self, alert: AlertRule) -> bool:
        """Create alert rule in Grafana"""
        if not self._check_connection():
            return False
        
        try:
            alert_json = self._build_alert_json(alert)
            
            response = self.session.post(
                f"{self.grafana_url}/api/ruler/grafana/api/v1/rules/asr-alerts",
                json=alert_json,
                timeout=30
            )
            response.raise_for_status()
            
            logger.info(f"Alert rule '{alert.name}' created")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create alert rule: {e}")
            return False
    
    def get_dashboard_status(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Get dashboard status and metrics"""
        if not self._check_connection():
            return None
        
        try:
            response = self.session.get(
                f"{self.grafana_url}/api/dashboards/uid/{dashboard_id}",
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to get dashboard status: {e}")
            return None
    
    def _check_connection(self) -> bool:
        """Check Grafana connection"""
        if not self.grafana_url or not self.api_key:
            logger.warning("Grafana URL or API key not configured")
            return False
        
        try:
            response = self.session.get(f"{self.grafana_url}/api/health", timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Grafana connection failed: {e}")
            return False
    
    def _build_dashboard_json(self, dashboard: Dashboard) -> Dict[str, Any]:
        """Build Grafana dashboard JSON"""
        return {
            'id': None,
            'uid': dashboard.id,
            'title': dashboard.title,
            'description': dashboard.description,
            'tags': dashboard.tags,
            'timezone': 'UTC',
            'refresh': dashboard.refresh_interval,
            'time': {
                'from': f'now-{dashboard.time_range}',
                'to': 'now'
            },
            'panels': [
                self._build_panel_json(widget, i)
                for i, widget in enumerate(dashboard.widgets)
            ],
            'version': 1,
            'editable': True,
            'gnetId': None
        }
    
    def _build_panel_json(self, widget: DashboardWidget, panel_id: int) -> Dict[str, Any]:
        """Build Grafana panel JSON"""
        return {
            'id': panel_id,
            'title': widget.title,
            'type': widget.type,
            'targets': [
                {
                    'expr': widget.query,
                    'refId': 'A'
                }
            ],
            'gridPos': widget.position or {
                'h': 8,
                'w': 12,
                'x': 0,
                'y': panel_id * 9
            },
            'options': widget.options,
            'fieldConfig': {
                'defaults': {
                    'unit': 'short',
                    'min': 0
                }
            }
        }
    
    def _build_alert_json(self, alert: AlertRule) -> Dict[str, Any]:
        """Build Grafana alert rule JSON"""
        return {
            'uid': f"alert_{alert.name.lower().replace(' ', '_')}",
            'title': alert.name,
            'condition': 'A',
            'data': [
                {
                    'refId': 'A',
                    'queryType': '',
                    'model': {
                        'expr': alert.query,
                        'refId': 'A'
                    }
                }
            ],
            'intervalSeconds': 60,
            'maxDataPoints': 43200,
            'noDataState': 'NoData',
            'execErrState': 'Alerting',
            'for': alert.duration,
            'annotations': alert.annotations,
            'labels': alert.labels
        }


class CustomDashboardSystem:
    """Custom dashboard system for specialized monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dashboards: Dict[str, Dashboard] = {}
        self.dashboard_data: Dict[str, Dict[str, Any]] = {}
        self.update_interval = config.get('update_interval_seconds', 30)
        
        # Background update thread
        self.update_thread = None
        self.stop_updates = threading.Event()
        
    def start_dashboard_updates(self):
        """Start background dashboard data updates"""
        if self.update_thread and self.update_thread.is_alive():
            return
        
        self.stop_updates.clear()
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        logger.info("Started custom dashboard updates")
    
    def stop_dashboard_updates(self):
        """Stop background dashboard updates"""
        if self.update_thread:
            self.stop_updates.set()
            self.update_thread.join(timeout=10)
        
    def add_dashboard(self, dashboard: Dashboard):
        """Add a dashboard to the system"""
        self.dashboards[dashboard.id] = dashboard
        self.dashboard_data[dashboard.id] = {}
        logger.info(f"Added dashboard: {dashboard.title}")
    
    def update_dashboard_data(self, dashboard_id: str, data: Dict[str, Any]):
        """Update dashboard data"""
        if dashboard_id in self.dashboard_data:
            self.dashboard_data[dashboard_id].update({
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'data': data
            })
    
    def get_dashboard_html(self, dashboard_id: str) -> str:
        """Generate HTML for dashboard"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return "<p>Dashboard not found</p>"
        
        data = self.dashboard_data.get(dashboard_id, {})
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{dashboard.title}</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard-header {{ border-bottom: 2px solid #ddd; padding-bottom: 10px; margin-bottom: 20px; }}
                .widget {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .widget-title {{ font-weight: bold; margin-bottom: 10px; }}
                .metric-value {{ font-size: 24px; color: #007bff; }}
                .metric-unit {{ font-size: 14px; color: #666; }}
                .status-healthy {{ color: #28a745; }}
                .status-warning {{ color: #ffc107; }}
                .status-critical {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <h1>{dashboard.title}</h1>
                <p>{dashboard.description}</p>
                <p>Last Updated: {data.get('last_updated', 'Never')}</p>
            </div>
            <div class="dashboard-content">
        """
        
        # Add widgets
        for widget in dashboard.widgets:
            widget_data = data.get('data', {}).get(widget.id, {})
            html += self._generate_widget_html(widget, widget_data)
        
        html += """
            </div>
            <script>
                // Auto-refresh every 30 seconds
                setTimeout(function() {
                    location.reload();
                }, 30000);
            </script>
        </body>
        </html>
        """
        
        return html
    
    def _generate_widget_html(self, widget: DashboardWidget, data: Dict[str, Any]) -> str:
        """Generate HTML for a widget"""
        if widget.type == 'stat':
            return self._generate_stat_widget(widget, data)
        elif widget.type == 'graph':
            return self._generate_graph_widget(widget, data)
        elif widget.type == 'table':
            return self._generate_table_widget(widget, data)
        else:
            return f'<div class="widget">Unsupported widget type: {widget.type}</div>'
    
    def _generate_stat_widget(self, widget: DashboardWidget, data: Dict[str, Any]) -> str:
        """Generate stat widget HTML"""
        value = data.get('value', 'N/A')
        unit = data.get('unit', '')
        status_class = f"status-{data.get('status', 'healthy')}"
        
        return f"""
        <div class="widget">
            <div class="widget-title">{widget.title}</div>
            <div class="metric-value {status_class}">{value}</div>
            <div class="metric-unit">{unit}</div>
        </div>
        """
    
    def _generate_graph_widget(self, widget: DashboardWidget, data: Dict[str, Any]) -> str:
        """Generate graph widget HTML"""
        chart_id = f"chart_{widget.id}"
        chart_data = json.dumps(data.get('chart_data', {}))
        
        return f"""
        <div class="widget">
            <div class="widget-title">{widget.title}</div>
            <canvas id="{chart_id}" width="400" height="200"></canvas>
            <script>
                var ctx = document.getElementById('{chart_id}').getContext('2d');
                var chart = new Chart(ctx, {chart_data});
            </script>
        </div>
        """
    
    def _generate_table_widget(self, widget: DashboardWidget, data: Dict[str, Any]) -> str:
        """Generate table widget HTML"""
        rows = data.get('rows', [])
        headers = data.get('headers', [])
        
        html = f"""
        <div class="widget">
            <div class="widget-title">{widget.title}</div>
            <table border="1" cellpadding="5" cellspacing="0" style="width: 100%;">
        """
        
        if headers:
            html += "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
        
        for row in rows:
            html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        
        html += "</table></div>"
        return html
    
    def _update_loop(self):
        """Background update loop for dashboard data"""
        while not self.stop_updates.wait(self.update_interval):
            try:
                for dashboard_id in self.dashboards:
                    # This would integrate with metrics collector
                    # For now, just update timestamp
                    self.dashboard_data[dashboard_id]['last_updated'] = datetime.now(timezone.utc).isoformat()
            except Exception as e:
                logger.error(f"Error updating dashboard data: {e}")


class DashboardIntegrationManager:
    """
    Main dashboard integration manager
    Coordinates multiple dashboard systems and provides unified interface
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize integrations based on configuration
        self.prometheus = None
        self.grafana = None
        self.custom = None
        
        if 'prometheus' in config:
            self.prometheus = PrometheusIntegration(config['prometheus'])
        
        if 'grafana' in config:
            self.grafana = GrafanaIntegration(config['grafana'])
        
        if 'custom' in config:
            self.custom = CustomDashboardSystem(config['custom'])
            self.custom.start_dashboard_updates()
        
        # Standard dashboards
        self._create_standard_dashboards()
        
        logger.info("Dashboard integration manager initialized")
    
    def push_metrics_to_all(self, metrics: Dict[str, Any]):
        """Push metrics to all configured dashboard systems"""
        # Push to Prometheus
        if self.prometheus:
            self.prometheus.push_metrics(metrics)
        
        # Update custom dashboards
        if self.custom:
            for dashboard_id in self.custom.dashboards:
                dashboard_data = self._transform_metrics_for_dashboard(metrics, dashboard_id)
                self.custom.update_dashboard_data(dashboard_id, dashboard_data)
    
    def create_production_dashboard(self) -> bool:
        """Create production monitoring dashboard"""
        dashboard = Dashboard(
            id='production-overview',
            title='ASR Processor - Production Overview',
            description='Key production metrics and system health',
            tags=['production', 'overview'],
            widgets=[
                DashboardWidget(
                    id='system_health',
                    title='System Health',
                    type='stat',
                    query='up{job="asr-processor"}',
                    options={'colorMode': 'background'}
                ),
                DashboardWidget(
                    id='cpu_usage',
                    title='CPU Usage',
                    type='graph',
                    query='cpu_usage_percent{job="asr-processor"}',
                    options={'yAxis': {'max': 100}}
                ),
                DashboardWidget(
                    id='memory_usage',
                    title='Memory Usage',
                    type='graph',
                    query='memory_usage_percent{job="asr-processor"}',
                    options={'yAxis': {'max': 100}}
                ),
                DashboardWidget(
                    id='request_rate',
                    title='Request Rate',
                    type='graph',
                    query='rate(request_rate[5m])',
                    options={'yAxis': {'min': 0}}
                ),
                DashboardWidget(
                    id='error_rate',
                    title='Error Rate',
                    type='graph',
                    query='error_rate{job="asr-processor"}',
                    options={'yAxis': {'max': 10}}
                )
            ]
        )
        
        success = True
        
        # Create in Grafana if available
        if self.grafana:
            success &= self.grafana.create_dashboard(dashboard)
        
        # Add to custom system if available
        if self.custom:
            self.custom.add_dashboard(dashboard)
        
        return success
    
    def create_performance_alerts(self) -> bool:
        """Create standard performance alert rules"""
        alerts = [
            AlertRule(
                name='High CPU Usage',
                query='cpu_usage_percent > 85',
                condition='gt',
                threshold=85,
                severity=AlertSeverity.WARNING,
                duration='5m',
                annotations={'description': 'CPU usage is above 85% for 5 minutes'}
            ),
            AlertRule(
                name='Critical CPU Usage',
                query='cpu_usage_percent > 95',
                condition='gt',
                threshold=95,
                severity=AlertSeverity.CRITICAL,
                duration='2m',
                annotations={'description': 'CPU usage is above 95% for 2 minutes'}
            ),
            AlertRule(
                name='High Memory Usage',
                query='memory_usage_percent > 90',
                condition='gt',
                threshold=90,
                severity=AlertSeverity.WARNING,
                duration='5m',
                annotations={'description': 'Memory usage is above 90% for 5 minutes'}
            ),
            AlertRule(
                name='High Error Rate',
                query='error_rate > 5',
                condition='gt',
                threshold=5,
                severity=AlertSeverity.CRITICAL,
                duration='2m',
                annotations={'description': 'Error rate is above 5% for 2 minutes'}
            )
        ]
        
        success = True
        if self.grafana:
            for alert in alerts:
                success &= self.grafana.create_alert_rule(alert)
        
        return success
    
    def get_dashboard_url(self, dashboard_id: str, dashboard_type: DashboardType) -> Optional[str]:
        """Get URL for accessing dashboard"""
        if dashboard_type == DashboardType.GRAFANA and self.grafana:
            return f"{self.grafana.grafana_url}/d/{dashboard_id}"
        elif dashboard_type == DashboardType.CUSTOM and self.custom:
            return f"/dashboards/{dashboard_id}"  # Assuming custom web server
        return None
    
    def _create_standard_dashboards(self):
        """Create standard production dashboards"""
        # This would be called during initialization
        # For now, just log
        logger.info("Standard dashboards will be created on first metrics push")
    
    def _transform_metrics_for_dashboard(self, metrics: Dict[str, Any], dashboard_id: str) -> Dict[str, Any]:
        """Transform metrics for specific dashboard"""
        # Transform metrics based on dashboard requirements
        transformed = {}
        
        metrics_data = metrics.get('metrics', {})
        
        for widget_id in ['system_health', 'cpu_usage', 'memory_usage', 'request_rate', 'error_rate']:
            if widget_id in metrics_data:
                metric = metrics_data[widget_id]
                transformed[widget_id] = {
                    'value': metric.get('current_value', 0),
                    'unit': metric.get('unit', ''),
                    'status': 'healthy' if metric.get('current_value', 0) < 80 else 'warning'
                }
        
        return transformed


# Global dashboard integration
_global_dashboard_integration: Optional[DashboardIntegrationManager] = None


def initialize_dashboard_integration(config: Dict[str, Any]) -> DashboardIntegrationManager:
    """Initialize global dashboard integration"""
    global _global_dashboard_integration
    _global_dashboard_integration = DashboardIntegrationManager(config)
    return _global_dashboard_integration


def get_dashboard_integration() -> Optional[DashboardIntegrationManager]:
    """Get global dashboard integration"""
    return _global_dashboard_integration