"""
Quality Dashboard and Reporting System for Story 5.5: Testing & Quality Assurance Framework

This module provides real-time quality dashboards with automated quality reporting
and trend analysis, quality gates and failure notifications, quality improvement
tracking and recommendations, and quality compliance and audit reporting.

Author: James the Developer
Date: August 20, 2025
Story: 5.5 Testing & Quality Assurance Framework
Task: 6 - Quality Metrics and Monitoring (AC5)
"""

import os
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import statistics
import html
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse


class DashboardTheme(Enum):
    """Dashboard visual themes."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


@dataclass
class DashboardConfig:
    """Configuration for quality dashboard."""
    port: int = 8080
    host: str = "localhost"
    theme: DashboardTheme = DashboardTheme.LIGHT
    auto_refresh_seconds: int = 30
    max_history_days: int = 30
    enable_alerts: bool = True
    enable_trends: bool = True


class QualityDashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for quality dashboard."""
    
    def __init__(self, *args, dashboard_generator=None, **kwargs):
        """Initialize handler with dashboard generator."""
        self.dashboard_generator = dashboard_generator
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        try:
            if self.path == "/" or self.path == "/dashboard":
                self._serve_dashboard()
            elif self.path == "/api/metrics":
                self._serve_metrics_api()
            elif self.path == "/api/alerts":
                self._serve_alerts_api()
            elif self.path == "/api/trends":
                self._serve_trends_api()
            elif self.path.startswith("/static/"):
                self._serve_static_file()
            else:
                self._serve_404()
        except Exception as e:
            self.log_error(f"Error handling request: {e}")
            self._serve_500()
    
    def _serve_dashboard(self):
        """Serve the main dashboard HTML."""
        if not self.dashboard_generator:
            self._serve_500()
            return
        
        try:
            html_content = self.dashboard_generator.generate_dashboard_html()
            self._send_response(200, html_content, "text/html")
        except Exception as e:
            self.log_error(f"Error generating dashboard: {e}")
            self._serve_500()
    
    def _serve_metrics_api(self):
        """Serve metrics data as JSON API."""
        if not self.dashboard_generator:
            self._serve_500()
            return
        
        try:
            metrics_data = self.dashboard_generator.get_metrics_data()
            json_content = json.dumps(metrics_data, indent=2)
            self._send_response(200, json_content, "application/json")
        except Exception as e:
            self.log_error(f"Error serving metrics API: {e}")
            self._serve_500()
    
    def _serve_alerts_api(self):
        """Serve alerts data as JSON API."""
        if not self.dashboard_generator:
            self._serve_500()
            return
        
        try:
            alerts_data = self.dashboard_generator.get_alerts_data()
            json_content = json.dumps(alerts_data, indent=2)
            self._send_response(200, json_content, "application/json")
        except Exception as e:
            self.log_error(f"Error serving alerts API: {e}")
            self._serve_500()
    
    def _serve_trends_api(self):
        """Serve trends data as JSON API."""
        if not self.dashboard_generator:
            self._serve_500()
            return
        
        try:
            trends_data = self.dashboard_generator.get_trends_data()
            json_content = json.dumps(trends_data, indent=2)
            self._send_response(200, json_content, "application/json")
        except Exception as e:
            self.log_error(f"Error serving trends API: {e}")
            self._serve_500()
    
    def _serve_static_file(self):
        """Serve static files (CSS, JS, images)."""
        # For now, return 404 for static files
        self._serve_404()
    
    def _serve_404(self):
        """Serve 404 Not Found response."""
        content = "<html><body><h1>404 Not Found</h1></body></html>"
        self._send_response(404, content, "text/html")
    
    def _serve_500(self):
        """Serve 500 Internal Server Error response."""
        content = "<html><body><h1>500 Internal Server Error</h1></body></html>"
        self._send_response(500, content, "text/html")
    
    def _send_response(self, status_code: int, content: str, content_type: str):
        """Send HTTP response."""
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content.encode())))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content.encode())
    
    def log_message(self, format, *args):
        """Override to reduce logging noise."""
        pass  # Suppress default logging


class QualityDashboardGenerator:
    """Generates quality dashboard HTML and data."""
    
    def __init__(self, metrics_collector, config: DashboardConfig):
        """Initialize dashboard generator."""
        self.metrics_collector = metrics_collector
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_dashboard_html(self) -> str:
        """Generate complete dashboard HTML."""
        dashboard_data = self.metrics_collector.get_quality_dashboard_data()
        
        # Generate CSS styles based on theme
        css_styles = self._generate_css_styles()
        
        # Generate JavaScript for dynamic updates
        javascript = self._generate_javascript()
        
        # Generate main dashboard content
        dashboard_content = self._generate_dashboard_content(dashboard_data)
        
        # Combine into complete HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quality Dashboard - ASR Post-Processing System</title>
    <style>{css_styles}</style>
</head>
<body>
    <div class="dashboard-container">
        <header class="dashboard-header">
            <h1>🎯 Quality Dashboard</h1>
            <div class="header-info">
                <span class="last-updated">Last Updated: {dashboard_data['timestamp']}</span>
                <span class="auto-refresh">Auto-refresh: {self.config.auto_refresh_seconds}s</span>
            </div>
        </header>
        
        {dashboard_content}
    </div>
    
    <script>{javascript}</script>
</body>
</html>
"""
        return html
    
    def _generate_css_styles(self) -> str:
        """Generate CSS styles based on theme."""
        if self.config.theme == DashboardTheme.DARK:
            bg_color = "#1a1a1a"
            text_color = "#ffffff"
            card_bg = "#2d2d2d"
            border_color = "#444444"
        else:  # Light theme (default)
            bg_color = "#f5f5f5"
            text_color = "#333333"
            card_bg = "#ffffff"
            border_color = "#dddddd"
        
        return f"""
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: {bg_color};
            color: {text_color};
            line-height: 1.6;
        }}
        
        .dashboard-container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .dashboard-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding: 20px;
            background: {card_bg};
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .dashboard-header h1 {{
            font-size: 2.5em;
            font-weight: 600;
        }}
        
        .header-info {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 5px;
        }}
        
        .header-info span {{
            font-size: 0.9em;
            opacity: 0.7;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .summary-card {{
            background: {card_bg};
            padding: 20px;
            border-radius: 8px;
            border: 1px solid {border_color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .summary-card h3 {{
            font-size: 1.2em;
            margin-bottom: 10px;
            color: #666;
        }}
        
        .summary-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .summary-trend {{
            font-size: 0.9em;
            opacity: 0.8;
        }}
        
        .category-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .category-card {{
            background: {card_bg};
            padding: 20px;
            border-radius: 8px;
            border: 1px solid {border_color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .category-card h3 {{
            font-size: 1.3em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid {border_color};
        }}
        
        .metric-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid {border_color};
        }}
        
        .metric-item:last-child {{
            border-bottom: none;
        }}
        
        .metric-name {{
            font-weight: 500;
        }}
        
        .metric-value {{
            font-weight: bold;
        }}
        
        .alerts-section {{
            margin-bottom: 30px;
        }}
        
        .alerts-card {{
            background: {card_bg};
            padding: 20px;
            border-radius: 8px;
            border: 1px solid {border_color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .alert-item {{
            display: flex;
            align-items: center;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            border-left: 4px solid;
        }}
        
        .alert-critical {{
            background-color: #fee;
            border-left-color: #dc3545;
        }}
        
        .alert-error {{
            background-color: #fef5e7;
            border-left-color: #fd7e14;
        }}
        
        .alert-warning {{
            background-color: #fff3cd;
            border-left-color: #ffc107;
        }}
        
        .alert-info {{
            background-color: #e3f2fd;
            border-left-color: #2196f3;
        }}
        
        .alert-icon {{
            margin-right: 10px;
            font-size: 1.2em;
        }}
        
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        
        .status-excellent {{ background-color: #28a745; }}
        .status-good {{ background-color: #17a2b8; }}
        .status-acceptable {{ background-color: #ffc107; }}
        .status-poor {{ background-color: #fd7e14; }}
        .status-critical {{ background-color: #dc3545; }}
        
        .value-excellent {{ color: #28a745; }}
        .value-good {{ color: #17a2b8; }}
        .value-acceptable {{ color: #ffc107; }}
        .value-poor {{ color: #fd7e14; }}
        .value-critical {{ color: #dc3545; }}
        
        @media (max-width: 768px) {{
            .dashboard-header {{
                flex-direction: column;
                text-align: center;
                gap: 10px;
            }}
            
            .summary-grid {{
                grid-template-columns: 1fr;
            }}
            
            .category-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        """
    
    def _generate_javascript(self) -> str:
        """Generate JavaScript for dynamic dashboard functionality."""
        return f"""
        // Auto-refresh functionality
        let refreshInterval = {self.config.auto_refresh_seconds * 1000};
        let refreshTimer;
        
        function startAutoRefresh() {{
            refreshTimer = setInterval(function() {{
                window.location.reload();
            }}, refreshInterval);
        }}
        
        function stopAutoRefresh() {{
            if (refreshTimer) {{
                clearInterval(refreshTimer);
                refreshTimer = null;
            }}
        }}
        
        // Start auto-refresh when page loads
        window.addEventListener('load', function() {{
            if ({str(self.config.auto_refresh_seconds > 0).lower()}) {{
                startAutoRefresh();
            }}
        }});
        
        // Stop auto-refresh when page is hidden (e.g., when user switches tabs)
        document.addEventListener('visibilitychange', function() {{
            if (document.hidden) {{
                stopAutoRefresh();
            }} else {{
                if ({str(self.config.auto_refresh_seconds > 0).lower()}) {{
                    startAutoRefresh();
                }}
            }}
        }});
        
        // Format numbers with appropriate units
        function formatMetricValue(value, unit) {{
            if (typeof value !== 'number') return value;
            
            if (unit === 'percent') {{
                return value.toFixed(1) + '%';
            }} else if (unit === 'seconds') {{
                return value.toFixed(2) + 's';
            }} else if (unit === 'segments/sec') {{
                return value.toFixed(1) + ' seg/s';
            }} else if (unit === 'count') {{
                return Math.round(value).toLocaleString();
            }} else {{
                return value.toFixed(1);
            }}
        }}
        
        // Determine quality level based on value
        function getQualityLevel(value, threshold_good, threshold_acceptable) {{
            if (value >= threshold_good) return 'excellent';
            if (value >= threshold_acceptable) return 'good';
            if (value >= threshold_acceptable * 0.8) return 'acceptable';
            if (value >= threshold_acceptable * 0.6) return 'poor';
            return 'critical';
        }}
        """
    
    def _generate_dashboard_content(self, dashboard_data: Dict[str, Any]) -> str:
        """Generate main dashboard content."""
        # Summary section
        summary_html = self._generate_summary_section(dashboard_data)
        
        # Categories section
        categories_html = self._generate_categories_section(dashboard_data)
        
        # Alerts section
        alerts_html = self._generate_alerts_section(dashboard_data)
        
        return f"""
        {summary_html}
        {categories_html}
        {alerts_html}
        """
    
    def _generate_summary_section(self, dashboard_data: Dict[str, Any]) -> str:
        """Generate summary section with key metrics."""
        summary = dashboard_data.get("summary", {})
        
        health_score = summary.get("health_score", 0)
        total_metrics = summary.get("total_metrics", 0)
        active_alerts = summary.get("active_alerts", 0)
        critical_alerts = summary.get("critical_alerts", 0)
        
        # Determine health status
        if health_score >= 90:
            health_status = "excellent"
            health_icon = "🟢"
        elif health_score >= 75:
            health_status = "good"
            health_icon = "🟡"
        elif health_score >= 50:
            health_status = "acceptable"
            health_icon = "🟠"
        else:
            health_status = "critical"
            health_icon = "🔴"
        
        return f"""
        <div class="summary-grid">
            <div class="summary-card">
                <h3>System Health</h3>
                <div class="summary-value value-{health_status}">
                    {health_icon} {health_score:.1f}%
                </div>
                <div class="summary-trend">Overall system health score</div>
            </div>
            
            <div class="summary-card">
                <h3>Active Metrics</h3>
                <div class="summary-value">
                    📊 {total_metrics}
                </div>
                <div class="summary-trend">Quality metrics being monitored</div>
            </div>
            
            <div class="summary-card">
                <h3>Active Alerts</h3>
                <div class="summary-value {'value-critical' if active_alerts > 0 else ''}">
                    🚨 {active_alerts}
                </div>
                <div class="summary-trend">{critical_alerts} critical alerts</div>
            </div>
        </div>
        """
    
    def _generate_categories_section(self, dashboard_data: Dict[str, Any]) -> str:
        """Generate categories section with detailed metrics."""
        categories = dashboard_data.get("categories", {})
        
        category_cards = []
        category_icons = {
            "coverage": "🧪",
            "performance": "⚡",
            "quality": "✨",
            "security": "🔒",
            "testing": "🧾",
            "documentation": "📚"
        }
        
        for category_name, category_data in categories.items():
            if not category_data:
                continue
            
            icon = category_icons.get(category_name, "📊")
            category_title = category_name.replace("_", " ").title()
            
            metric_items = []
            for metric_name, summary in category_data.items():
                if "error" in summary:
                    continue
                
                latest_value = summary.get("latest_value", 0)
                metric_display = metric_name.replace("_", " ").title()
                
                # Determine status based on metric value (this is simplified)
                if latest_value >= 90:
                    status_class = "excellent"
                elif latest_value >= 80:
                    status_class = "good"
                elif latest_value >= 70:
                    status_class = "acceptable"
                elif latest_value >= 50:
                    status_class = "poor"
                else:
                    status_class = "critical"
                
                # Format value based on metric type
                if "coverage" in metric_name or "score" in metric_name or "rate" in metric_name:
                    formatted_value = f"{latest_value:.1f}%"
                elif "throughput" in metric_name:
                    formatted_value = f"{latest_value:.1f} seg/s"
                elif "time" in metric_name or "latency" in metric_name:
                    formatted_value = f"{latest_value:.2f}s"
                elif "count" in metric_name or "violations" in metric_name:
                    formatted_value = str(int(latest_value))
                else:
                    formatted_value = f"{latest_value:.1f}"
                
                metric_items.append(f"""
                <div class="metric-item">
                    <span class="metric-name">
                        <span class="status-indicator status-{status_class}"></span>
                        {metric_display}
                    </span>
                    <span class="metric-value value-{status_class}">{formatted_value}</span>
                </div>
                """)
            
            if metric_items:
                category_cards.append(f"""
                <div class="category-card">
                    <h3>{icon} {category_title}</h3>
                    {''.join(metric_items)}
                </div>
                """)
        
        return f"""
        <div class="category-grid">
            {''.join(category_cards)}
        </div>
        """
    
    def _generate_alerts_section(self, dashboard_data: Dict[str, Any]) -> str:
        """Generate alerts section."""
        alerts = dashboard_data.get("alerts", [])
        
        if not alerts:
            return f"""
            <div class="alerts-section">
                <div class="alerts-card">
                    <h3>🎉 No Active Alerts</h3>
                    <p>All quality metrics are within acceptable thresholds.</p>
                </div>
            </div>
            """
        
        alert_items = []
        alert_icons = {
            "critical": "🔴",
            "error": "🟠",
            "warning": "🟡",
            "info": "🔵"
        }
        
        for alert in alerts:
            severity = alert.get("severity", "info")
            icon = alert_icons.get(severity, "📊")
            message = html.escape(alert.get("message", ""))
            timestamp = alert.get("timestamp", "")
            
            # Parse timestamp for display
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                time_str = dt.strftime("%H:%M:%S")
            except:
                time_str = timestamp
            
            alert_items.append(f"""
            <div class="alert-item alert-{severity}">
                <span class="alert-icon">{icon}</span>
                <div>
                    <div>{message}</div>
                    <small>at {time_str}</small>
                </div>
            </div>
            """)
        
        return f"""
        <div class="alerts-section">
            <div class="alerts-card">
                <h3>🚨 Active Alerts ({len(alerts)})</h3>
                {''.join(alert_items)}
            </div>
        </div>
        """
    
    def get_metrics_data(self) -> Dict[str, Any]:
        """Get metrics data for API."""
        return self.metrics_collector.get_quality_dashboard_data()
    
    def get_alerts_data(self) -> Dict[str, Any]:
        """Get alerts data for API."""
        dashboard_data = self.metrics_collector.get_quality_dashboard_data()
        return {"alerts": dashboard_data.get("alerts", [])}
    
    def get_trends_data(self) -> Dict[str, Any]:
        """Get trends data for API."""
        # This would include historical trend analysis
        return {
            "trends": dashboard_data.get("trends", {}),
            "period": "24h"
        }


class QualityDashboard:
    """Quality dashboard server and management."""
    
    def __init__(self, metrics_collector, config: Optional[DashboardConfig] = None):
        """Initialize quality dashboard."""
        self.metrics_collector = metrics_collector
        self.config = config or DashboardConfig()
        self.logger = logging.getLogger(__name__)
        
        self.generator = QualityDashboardGenerator(metrics_collector, self.config)
        self.server = None
        self.server_thread = None
        self.running = False
    
    def start_server(self) -> bool:
        """Start the dashboard web server."""
        try:
            def create_handler(*args, **kwargs):
                return QualityDashboardHandler(*args, dashboard_generator=self.generator, **kwargs)
            
            self.server = HTTPServer((self.config.host, self.config.port), create_handler)
            
            def server_loop():
                self.running = True
                self.logger.info(f"Quality dashboard server started at http://{self.config.host}:{self.config.port}")
                try:
                    self.server.serve_forever()
                except Exception as e:
                    if self.running:  # Only log if we weren't intentionally stopped
                        self.logger.error(f"Dashboard server error: {e}")
                finally:
                    self.running = False
            
            self.server_thread = threading.Thread(target=server_loop, daemon=True)
            self.server_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start dashboard server: {e}")
            return False
    
    def stop_server(self):
        """Stop the dashboard web server."""
        if self.server:
            self.running = False
            self.server.shutdown()
            self.server.server_close()
            
            if self.server_thread:
                self.server_thread.join(timeout=5)
            
            self.logger.info("Quality dashboard server stopped")
    
    def open_browser(self):
        """Open the dashboard in the default web browser."""
        url = f"http://{self.config.host}:{self.config.port}"
        try:
            webbrowser.open(url)
            self.logger.info(f"Opened dashboard in browser: {url}")
        except Exception as e:
            self.logger.warning(f"Failed to open browser: {e}")
            self.logger.info(f"Dashboard available at: {url}")
    
    def generate_static_report(self, output_path: str) -> bool:
        """Generate a static HTML report file."""
        try:
            html_content = self.generator.generate_dashboard_html()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Static dashboard report generated: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate static report: {e}")
            return False


def main():
    """Main function for running quality dashboard."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Import metrics collector (would normally be passed in)
    from qa.metrics.quality_metrics_collector import QualityMetricsCollector
    
    # Initialize metrics collector
    collector = QualityMetricsCollector()
    
    # Record some sample metrics for demonstration
    collector.record_gauge("test_coverage_total", 87.5)
    collector.record_gauge("processing_throughput", 12.3)
    collector.record_gauge("code_quality_score", 91.2)
    collector.record_gauge("security_score", 98.5)
    collector.record_gauge("documentation_coverage", 75.0)
    
    # Initialize dashboard
    config = DashboardConfig(port=8080, auto_refresh_seconds=30)
    dashboard = QualityDashboard(collector, config)
    
    # Start server
    if dashboard.start_server():
        print(f"Quality Dashboard started at http://{config.host}:{config.port}")
        print("Press Ctrl+C to stop...")
        
        # Open browser
        dashboard.open_browser()
        
        try:
            # Keep the main thread alive
            while dashboard.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping dashboard...")
            dashboard.stop_server()
    else:
        print("Failed to start dashboard server")


if __name__ == "__main__":
    main()