"""
Epic 4 - Story 4.3: Benchmarking & Continuous Improvement
Quality Metrics Dashboard for real-time monitoring and benchmarking

This module provides:
- Real-time quality metrics monitoring
- Interactive web dashboard using Streamlit
- Performance benchmarking and trend analysis
- Automated alerting for quality degradation
- Historical metrics tracking and visualization
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from pathlib import Path

# Custom imports
from ..validation.golden_dataset_validator import GoldenDatasetValidator, ValidationMetrics
from ...utils.metrics_collector import MetricsCollector


class MetricsDatabase:
    """Database for storing and retrieving metrics history."""
    
    def __init__(self, db_path: str = "data/metrics/quality_metrics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize the metrics database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                overall_accuracy REAL,
                word_error_rate REAL,
                sanskrit_accuracy REAL,
                hindi_accuracy REAL,
                iast_compliance REAL,
                verse_accuracy REAL,
                character_accuracy REAL,
                total_segments INTEGER,
                processed_segments INTEGER,
                failed_segments INTEGER,
                processing_time REAL,
                segments_per_second REAL,
                high_quality_segments INTEGER,
                medium_quality_segments INTEGER,
                low_quality_segments INTEGER,
                batch_id TEXT,
                version TEXT
            )
        ''')
        
        # Create performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                throughput REAL,
                latency REAL,
                memory_usage REAL,
                cpu_usage REAL,
                error_rate REAL,
                cache_hit_ratio REAL,
                batch_id TEXT
            )
        ''')
        
        # Create alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                metric_name TEXT,
                current_value REAL,
                threshold_value REAL,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_quality_metrics(self, metrics: ValidationMetrics, batch_id: str = None, version: str = None):
        """Store quality metrics in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO quality_metrics (
                timestamp, overall_accuracy, word_error_rate, sanskrit_accuracy,
                hindi_accuracy, iast_compliance, verse_accuracy, character_accuracy,
                total_segments, processed_segments, failed_segments,
                processing_time, segments_per_second, high_quality_segments,
                medium_quality_segments, low_quality_segments, batch_id, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow().isoformat(),
            metrics.overall_accuracy,
            metrics.word_error_rate,
            metrics.sanskrit_accuracy,
            metrics.hindi_accuracy,
            metrics.iast_compliance,
            metrics.verse_accuracy,
            metrics.character_accuracy,
            metrics.total_segments,
            metrics.processed_segments,
            metrics.failed_segments,
            metrics.processing_time,
            metrics.segments_per_second,
            metrics.high_quality_segments,
            metrics.medium_quality_segments,
            metrics.low_quality_segments,
            batch_id,
            version
        ))
        
        conn.commit()
        conn.close()
    
    def store_performance_metrics(self, metrics: Dict[str, Any]):
        """Store performance metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_metrics (
                timestamp, throughput, latency, memory_usage,
                cpu_usage, error_rate, cache_hit_ratio, batch_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow().isoformat(),
            metrics.get('throughput', 0),
            metrics.get('latency', 0),
            metrics.get('memory_usage', 0),
            metrics.get('cpu_usage', 0),
            metrics.get('error_rate', 0),
            metrics.get('cache_hit_ratio', 0),
            metrics.get('batch_id')
        ))
        
        conn.commit()
        conn.close()
    
    def store_alert(self, alert_type: str, severity: str, message: str, 
                   metric_name: str = None, current_value: float = None, 
                   threshold_value: float = None):
        """Store quality alert."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO quality_alerts (
                timestamp, alert_type, severity, message,
                metric_name, current_value, threshold_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow().isoformat(),
            alert_type,
            severity,
            message,
            metric_name,
            current_value,
            threshold_value
        ))
        
        conn.commit()
        conn.close()
    
    def get_quality_metrics_history(self, hours: int = 24) -> pd.DataFrame:
        """Get quality metrics history."""
        cutoff_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM quality_metrics 
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        ''', conn, params=(cutoff_time,))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_performance_metrics_history(self, hours: int = 24) -> pd.DataFrame:
        """Get performance metrics history."""
        cutoff_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM performance_metrics 
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        ''', conn, params=(cutoff_time,))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_recent_alerts(self, hours: int = 24) -> pd.DataFrame:
        """Get recent alerts."""
        cutoff_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM quality_alerts 
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        ''', conn, params=(cutoff_time,))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df


class QualityMonitor:
    """Monitor for detecting quality issues and generating alerts."""
    
    def __init__(self, db: MetricsDatabase):
        self.db = db
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Alert thresholds
        self.thresholds = {
            'overall_accuracy': 0.80,
            'word_error_rate': 0.25,
            'sanskrit_accuracy': 0.75,
            'hindi_accuracy': 0.75,
            'iast_compliance': 0.85,
            'verse_accuracy': 0.70,
            'processing_time': 300.0,  # seconds
            'error_rate': 0.05
        }
    
    def check_quality_metrics(self, metrics: ValidationMetrics) -> List[Dict[str, Any]]:
        """Check metrics against thresholds and generate alerts."""
        alerts = []
        
        # Overall accuracy check
        if metrics.overall_accuracy < self.thresholds['overall_accuracy']:
            alert = {
                'type': 'quality_degradation',
                'severity': 'high',
                'message': f'Overall accuracy ({metrics.overall_accuracy:.1%}) below threshold ({self.thresholds["overall_accuracy"]:.1%})',
                'metric_name': 'overall_accuracy',
                'current_value': metrics.overall_accuracy,
                'threshold_value': self.thresholds['overall_accuracy']
            }
            alerts.append(alert)
            self.db.store_alert(**alert)
        
        # Word error rate check
        if metrics.word_error_rate > self.thresholds['word_error_rate']:
            alert = {
                'type': 'quality_degradation',
                'severity': 'high',
                'message': f'Word error rate ({metrics.word_error_rate:.1%}) above threshold ({self.thresholds["word_error_rate"]:.1%})',
                'metric_name': 'word_error_rate',
                'current_value': metrics.word_error_rate,
                'threshold_value': self.thresholds['word_error_rate']
            }
            alerts.append(alert)
            self.db.store_alert(**alert)
        
        # Sanskrit accuracy check
        if metrics.sanskrit_accuracy < self.thresholds['sanskrit_accuracy']:
            alert = {
                'type': 'sanskrit_quality',
                'severity': 'medium',
                'message': f'Sanskrit accuracy ({metrics.sanskrit_accuracy:.1%}) below threshold',
                'metric_name': 'sanskrit_accuracy',
                'current_value': metrics.sanskrit_accuracy,
                'threshold_value': self.thresholds['sanskrit_accuracy']
            }
            alerts.append(alert)
            self.db.store_alert(**alert)
        
        # IAST compliance check
        if metrics.iast_compliance < self.thresholds['iast_compliance']:
            alert = {
                'type': 'iast_compliance',
                'severity': 'medium',
                'message': f'IAST compliance ({metrics.iast_compliance:.1%}) below threshold',
                'metric_name': 'iast_compliance',
                'current_value': metrics.iast_compliance,
                'threshold_value': self.thresholds['iast_compliance']
            }
            alerts.append(alert)
            self.db.store_alert(**alert)
        
        return alerts
    
    def check_performance_trends(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for performance trends and anomalies."""
        alerts = []
        
        if df.empty or len(df) < 5:
            return alerts
        
        # Check for declining accuracy trend
        recent_accuracy = df['overall_accuracy'].tail(5).mean()
        older_accuracy = df['overall_accuracy'].head(5).mean()
        
        if recent_accuracy < older_accuracy - 0.05:  # 5% decline
            alerts.append({
                'type': 'trending_degradation',
                'severity': 'medium',
                'message': f'Declining accuracy trend detected: {recent_accuracy:.1%} vs {older_accuracy:.1%}',
                'metric_name': 'overall_accuracy_trend',
                'current_value': recent_accuracy,
                'threshold_value': older_accuracy
            })
        
        return alerts


class MetricsDashboard:
    """
    Streamlit-based dashboard for quality metrics monitoring.
    
    Features:
    - Real-time metrics visualization
    - Historical trend analysis
    - Performance benchmarking
    - Alert management
    - Quality regression detection
    """
    
    def __init__(self):
        self.db = MetricsDatabase()
        self.monitor = QualityMonitor(self.db)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run_dashboard(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="Sanskrit Processing Quality Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📊 Sanskrit Processing Quality Dashboard")
        st.markdown("Real-time monitoring of processing quality and performance metrics")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main dashboard content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_main_metrics()
            self._render_quality_trends()
            self._render_performance_metrics()
        
        with col2:
            self._render_alerts()
            self._render_system_status()
            self._render_quick_actions()
    
    def _render_sidebar(self):
        """Render the sidebar controls."""
        st.sidebar.header("Dashboard Controls")
        
        # Time range selector
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["Last 1 Hour", "Last 4 Hours", "Last 12 Hours", "Last 24 Hours", "Last 7 Days"],
            index=2
        )
        
        hours_map = {
            "Last 1 Hour": 1,
            "Last 4 Hours": 4,
            "Last 12 Hours": 12,
            "Last 24 Hours": 24,
            "Last 7 Days": 168
        }
        st.session_state.time_range_hours = hours_map[time_range]
        
        # Auto-refresh control
        auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
        if auto_refresh:
            st.rerun()
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()
        
        # Dashboard sections
        st.sidebar.header("Dashboard Sections")
        show_alerts = st.sidebar.checkbox("Show Alerts", value=True)
        show_trends = st.sidebar.checkbox("Show Trends", value=True)
        show_performance = st.sidebar.checkbox("Show Performance", value=True)
        
        st.session_state.show_alerts = show_alerts
        st.session_state.show_trends = show_trends
        st.session_state.show_performance = show_performance
    
    def _render_main_metrics(self):
        """Render the main quality metrics."""
        st.header("📈 Quality Metrics")
        
        # Get latest metrics
        df = self.db.get_quality_metrics_history(hours=st.session_state.time_range_hours)
        
        if df.empty:
            st.warning("No quality metrics data available for the selected time range.")
            return
        
        # Latest metrics
        latest = df.iloc[0]
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Overall Accuracy",
                f"{latest['overall_accuracy']:.1%}",
                delta=self._calculate_delta(df, 'overall_accuracy'),
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "Word Error Rate",
                f"{latest['word_error_rate']:.1%}",
                delta=self._calculate_delta(df, 'word_error_rate'),
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                "Sanskrit Accuracy",
                f"{latest['sanskrit_accuracy']:.1%}",
                delta=self._calculate_delta(df, 'sanskrit_accuracy')
            )
        
        with col4:
            st.metric(
                "Processing Speed",
                f"{latest['segments_per_second']:.1f} seg/s",
                delta=self._calculate_delta(df, 'segments_per_second')
            )
        
        # Additional metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("IAST Compliance", f"{latest['iast_compliance']:.1%}")
        
        with col2:
            st.metric("Verse Accuracy", f"{latest['verse_accuracy']:.1%}")
        
        with col3:
            st.metric("Character Accuracy", f"{latest['character_accuracy']:.1%}")
        
        with col4:
            st.metric("Total Segments", f"{latest['total_segments']:,}")
    
    def _calculate_delta(self, df: pd.DataFrame, column: str) -> Optional[str]:
        """Calculate delta for metrics display."""
        if len(df) < 2:
            return None
        
        current = df.iloc[0][column]
        previous = df.iloc[1][column]
        delta = current - previous
        
        if abs(delta) < 0.001:  # Minimal change
            return None
        
        return f"{delta:+.1%}" if abs(delta) < 1 else f"{delta:+.1f}"
    
    def _render_quality_trends(self):
        """Render quality trend visualizations."""
        if not st.session_state.get('show_trends', True):
            return
        
        st.header("📊 Quality Trends")
        
        df = self.db.get_quality_metrics_history(hours=st.session_state.time_range_hours)
        
        if df.empty:
            st.info("No trend data available.")
            return
        
        # Create subplots for different metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Accuracy', 'Word Error Rate', 'Sanskrit/Hindi Accuracy', 'IAST Compliance'),
            vertical_spacing=0.1
        )
        
        # Overall accuracy
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['overall_accuracy'],
                mode='lines+markers',
                name='Overall Accuracy',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Word error rate
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['word_error_rate'],
                mode='lines+markers',
                name='Word Error Rate',
                line=dict(color='red')
            ),
            row=1, col=2
        )
        
        # Sanskrit and Hindi accuracy
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['sanskrit_accuracy'],
                mode='lines+markers',
                name='Sanskrit',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['hindi_accuracy'],
                mode='lines+markers',
                name='Hindi',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        
        # IAST compliance
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['iast_compliance'],
                mode='lines+markers',
                name='IAST Compliance',
                line=dict(color='purple')
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Accuracy")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_performance_metrics(self):
        """Render performance metrics."""
        if not st.session_state.get('show_performance', True):
            return
        
        st.header("⚡ Performance Metrics")
        
        perf_df = self.db.get_performance_metrics_history(hours=st.session_state.time_range_hours)
        
        if perf_df.empty:
            st.info("No performance data available.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Throughput chart
            fig = px.line(
                perf_df, 
                x='timestamp', 
                y='throughput',
                title='Processing Throughput',
                labels={'throughput': 'Files/Hour', 'timestamp': 'Time'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Latency chart
            fig = px.line(
                perf_df, 
                x='timestamp', 
                y='latency',
                title='Processing Latency',
                labels={'latency': 'Seconds', 'timestamp': 'Time'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Resource utilization
        col1, col2 = st.columns(2)
        
        with col1:
            if 'memory_usage' in perf_df.columns:
                fig = px.line(
                    perf_df, 
                    x='timestamp', 
                    y='memory_usage',
                    title='Memory Usage',
                    labels={'memory_usage': 'MB', 'timestamp': 'Time'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'cache_hit_ratio' in perf_df.columns:
                fig = px.line(
                    perf_df, 
                    x='timestamp', 
                    y='cache_hit_ratio',
                    title='Cache Hit Ratio',
                    labels={'cache_hit_ratio': 'Ratio', 'timestamp': 'Time'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_alerts(self):
        """Render alerts section."""
        if not st.session_state.get('show_alerts', True):
            return
        
        st.header("🚨 Recent Alerts")
        
        alerts_df = self.db.get_recent_alerts(hours=24)
        
        if alerts_df.empty:
            st.success("No recent alerts! 🎉")
            return
        
        # Group alerts by severity
        critical_alerts = alerts_df[alerts_df['severity'] == 'high']
        warning_alerts = alerts_df[alerts_df['severity'] == 'medium']
        
        if not critical_alerts.empty:
            st.error(f"🔥 {len(critical_alerts)} Critical Alerts")
            for _, alert in critical_alerts.head(5).iterrows():
                with st.expander(f"⚠️ {alert['alert_type']} - {alert['timestamp'].strftime('%H:%M')}"):
                    st.write(alert['message'])
                    if alert['metric_name']:
                        st.write(f"Metric: {alert['metric_name']}")
                        st.write(f"Current: {alert['current_value']:.3f}")
                        st.write(f"Threshold: {alert['threshold_value']:.3f}")
        
        if not warning_alerts.empty:
            st.warning(f"⚠️ {len(warning_alerts)} Warning Alerts")
            for _, alert in warning_alerts.head(3).iterrows():
                with st.expander(f"⚠️ {alert['alert_type']} - {alert['timestamp'].strftime('%H:%M')}"):
                    st.write(alert['message'])
    
    def _render_system_status(self):
        """Render system status section."""
        st.header("💚 System Status")
        
        # Get latest metrics for status
        df = self.db.get_quality_metrics_history(hours=1)
        
        if df.empty:
            st.error("System Status: Unknown")
            return
        
        latest = df.iloc[0]
        
        # Determine overall status
        status_score = (
            latest['overall_accuracy'] * 0.4 +
            (1 - latest['word_error_rate']) * 0.3 +
            latest['sanskrit_accuracy'] * 0.2 +
            latest['iast_compliance'] * 0.1
        )
        
        if status_score >= 0.85:
            st.success("✅ System Healthy")
        elif status_score >= 0.75:
            st.warning("⚠️ System Degraded")
        else:
            st.error("🔥 System Critical")
        
        # Status details
        st.write(f"Overall Score: {status_score:.1%}")
        st.write(f"Last Update: {latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Quick stats
        st.subheader("Quick Stats")
        st.write(f"• Processed: {latest['processed_segments']:,} segments")
        st.write(f"• Failed: {latest['failed_segments']:,} segments")
        st.write(f"• Success Rate: {(latest['processed_segments']/(latest['total_segments'] or 1)):.1%}")
    
    def _render_quick_actions(self):
        """Render quick actions section."""
        st.header("🔧 Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Run Validation", key="run_validation"):
                st.info("Validation would be triggered here...")
        
        with col2:
            if st.button("📊 Generate Report", key="generate_report"):
                st.info("Report generation would be triggered here...")
        
        if st.button("🧹 Clear Alerts", key="clear_alerts"):
            st.info("Alerts would be cleared here...")
        
        if st.button("⚙️ System Health Check", key="health_check"):
            st.info("Health check would be performed here...")


# Utility functions for dashboard deployment

def run_metrics_dashboard():
    """Run the metrics dashboard application."""
    dashboard = MetricsDashboard()
    dashboard.run_dashboard()


def start_dashboard_server(port: int = 8501, host: str = "0.0.0.0"):
    """Start the dashboard server."""
    import subprocess
    import sys
    
    dashboard_file = __file__
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", dashboard_file,
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true"
    ]
    
    subprocess.run(cmd)


if __name__ == "__main__":
    # Run dashboard when executed directly
    run_metrics_dashboard()