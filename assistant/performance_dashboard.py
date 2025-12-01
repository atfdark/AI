#!/usr/bin/env python3
"""
Performance Monitoring Dashboard for Voice Assistant ML Components

This module provides a web-based dashboard for monitoring:
- Real-time performance metrics
- Component health status
- Error rates and trends
- Usage analytics
- Model performance tracking
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path
import threading
import queue

# Import our analysis tools
try:
    from .error_analysis import ErrorAnalyzer
    from .logger import get_logger
    logger = get_logger('performance_dashboard')
except ImportError:
    import logging
    logger = logging.getLogger('performance_dashboard')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    # Fallback error analyzer
    class ErrorAnalyzer:
        def analyze_recent_logs(self, hours=24):
            return {"error": "ErrorAnalyzer not available"}


class PerformanceDashboard:
    """Streamlit-based performance monitoring dashboard."""

    def __init__(self):
        self.analyzer = ErrorAnalyzer()
        self.refresh_interval = 30  # seconds
        self.data_cache = {}
        self.cache_timeout = 60  # seconds

    def run_dashboard(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="Voice Assistant Performance Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🎙️ Voice Assistant Performance Dashboard")
        st.markdown("Real-time monitoring of ML components and system health")

        # Sidebar controls
        st.sidebar.header("Dashboard Controls")

        time_range = st.sidebar.selectbox(
            "Time Range",
            ["1 hour", "6 hours", "24 hours", "7 days"],
            index=2
        )

        hours_map = {
            "1 hour": 1,
            "6 hours": 6,
            "24 hours": 24,
            "7 days": 168
        }
        selected_hours = hours_map[time_range]

        auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
        refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 10, 300, 30)

        if auto_refresh:
            # Auto-refresh logic would go here in a real implementation
            pass

        # Main dashboard content
        try:
            analysis_data = self._get_analysis_data(selected_hours)

            if "error" in analysis_data:
                st.error(f"Failed to load analysis data: {analysis_data['error']}")
                return

            # Overview metrics
            self._display_overview_metrics(analysis_data)

            # Component health
            self._display_component_health(analysis_data)

            # Performance charts
            self._display_performance_charts(analysis_data)

            # Error analysis
            self._display_error_analysis(analysis_data)

            # Usage patterns
            self._display_usage_patterns(analysis_data)

        except Exception as e:
            st.error(f"Dashboard error: {e}")
            logger.error(f"Dashboard error: {e}")

    def _get_analysis_data(self, hours: int) -> dict:
        """Get analysis data with caching."""
        cache_key = f"analysis_{hours}h"
        current_time = time.time()

        if cache_key in self.data_cache:
            cache_time, data = self.data_cache[cache_key]
            if current_time - cache_time < self.cache_timeout:
                return data

        # Fetch fresh data
        try:
            data = self.analyzer.analyze_recent_logs(hours)
            self.data_cache[cache_key] = (current_time, data)
            return data
        except Exception as e:
            logger.error(f"Failed to get analysis data: {e}")
            return {"error": str(e)}

    def _display_overview_metrics(self, data: dict):
        """Display overview metrics in columns."""
        st.header("📈 Overview Metrics")

        col1, col2, col3, col4 = st.columns(4)

        error_summary = data.get('error_summary', {})
        perf_metrics = data.get('performance_metrics', {})

        with col1:
            total_logs = data.get('total_logs', 0)
            st.metric("Total Logs", f"{total_logs:,}")

        with col2:
            total_errors = error_summary.get('total_errors', 0)
            error_rate = error_summary.get('error_rate', 0)
            st.metric("Total Errors", f"{total_errors:,}", f"{error_rate:.1%}")

        with col3:
            # Average prediction confidence
            if 'prediction_confidence' in perf_metrics:
                avg_confidence = perf_metrics['prediction_confidence'].get('mean', 0)
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
            else:
                st.metric("Avg Confidence", "N/A")

        with col4:
            # Average prediction time
            if 'prediction_time' in perf_metrics:
                avg_time = perf_metrics['prediction_time'].get('mean', 0)
                st.metric("Avg Response Time", f"{avg_time:.1f}ms")
            else:
                st.metric("Avg Response Time", "N/A")

    def _display_component_health(self, data: dict):
        """Display component health status."""
        st.header("🏥 Component Health")

        health_data = data.get('component_health', {})

        if not health_data:
            st.info("No component health data available")
            return

        # Create health dataframe
        health_df = pd.DataFrame.from_dict(health_data, orient='index')
        health_df = health_df.reset_index().rename(columns={'index': 'Component'})

        # Health score gauge chart
        fig = go.Figure()

        for _, row in health_df.iterrows():
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=row['health_score'],
                title={'text': row['Component']},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"}
                    ]
                },
                domain={'row': 0, 'column': health_df.index.get_loc(row.name)}
            ))

        fig.update_layout(
            grid={'rows': 1, 'columns': len(health_df)},
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

        # Health details table
        st.subheader("Health Details")
        display_df = health_df[['Component', 'health_score', 'error_rate', 'total_logs', 'last_activity']]
        display_df.columns = ['Component', 'Health Score', 'Error Rate', 'Total Logs', 'Last Activity']
        st.dataframe(display_df, use_container_width=True)

    def _display_performance_charts(self, data: dict):
        """Display performance metrics charts."""
        st.header("📊 Performance Metrics")

        perf_metrics = data.get('performance_metrics', {})

        if not perf_metrics:
            st.info("No performance metrics available")
            return

        # Create subplots for different metrics
        metrics_to_show = ['prediction_time', 'prediction_confidence']
        available_metrics = [m for m in metrics_to_show if m in perf_metrics]

        if not available_metrics:
            st.info("No relevant performance metrics found")
            return

        fig = make_subplots(
            rows=1, cols=len(available_metrics),
            subplot_titles=[m.replace('_', ' ').title() for m in available_metrics]
        )

        for i, metric_name in enumerate(available_metrics):
            metric_data = perf_metrics[metric_name]

            # Create a simple bar chart for the metric statistics
            fig.add_trace(
                go.Bar(
                    x=['Mean', 'Min', 'Max'],
                    y=[
                        metric_data.get('mean', 0),
                        metric_data.get('min', 0),
                        metric_data.get('max', 0)
                    ],
                    name=metric_name.replace('_', ' ').title(),
                    showlegend=False
                ),
                row=1, col=i+1
            )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    def _display_error_analysis(self, data: dict):
        """Display error analysis section."""
        st.header("🚨 Error Analysis")

        error_summary = data.get('error_summary', {})

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Errors by Component")
            errors_by_component = error_summary.get('errors_by_component', {})

            if errors_by_component:
                fig = px.pie(
                    values=list(errors_by_component.values()),
                    names=list(errors_by_component.keys()),
                    title="Error Distribution by Component"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No error data available")

        with col2:
            st.subheader("Errors by Type")
            errors_by_type = error_summary.get('errors_by_type', {})

            if errors_by_type:
                fig = px.bar(
                    x=list(errors_by_type.keys()),
                    y=list(errors_by_type.values()),
                    title="Errors by Type"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No error type data available")

        # Most common errors
        st.subheader("Most Common Errors")
        common_errors = error_summary.get('most_common_errors', [])

        if common_errors:
            error_df = pd.DataFrame(common_errors, columns=['Error Message', 'Count'])
            st.dataframe(error_df, use_container_width=True)
        else:
            st.info("No common error data available")

    def _display_usage_patterns(self, data: dict):
        """Display usage patterns."""
        st.header("📈 Usage Patterns")

        usage_patterns = data.get('usage_patterns', {})

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Hourly Activity")
            hourly_activity = usage_patterns.get('hourly_activity', {})

            if hourly_activity:
                hours = list(range(24))
                activity_values = [hourly_activity.get(h, 0) for h in hours]

                fig = px.bar(
                    x=hours,
                    y=activity_values,
                    title="Activity by Hour",
                    labels={'x': 'Hour of Day', 'y': 'Activity Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hourly activity data available")

        with col2:
            st.subheader("Component Usage")
            component_usage = usage_patterns.get('component_usage', {})

            if component_usage:
                fig = px.pie(
                    values=list(component_usage.values()),
                    names=list(component_usage.keys()),
                    title="Usage by Component"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No component usage data available")

        # Peak hours
        peak_hours = usage_patterns.get('peak_hours', [])
        if peak_hours:
            st.subheader("Peak Usage Hours")
            st.write(f"Peak activity occurs at: {', '.join(map(str, sorted(peak_hours)))}:00 hours")


class BackgroundDataUpdater:
    """Background thread for updating dashboard data."""

    def __init__(self, dashboard: PerformanceDashboard):
        self.dashboard = dashboard
        self.running = False
        self.thread = None
        self.update_queue = queue.Queue()

    def start(self):
        """Start background updates."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        logger.info("Background data updater started")

    def stop(self):
        """Stop background updates."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Background data updater stopped")

    def _update_loop(self):
        """Main update loop."""
        while self.running:
            try:
                # Clear old cache entries
                current_time = time.time()
                expired_keys = [
                    key for key, (cache_time, _) in self.dashboard.data_cache.items()
                    if current_time - cache_time > self.dashboard.cache_timeout
                ]
                for key in expired_keys:
                    del self.dashboard.data_cache[key]

                # Pre-fetch common data ranges
                for hours in [1, 6, 24]:
                    try:
                        self.dashboard._get_analysis_data(hours)
                    except Exception as e:
                        logger.error(f"Failed to pre-fetch {hours}h data: {e}")

            except Exception as e:
                logger.error(f"Background update error: {e}")

            time.sleep(60)  # Update every minute


def run_dashboard():
    """Run the performance dashboard."""
    dashboard = PerformanceDashboard()

    # Start background updater
    updater = BackgroundDataUpdater(dashboard)
    updater.start()

    try:
        dashboard.run_dashboard()
    finally:
        updater.stop()


if __name__ == "__main__":
    run_dashboard()