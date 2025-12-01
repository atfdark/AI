#!/usr/bin/env python3
"""
Automated Reporting System for Voice Assistant ML Components

This module provides automated report generation and alerting:
- Scheduled report generation
- Email/Slack notifications for critical issues
- Performance trend analysis
- Alert thresholds and escalation
"""

import json
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import threading
import schedule
import requests

# Import our analysis tools
try:
    from .error_analysis import ErrorAnalyzer, run_error_analysis
    from .logger import get_logger
    logger = get_logger('automated_reporting')
except ImportError:
    import logging
    logger = logging.getLogger('automated_reporting')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    # Fallback classes
    class ErrorAnalyzer:
        def analyze_recent_logs(self, hours=24):
            return {"error": "ErrorAnalyzer not available"}

    def run_error_analysis(**kwargs):
        return {"error": "run_error_analysis not available"}


class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), '..', 'config.json')
        self.config = self._load_config()
        self.alert_history = []
        self.cooldown_period = 3600  # 1 hour between similar alerts

    def _load_config(self) -> dict:
        """Load configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {
                'alerting': {
                    'email': {
                        'enabled': False,
                        'smtp_server': 'smtp.gmail.com',
                        'smtp_port': 587,
                        'sender_email': '',
                        'sender_password': '',
                        'recipients': []
                    },
                    'slack': {
                        'enabled': False,
                        'webhook_url': '',
                        'channel': '#alerts'
                    }
                }
            }

    def check_and_alert(self, analysis_data: dict, alert_rules: dict):
        """Check conditions and send alerts if needed."""
        alerts_triggered = []

        # Check error rate alerts
        error_rate = analysis_data.get('error_summary', {}).get('error_rate', 0)
        if error_rate > alert_rules.get('max_error_rate', 0.1):
            alert = {
                'type': 'error_rate',
                'severity': 'high' if error_rate > 0.2 else 'medium',
                'message': f"High error rate detected: {error_rate:.1%}",
                'details': {
                    'error_rate': error_rate,
                    'threshold': alert_rules.get('max_error_rate', 0.1),
                    'total_errors': analysis_data.get('error_summary', {}).get('total_errors', 0)
                }
            }
            alerts_triggered.append(alert)

        # Check component health alerts
        component_health = analysis_data.get('component_health', {})
        for component, health in component_health.items():
            health_score = health.get('health_score', 100)
            if health_score < alert_rules.get('min_health_score', 50):
                alert = {
                    'type': 'component_health',
                    'severity': 'high' if health_score < 30 else 'medium',
                    'message': f"Component '{component}' health critical: {health_score}/100",
                    'details': {
                        'component': component,
                        'health_score': health_score,
                        'error_rate': health.get('error_rate', 0)
                    }
                }
                alerts_triggered.append(alert)

        # Check performance degradation
        perf_metrics = analysis_data.get('performance_metrics', {})
        if 'prediction_time' in perf_metrics:
            avg_time = perf_metrics['prediction_time'].get('mean', 0)
            if avg_time > alert_rules.get('max_prediction_time', 2000):  # 2 seconds
                alert = {
                    'type': 'performance',
                    'severity': 'medium',
                    'message': f"Slow response time: {avg_time:.1f}ms average",
                    'details': {
                        'avg_prediction_time': avg_time,
                        'threshold': alert_rules.get('max_prediction_time', 2000)
                    }
                }
                alerts_triggered.append(alert)

        # Send alerts (with cooldown)
        for alert in alerts_triggered:
            if self._should_send_alert(alert):
                self._send_alert(alert)
                self.alert_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'alert': alert
                })

        return alerts_triggered

    def _should_send_alert(self, alert: dict) -> bool:
        """Check if alert should be sent (cooldown logic)."""
        current_time = datetime.now()

        # Check recent alerts of same type
        for history_item in reversed(self.alert_history[-10:]):  # Check last 10 alerts
            alert_time = datetime.fromisoformat(history_item['timestamp'])
            if (current_time - alert_time).seconds < self.cooldown_period:
                if history_item['alert']['type'] == alert['type']:
                    if history_item['alert'].get('severity') == alert.get('severity'):
                        return False  # Same type and severity within cooldown

        return True

    def _send_alert(self, alert: dict):
        """Send alert via configured channels."""
        alert_config = self.config.get('alerting', {})

        # Email alerts
        if alert_config.get('email', {}).get('enabled', False):
            self._send_email_alert(alert)

        # Slack alerts
        if alert_config.get('slack', {}).get('enabled', False):
            self._send_slack_alert(alert)

        logger.warning(f"Alert sent: {alert['message']}")

    def _send_email_alert(self, alert: dict):
        """Send alert via email."""
        email_config = self.config['alerting']['email']

        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['sender_email']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"Voice Assistant Alert: {alert['type'].replace('_', ' ').title()}"

            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            emoji = severity_emoji.get(alert.get('severity', 'medium'), '🟡')

            body = f"""
{emoji} **{alert['severity'].upper()} ALERT**

{alert['message']}

**Details:**
{json.dumps(alert.get('details', {}), indent=2)}

**Time:** {datetime.now().isoformat()}

This is an automated alert from the Voice Assistant monitoring system.
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            text = msg.as_string()
            server.sendmail(email_config['sender_email'], email_config['recipients'], text)
            server.quit()

            logger.info("Email alert sent successfully")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _send_slack_alert(self, alert: dict):
        """Send alert via Slack webhook."""
        slack_config = self.config['alerting']['slack']

        try:
            severity_colors = {'high': 'danger', 'medium': 'warning', 'low': 'good'}

            payload = {
                "channel": slack_config['channel'],
                "attachments": [{
                    "color": severity_colors.get(alert.get('severity', 'medium'), 'warning'),
                    "title": f"Voice Assistant Alert: {alert['type'].replace('_', ' ').title()}",
                    "text": alert['message'],
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.get('severity', 'medium').upper(),
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "short": True
                        }
                    ],
                    "footer": "Voice Assistant Monitoring System"
                }]
            }

            # Add details if available
            if alert.get('details'):
                details_text = "\n".join([f"• {k}: {v}" for k, v in alert['details'].items()])
                payload['attachments'][0]['text'] += f"\n\n*Details:*\n{details_text}"

            response = requests.post(slack_config['webhook_url'], json=payload)
            response.raise_for_status()

            logger.info("Slack alert sent successfully")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")


class AutomatedReporter:
    """Automated report generation and scheduling."""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), '..', 'config.json')
        self.config = self._load_config()
        self.analyzer = ErrorAnalyzer()
        self.alert_manager = AlertManager(config_path)
        self.running = False
        self.thread = None

    def _load_config(self) -> dict:
        """Load configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {
                'reporting': {
                    'enabled': True,
                    'schedule': {
                        'daily_report': '18:00',  # 6 PM daily
                        'weekly_report': 'Monday 09:00',
                        'alert_check_interval': 300  # 5 minutes
                    },
                    'alert_rules': {
                        'max_error_rate': 0.1,
                        'min_health_score': 50,
                        'max_prediction_time': 2000
                    },
                    'output_dir': 'reports'
                }
            }

    def start_automated_reporting(self):
        """Start automated reporting system."""
        if self.running:
            return

        self.running = True

        # Setup scheduled tasks
        reporting_config = self.config.get('reporting', {})
        schedule_config = reporting_config.get('schedule', {})

        # Daily report
        if 'daily_report' in schedule_config:
            schedule.every().day.at(schedule_config['daily_report']).do(self._generate_daily_report)

        # Weekly report
        if 'weekly_report' in schedule_config:
            # Parse weekly schedule (e.g., "Monday 09:00")
            try:
                day_time = schedule_config['weekly_report'].split()
                if len(day_time) == 2:
                    day, time_str = day_time
                    getattr(schedule.every(), day.lower()).at(time_str).do(self._generate_weekly_report)
            except:
                logger.error("Invalid weekly report schedule format")

        # Alert checking
        alert_interval = schedule_config.get('alert_check_interval', 300)
        schedule.every(alert_interval).seconds.do(self._check_alerts)

        # Start scheduler thread
        self.thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.thread.start()

        logger.info("Automated reporting system started")

    def stop_automated_reporting(self):
        """Stop automated reporting system."""
        self.running = False
        schedule.clear()
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("Automated reporting system stopped")

    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def _generate_daily_report(self):
        """Generate daily performance report."""
        try:
            logger.info("Generating daily report...")

            # Analyze last 24 hours
            results = run_error_analysis(hours=24, generate_report=True)

            if 'error' in results:
                logger.error(f"Failed to generate daily report: {results['error']}")
                return

            # Move report to reports directory
            self._organize_report_files(results, 'daily')

            logger.info("Daily report generated successfully")

        except Exception as e:
            logger.error(f"Daily report generation failed: {e}")

    def _generate_weekly_report(self):
        """Generate weekly performance report."""
        try:
            logger.info("Generating weekly report...")

            # Analyze last 7 days
            results = run_error_analysis(hours=168, generate_report=True)

            if 'error' in results:
                logger.error(f"Failed to generate weekly report: {results['error']}")
                return

            # Move report to reports directory
            self._organize_report_files(results, 'weekly')

            logger.info("Weekly report generated successfully")

        except Exception as e:
            logger.error(f"Weekly report generation failed: {e}")

    def _check_alerts(self):
        """Check for alert conditions."""
        try:
            # Analyze recent logs (last hour for alerts)
            analysis_data = self.analyzer.analyze_recent_logs(hours=1)

            if 'error' in analysis_data:
                logger.error(f"Failed to check alerts: {analysis_data['error']}")
                return

            # Check alert conditions
            alert_rules = self.config.get('reporting', {}).get('alert_rules', {})
            alerts = self.alert_manager.check_and_alert(analysis_data, alert_rules)

            if alerts:
                logger.info(f"Triggered {len(alerts)} alerts")

        except Exception as e:
            logger.error(f"Alert checking failed: {e}")

    def _organize_report_files(self, results: dict, report_type: str):
        """Organize generated report files."""
        try:
            reporting_config = self.config.get('reporting', {})
            output_dir = Path(reporting_config.get('output_dir', 'reports'))
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Move report file
            if results.get('report_file') and os.path.exists(results['report_file']):
                new_name = f"{report_type}_report_{timestamp}.md"
                new_path = output_dir / new_name
                os.rename(results['report_file'], new_path)
                logger.info(f"Report moved to: {new_path}")

            # Move export directory
            if results.get('export_dir') and os.path.exists(results['export_dir']):
                new_export_dir = output_dir / f"{report_type}_data_{timestamp}"
                os.rename(results['export_dir'], new_export_dir)
                logger.info(f"Export data moved to: {new_export_dir}")

        except Exception as e:
            logger.error(f"Failed to organize report files: {e}")

    def generate_manual_report(self, hours: int = 24, report_type: str = 'manual') -> dict:
        """Generate a manual report on demand."""
        try:
            logger.info(f"Generating manual {hours}h report...")

            results = run_error_analysis(hours=hours, generate_report=True)

            if 'error' in results:
                return results

            # Organize files
            self._organize_report_files(results, report_type)

            return {
                'status': 'success',
                'report_file': results.get('report_file'),
                'export_dir': results.get('export_dir')
            }

        except Exception as e:
            error_msg = f"Manual report generation failed: {e}"
            logger.error(error_msg)
            return {'status': 'error', 'message': error_msg}


def configure_alerting(email_config: dict = None, slack_config: dict = None):
    """Configure alerting settings."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')

    try:
        # Load existing config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except:
        config = {}

    # Update alerting config
    if 'alerting' not in config:
        config['alerting'] = {}

    if email_config:
        config['alerting']['email'] = email_config

    if slack_config:
        config['alerting']['slack'] = slack_config

    # Save config
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, indent=2, ensure_ascii=False)

    logger.info("Alerting configuration updated")


def run_automated_reporting():
    """Run the automated reporting system."""
    reporter = AutomatedReporter()

    try:
        reporter.start_automated_reporting()

        # Keep running
        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Shutting down automated reporting...")
    finally:
        reporter.stop_automated_reporting()


if __name__ == "__main__":
    # Example usage
    reporter = AutomatedReporter()

    # Generate a manual report
    result = reporter.generate_manual_report(hours=24)
    print(f"Manual report result: {result}")

    # Start automated reporting (commented out for safety)
    # run_automated_reporting()