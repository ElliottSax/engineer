#!/usr/bin/env python3
"""
HEALTH MONITORING DAEMON
Background service for continuous platform health monitoring
Runs automated checks, logs events, sends alerts, and maintains statistics
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Any

# yaml is optional
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Import our health monitoring components
from platform_health_checker import PlatformHealthChecker, HealthStatus, HealthCheckResult
from self_healing_orchestrator import SelfHealingOrchestrator

# Import utilities
try:
    from utils.file_locking import safe_write_json, safe_read_json
    from utils.config_loader import get_config
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

# Configurable base directory (defaults to script location)
BASE_DIR = Path(os.environ.get('HEALTH_MONITOR_BASE_DIR', Path(__file__).parent))

# Configure logging with rotation
_log_file = BASE_DIR / 'health_monitoring.log'
_log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        RotatingFileHandler(
            _log_file,
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HealthMonitoringDaemon:
    """Background daemon for continuous health monitoring"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = str(BASE_DIR / 'health_config.yaml')
        self.config = self._load_config(config_path)
        self.health_checker = PlatformHealthChecker()
        self.orchestrator = SelfHealingOrchestrator()
        self.running = False
        self.stats = self._init_stats()
        self.alert_handlers = self._init_alert_handlers()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""

        # Try centralized config first
        if UTILS_AVAILABLE:
            health_config = get_config('health_monitoring', default=None)
            if health_config:
                logger.info("Loaded configuration from centralized config.yaml")
                return {'monitoring': health_config, **self._default_config()}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if YAML_AVAILABLE:
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._default_config()
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid config file format: {e}")
            return self._default_config()

    def _default_config(self) -> Dict:
        """Return default configuration"""

        return {
            'monitoring': {
                'check_interval': 300,  # 5 minutes
                'detailed_check_interval': 3600,  # 1 hour
                'enable_predictive_alerts': True,
                'enable_auto_failover': True,
            },
            'alerts': {
                'console': True,
                'log_file': True,
                'webhook': False,
                'webhook_url': '',
                'email': False,
                'email_to': '',
            },
            'thresholds': {
                'health_score_warning': 70,
                'health_score_critical': 50,
                'uptime_warning': 0.95,
                'uptime_critical': 0.80,
            },
            'platforms': {
                'enabled': [
                    'google_colab',
                    'kaggle',
                    'huggingface_spaces',
                    'github_codespaces',
                    'gitpod',
                    'oracle_cloud',
                    'github_models',
                    'replit',
                    'glitch',
                ],
            },
        }

    def _init_stats(self) -> Dict:
        """Initialize statistics tracking"""

        return {
            'start_time': datetime.now(),
            'total_checks': 0,
            'total_alerts': 0,
            'total_failovers': 0,
            'total_recoveries': 0,
            'platform_uptime': {},
            'alert_history': [],
        }

    def _init_alert_handlers(self) -> Dict:
        """Initialize alert handlers"""

        return {
            'console': self._alert_console,
            'log': self._alert_log,
            'webhook': self._alert_webhook,
        }

    async def start(self):
        """Start the monitoring daemon"""

        logger.info("=" * 70)
        logger.info("HEALTH MONITORING DAEMON STARTING")
        logger.info("=" * 70)
        logger.info(f"Check interval: {self.config['monitoring']['check_interval']}s")
        logger.info(f"Auto-failover: {self.config['monitoring']['enable_auto_failover']}")
        logger.info(f"Predictive alerts: {self.config['monitoring']['enable_predictive_alerts']}")
        logger.info("=" * 70)

        self.running = True

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            await self._monitoring_loop()
        except Exception as e:
            logger.error(f"Daemon error: {e}", exc_info=True)
        finally:
            await self.stop()

    async def _monitoring_loop(self):
        """Main monitoring loop"""

        check_count = 0

        while self.running:
            try:
                check_count += 1
                logger.info(f"\n{'=' * 70}")
                logger.info(f"Health Check #{check_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'=' * 70}")

                # Run health checks
                await self._run_health_checks()

                # Check for predictive failures
                if self.config['monitoring']['enable_predictive_alerts']:
                    await self._check_predictive_failures()

                # Auto-healing if enabled
                if self.config['monitoring']['enable_auto_failover']:
                    await self._auto_healing()

                # Generate and save dashboard
                self._update_dashboard()

                # Save statistics
                self._save_statistics()

                # Log summary
                self._log_summary()

                # Wait for next check
                await asyncio.sleep(self.config['monitoring']['check_interval'])

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(10)  # Brief pause before retry

    async def _run_health_checks(self):
        """Run health checks on all enabled platforms"""

        logger.info("Running health checks on all platforms...")

        # Get enabled platforms
        enabled = self.config['platforms']['enabled']

        # Run checks
        results = {}
        for platform_id in enabled:
            try:
                result = await self.health_checker.check_platform_health(platform_id)
                results[platform_id] = result
                self.stats['total_checks'] += 1

                # Update uptime stats
                if platform_id not in self.stats['platform_uptime']:
                    self.stats['platform_uptime'][platform_id] = {
                        'checks': 0,
                        'successes': 0,
                        'failures': 0,
                    }

                self.stats['platform_uptime'][platform_id]['checks'] += 1

                if result.status in [HealthStatus.EXCELLENT, HealthStatus.HEALTHY]:
                    self.stats['platform_uptime'][platform_id]['successes'] += 1
                else:
                    self.stats['platform_uptime'][platform_id]['failures'] += 1

                # Check for alerts
                await self._check_alerts(result)

            except Exception as e:
                logger.error(f"Health check failed for {platform_id}: {e}")

        # Generate report
        if results:
            report = self.health_checker.generate_health_report(results)
            logger.info(f"\n{report}")

    async def _check_predictive_failures(self):
        """Check for predictive failure warnings"""

        logger.debug("Checking for predictive failures...")

        warnings = self.orchestrator.detect_predictive_failures()

        if warnings:
            logger.warning(f"Predictive failure warnings detected: {len(warnings)}")
            for warning in warnings:
                await self._send_alert(
                    severity='WARNING',
                    title='Predictive Failure Alert',
                    message=warning,
                )

    async def _auto_healing(self):
        """Perform auto-healing actions if needed"""

        logger.debug("Checking if healing actions needed...")

        # Update rankings
        self.orchestrator.update_reliability_rankings()

        # Check each platform
        for platform_id, platform in self.orchestrator.platforms.items():
            if platform['health'].value in ['dead', 'failing']:
                logger.warning(f"Platform {platform_id} is {platform['health'].value}")

                # Attempt intelligent failover
                backup_platform = self.orchestrator.intelligent_failover(
                    platform_id,
                    task_type='general'
                )

                if backup_platform:
                    self.stats['total_failovers'] += 1
                    await self._send_alert(
                        severity='INFO',
                        title='Auto-Failover Executed',
                        message=f'Failed over from {platform_id} to {backup_platform}',
                    )

                # Attempt recovery
                if platform['health'].value == 'dead':
                    logger.info(f"Scheduling recovery for {platform_id}")
                    asyncio.create_task(self.orchestrator.attempt_recovery(platform_id))

    async def _check_alerts(self, result: HealthCheckResult):
        """Check if health result should trigger alerts"""

        thresholds = self.config['thresholds']

        # Critical health score
        if result.health_score < thresholds['health_score_critical']:
            await self._send_alert(
                severity='CRITICAL',
                title=f'Critical Health: {result.platform}',
                message=f'Health score: {result.health_score:.1f}/100\nIssues: {", ".join(result.issues)}',
            )

        # Warning health score
        elif result.health_score < thresholds['health_score_warning']:
            await self._send_alert(
                severity='WARNING',
                title=f'Degraded Health: {result.platform}',
                message=f'Health score: {result.health_score:.1f}/100\nIssues: {", ".join(result.issues)}',
            )

        # Critical uptime
        if result.uptime_percentage < thresholds['uptime_critical']:
            await self._send_alert(
                severity='CRITICAL',
                title=f'Critical Uptime: {result.platform}',
                message=f'Uptime: {result.uptime_percentage*100:.1f}%',
            )

        # Warning uptime
        elif result.uptime_percentage < thresholds['uptime_warning']:
            await self._send_alert(
                severity='WARNING',
                title=f'Low Uptime: {result.platform}',
                message=f'Uptime: {result.uptime_percentage*100:.1f}%',
            )

    async def _send_alert(self, severity: str, title: str, message: str):
        """Send alert through configured channels"""

        self.stats['total_alerts'] += 1

        alert = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'title': title,
            'message': message,
        }

        # Store in history
        self.stats['alert_history'].append(alert)

        # Keep only last 100 alerts
        if len(self.stats['alert_history']) > 100:
            self.stats['alert_history'] = self.stats['alert_history'][-100:]

        # Send through configured channels
        if self.config['alerts']['console']:
            await self._alert_console(alert)

        if self.config['alerts']['log_file']:
            await self._alert_log(alert)

        if self.config['alerts']['webhook'] and self.config['alerts']['webhook_url']:
            await self._alert_webhook(alert)

    async def _alert_console(self, alert: Dict):
        """Send alert to console"""

        severity_icons = {
            'CRITICAL': '🔴',
            'WARNING': '🟡',
            'INFO': '🔵',
        }

        icon = severity_icons.get(alert['severity'], '⚪')
        print(f"\n{icon} [{alert['severity']}] {alert['title']}")
        print(f"   {alert['message']}")

    async def _alert_log(self, alert: Dict):
        """Send alert to log file"""

        if alert['severity'] == 'CRITICAL':
            logger.critical(f"{alert['title']}: {alert['message']}")
        elif alert['severity'] == 'WARNING':
            logger.warning(f"{alert['title']}: {alert['message']}")
        else:
            logger.info(f"{alert['title']}: {alert['message']}")

    async def _alert_webhook(self, alert: Dict):
        """Send alert to webhook"""

        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config['alerts']['webhook_url'],
                    json=alert,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.debug(f"Webhook alert sent: {alert['title']}")
                    else:
                        logger.error(f"Webhook alert failed: {response.status}")
        except Exception as e:
            logger.error(f"Webhook error: {e}")

    def _update_dashboard(self):
        """Update health dashboard"""

        dashboard_path = BASE_DIR / 'platform_health_dashboard.txt'

        # Get platform rankings
        rankings = self.orchestrator.reliability_rankings

        dashboard = f"""
╔══════════════════════════════════════════════════════════════════════╗
║              PLATFORM HEALTH MONITORING DASHBOARD                    ║
║                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                          ║
╚══════════════════════════════════════════════════════════════════════╝

📊 SYSTEM STATISTICS
├─ Uptime: {self._format_uptime(self.stats['start_time'])}
├─ Total Checks: {self.stats['total_checks']}
├─ Total Alerts: {self.stats['total_alerts']}
├─ Total Failovers: {self.stats['total_failovers']}
└─ Total Recoveries: {self.stats['total_recoveries']}

🏆 PLATFORM RELIABILITY RANKINGS
"""

        # Add platform rankings
        for i, (platform_id, score) in enumerate(rankings, 1):
            platform = self.orchestrator.platforms[platform_id]
            platform_name = platform['name']

            # Health icon
            if score >= 90:
                icon = "🟢"
            elif score >= 70:
                icon = "🟢"
            elif score >= 50:
                icon = "🟡"
            elif score >= 20:
                icon = "🟠"
            else:
                icon = "🔴"

            # Uptime
            if platform_id in self.stats['platform_uptime']:
                uptime_data = self.stats['platform_uptime'][platform_id]
                uptime = (uptime_data['successes'] / max(1, uptime_data['checks'])) * 100
            else:
                uptime = 0.0

            dashboard += f"{i}. {icon} {platform_name:25} | Score: {score:5.1f} | Uptime: {uptime:5.1f}%\n"

        # Recent alerts
        dashboard += f"""
{'─' * 74}
⚠️  RECENT ALERTS (Last 10)
"""

        recent_alerts = self.stats['alert_history'][-10:]
        if recent_alerts:
            for alert in reversed(recent_alerts):
                timestamp = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
                dashboard += f"[{timestamp}] {alert['severity']:8} - {alert['title']}\n"
        else:
            dashboard += "No recent alerts\n"

        # Failover history
        dashboard += f"""
{'─' * 74}
🔄 RECENT FAILOVERS (Last 5)
"""

        recent_failovers = self.orchestrator.failover_history[-5:]
        if recent_failovers:
            for failover in reversed(recent_failovers):
                timestamp = datetime.fromisoformat(failover['timestamp']).strftime('%H:%M:%S')
                dashboard += f"[{timestamp}] {failover['from']} → {failover['to']} (score: {failover['score']:.1f})\n"
        else:
            dashboard += "No failovers yet\n"

        dashboard += f"""
{'─' * 74}
🏥 HEALTH MONITORING STATUS
├─ Check Interval: {self.config['monitoring']['check_interval']}s
├─ Auto-Failover: {'Enabled' if self.config['monitoring']['enable_auto_failover'] else 'Disabled'}
├─ Predictive Alerts: {'Enabled' if self.config['monitoring']['enable_predictive_alerts'] else 'Disabled'}
└─ Daemon Status: {'Running' if self.running else 'Stopped'}
{'─' * 74}

Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        # Save dashboard
        try:
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(dashboard)
            logger.debug(f"Dashboard updated: {dashboard_path}")
        except IOError as e:
            logger.error(f"Failed to save dashboard: {e}")

    def _save_statistics(self) -> None:
        """Save statistics to file"""

        stats_path = BASE_DIR / 'health_monitoring_stats.json'

        stats_output = {
            **self.stats,
            'start_time': self.stats['start_time'].isoformat(),
        }

        try:
            if UTILS_AVAILABLE:
                safe_write_json(stats_path, stats_output)
            else:
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats_output, f, indent=2)
        except (IOError, OSError) as e:
            logger.error(f"Failed to save statistics: {e}")

    def _log_summary(self):
        """Log monitoring summary"""

        uptime = self._format_uptime(self.stats['start_time'])
        logger.info(f"Summary: {self.stats['total_checks']} checks, "
                   f"{self.stats['total_alerts']} alerts, "
                   f"{self.stats['total_failovers']} failovers, "
                   f"uptime: {uptime}")

    def _format_uptime(self, start_time: datetime) -> str:
        """Format uptime duration"""

        delta = datetime.now() - start_time
        hours = delta.total_seconds() // 3600
        minutes = (delta.total_seconds() % 3600) // 60

        if hours >= 24:
            days = hours // 24
            hours = hours % 24
            return f"{int(days)}d {int(hours)}h {int(minutes)}m"
        else:
            return f"{int(hours)}h {int(minutes)}m"

    async def stop(self):
        """Stop the monitoring daemon"""

        logger.info("Stopping health monitoring daemon...")
        self.running = False

        # Save final statistics
        self._save_statistics()

        # Update final dashboard
        self._update_dashboard()

        logger.info("Health monitoring daemon stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""

        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False


async def main():
    """Main entry point"""

    daemon = HealthMonitoringDaemon()
    await daemon.start()


if __name__ == "__main__":
    asyncio.run(main())
