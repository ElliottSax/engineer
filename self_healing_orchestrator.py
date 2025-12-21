#!/usr/bin/env python3
"""
SELF-HEALING ORCHESTRATOR
Automatically detects and recovers from platform failures
Manages failovers, retries, and platform substitutions
Enhanced with intelligent failover and health scoring
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
from enum import Enum
from collections import deque
import statistics

# Import health checker
try:
    from platform_health_checker import PlatformHealthChecker, HealthStatus, HealthCheckResult
except ImportError:
    PlatformHealthChecker = None

# Import utilities for config and file operations
try:
    from utils.config_loader import get_config
    from utils.file_locking import safe_write_json, safe_read_json
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if PlatformHealthChecker is None:
    logger.warning("platform_health_checker not found, using basic health checks")

class PlatformHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    DEAD = "dead"
    RECOVERING = "recovering"

class SelfHealingOrchestrator:
    """Manages platform health and automatic recovery"""

    # Maximum history items to prevent memory leaks
    MAX_HISTORY_ITEMS = 100

    def __init__(self):
        # Load config
        if UTILS_AVAILABLE:
            self.MAX_HISTORY_ITEMS = get_config(
                'health_monitoring', 'history_max_items', default=100
            )

        self.platforms = self._init_platforms()
        # Use deque for automatic memory management
        self.health_history: Dict[str, deque] = {}
        self.failover_mappings = self._init_failover_mappings()
        self.recovery_strategies = self._init_recovery_strategies()
        self.incidents: deque = deque(maxlen=self.MAX_HISTORY_ITEMS)
        self.healing_actions: deque = deque(maxlen=self.MAX_HISTORY_ITEMS)

        # Enhanced features
        self.health_checker = PlatformHealthChecker() if PlatformHealthChecker else None
        self.platform_scores: Dict[str, float] = {}
        self.reliability_rankings: List[Tuple[str, float]] = []
        self.failover_history: deque = deque(maxlen=self.MAX_HISTORY_ITEMS)
        self.recovery_attempts: Dict[str, int] = {}
        self.backoff_timers: Dict[str, float] = {}

    def _init_platforms(self) -> Dict:
        """Initialize platform configurations"""
        return {
            "google_colab": {
                "name": "Google Colab",
                "health": PlatformHealth.HEALTHY,
                "last_check": datetime.now(),
                "failure_count": 0,
                "success_count": 0,
                "capabilities": ["gpu", "tpu", "notebook"],
                "alternatives": ["kaggle", "oracle_cloud"]
            },
            "kaggle": {
                "name": "Kaggle",
                "health": PlatformHealth.HEALTHY,
                "last_check": datetime.now(),
                "failure_count": 0,
                "success_count": 0,
                "capabilities": ["gpu", "tpu", "notebook"],
                "alternatives": ["google_colab", "oracle_cloud"]
            },
            "huggingface_spaces": {
                "name": "HuggingFace Spaces",
                "health": PlatformHealth.HEALTHY,
                "last_check": datetime.now(),
                "failure_count": 0,
                "success_count": 0,
                "capabilities": ["hosting", "api", "gradio"],
                "alternatives": ["oracle_cloud", "github_pages"]
            },
            "github_codespaces": {
                "name": "GitHub Codespaces",
                "health": PlatformHealth.HEALTHY,
                "last_check": datetime.now(),
                "failure_count": 0,
                "success_count": 0,
                "capabilities": ["development", "cpu", "ide"],
                "alternatives": ["gitpod", "oracle_cloud"]
            },
            "gitpod": {
                "name": "Gitpod",
                "health": PlatformHealth.HEALTHY,
                "last_check": datetime.now(),
                "failure_count": 0,
                "success_count": 0,
                "capabilities": ["development", "cpu", "ide"],
                "alternatives": ["github_codespaces", "oracle_cloud"]
            },
            "oracle_cloud": {
                "name": "Oracle Cloud",
                "health": PlatformHealth.HEALTHY,
                "last_check": datetime.now(),
                "failure_count": 0,
                "success_count": 0,
                "capabilities": ["24/7", "cpu", "hosting", "api"],
                "alternatives": ["huggingface_spaces"]  # Limited alternatives for 24/7
            },
            "github_models": {
                "name": "GitHub Models API",
                "health": PlatformHealth.HEALTHY,
                "last_check": datetime.now(),
                "failure_count": 0,
                "success_count": 0,
                "capabilities": ["llm", "api", "inference"],
                "alternatives": ["deepseek", "alibaba_cloud"]
            }
        }

    def _init_failover_mappings(self) -> Dict:
        """Define failover strategies for different task types"""
        return {
            "gpu_tasks": {
                "primary": ["google_colab", "kaggle"],
                "secondary": ["oracle_cloud"],  # Can run CPU version
                "tertiary": ["github_codespaces"]  # Last resort
            },
            "hosting_tasks": {
                "primary": ["huggingface_spaces", "oracle_cloud"],
                "secondary": ["github_pages", "glitch"],
                "tertiary": ["replit"]
            },
            "development_tasks": {
                "primary": ["github_codespaces", "gitpod"],
                "secondary": ["oracle_cloud"],
                "tertiary": ["local_fallback"]
            },
            "api_tasks": {
                "primary": ["github_models", "huggingface_spaces"],
                "secondary": ["oracle_cloud"],
                "tertiary": ["deepseek"]
            }
        }

    def _init_recovery_strategies(self) -> Dict:
        """Define recovery strategies for different failure types"""
        return {
            "timeout": {
                "actions": ["retry_with_backoff", "increase_timeout", "switch_platform"],
                "max_retries": 3
            },
            "rate_limit": {
                "actions": ["exponential_backoff", "distribute_load", "queue_tasks"],
                "max_retries": 5
            },
            "quota_exceeded": {
                "actions": ["switch_platform", "wait_for_reset", "use_alternative"],
                "max_retries": 1
            },
            "platform_down": {
                "actions": ["immediate_failover", "notify_admin", "activate_backup"],
                "max_retries": 0
            },
            "unknown_error": {
                "actions": ["retry_once", "switch_platform", "escalate"],
                "max_retries": 2
            }
        }

    async def monitor_health_continuous(self):
        """Continuously monitor platform health"""

        logger.info("🏥 Starting continuous health monitoring...")

        while True:
            health_report = await self.check_all_platforms()

            # Take healing actions if needed
            await self.heal_unhealthy_platforms(health_report)

            # Update dashboard
            self.update_health_dashboard()

            # Wait before next check
            await asyncio.sleep(30)  # Check every 30 seconds

    async def check_all_platforms(self) -> Dict:
        """Check health of all platforms"""

        health_report = {
            "timestamp": datetime.now().isoformat(),
            "platforms": {},
            "overall_health": "healthy",
            "issues": []
        }

        unhealthy_count = 0

        for platform_id, platform in self.platforms.items():
            health = await self.check_platform_health(platform_id)
            health_report["platforms"][platform_id] = health

            if health["status"] != PlatformHealth.HEALTHY:
                unhealthy_count += 1
                health_report["issues"].append({
                    "platform": platform_id,
                    "status": health["status"],
                    "reason": health.get("reason", "Unknown")
                })

        # Determine overall health
        if unhealthy_count == 0:
            health_report["overall_health"] = "healthy"
        elif unhealthy_count < len(self.platforms) / 2:
            health_report["overall_health"] = "degraded"
        else:
            health_report["overall_health"] = "critical"

        return health_report

    async def check_platform_health(self, platform_id: str) -> Dict:
        """Check health of a specific platform"""

        platform = self.platforms[platform_id]
        health_check = {
            "platform": platform_id,
            "timestamp": datetime.now().isoformat(),
            "status": PlatformHealth.HEALTHY,
            "metrics": {}
        }

        try:
            # Simulate health check (in production, would actually ping platform)
            response_time = await self._ping_platform(platform_id)

            if response_time < 0:  # Platform is down
                health_check["status"] = PlatformHealth.DEAD
                health_check["reason"] = "Platform not responding"
                platform["failure_count"] += 1
            elif response_time > 5000:  # Slow response
                health_check["status"] = PlatformHealth.DEGRADED
                health_check["reason"] = f"High latency: {response_time}ms"
            else:
                health_check["status"] = PlatformHealth.HEALTHY
                platform["success_count"] += 1

            health_check["metrics"] = {
                "response_time": response_time,
                "failure_rate": platform["failure_count"] / max(1, platform["failure_count"] + platform["success_count"]),
                "uptime": platform["success_count"] / max(1, platform["failure_count"] + platform["success_count"])
            }

            # Update platform health status
            platform["health"] = health_check["status"]
            platform["last_check"] = datetime.now()

        except Exception as e:
            health_check["status"] = PlatformHealth.FAILING
            health_check["reason"] = str(e)
            platform["failure_count"] += 1

        return health_check

    async def _ping_platform(self, platform_id: str) -> float:
        """Ping a platform to check if it's responsive"""

        # Simulate platform ping (in production, would actually test platform)
        import random

        if platform_id == "oracle_cloud":
            # Oracle is always reliable in our simulation
            return random.uniform(100, 500)
        elif random.random() > 0.9:  # 10% chance of failure
            return -1  # Platform down
        elif random.random() > 0.8:  # 10% chance of slow response
            return random.uniform(5000, 10000)
        else:
            return random.uniform(100, 2000)  # Normal response

    async def heal_unhealthy_platforms(self, health_report: Dict):
        """Attempt to heal unhealthy platforms"""

        for issue in health_report["issues"]:
            platform_id = issue["platform"]
            status = issue["status"]

            logger.warning(f"⚠️ Platform {platform_id} is {status.value}")

            # Determine healing strategy
            if status == PlatformHealth.DEAD:
                await self.handle_dead_platform(platform_id)
            elif status == PlatformHealth.FAILING:
                await self.handle_failing_platform(platform_id)
            elif status == PlatformHealth.DEGRADED:
                await self.handle_degraded_platform(platform_id)

    async def handle_dead_platform(self, platform_id: str):
        """Handle a completely dead platform"""

        logger.error(f"💀 Platform {platform_id} is DEAD - initiating failover")

        # Record incident
        incident = {
            "id": f"incident_{datetime.now().timestamp()}",
            "platform": platform_id,
            "type": "platform_dead",
            "timestamp": datetime.now().isoformat(),
            "actions": []
        }

        # 1. Immediate failover to alternatives
        platform = self.platforms[platform_id]
        alternatives = platform.get("alternatives", [])

        if alternatives:
            for alt_platform in alternatives:
                if self.platforms[alt_platform]["health"] == PlatformHealth.HEALTHY:
                    logger.info(f"✅ Failover to {alt_platform}")
                    incident["actions"].append(f"Failover to {alt_platform}")

                    # Migrate tasks
                    await self.migrate_tasks(platform_id, alt_platform)
                    break
        else:
            logger.error(f"❌ No healthy alternatives for {platform_id}")

        # 2. Mark platform as recovering
        platform["health"] = PlatformHealth.RECOVERING

        # 3. Schedule recovery check
        asyncio.create_task(self.attempt_recovery(platform_id))

        self.incidents.append(incident)

    async def handle_failing_platform(self, platform_id: str):
        """Handle a failing platform"""

        logger.warning(f"⚠️ Platform {platform_id} is FAILING - attempting recovery")

        platform = self.platforms[platform_id]

        # Apply recovery strategy
        recovery = self.recovery_strategies.get("unknown_error", {})

        for action in recovery["actions"]:
            if action == "retry_once":
                await asyncio.sleep(5)
                health = await self.check_platform_health(platform_id)
                if health["status"] == PlatformHealth.HEALTHY:
                    logger.info(f"✅ Platform {platform_id} recovered after retry")
                    return
            elif action == "switch_platform":
                # Reduce load on failing platform
                await self.reduce_platform_load(platform_id)
            elif action == "escalate":
                logger.error(f"🚨 Escalating issue with {platform_id}")

    async def handle_degraded_platform(self, platform_id: str):
        """Handle a degraded platform"""

        logger.warning(f"⚡ Platform {platform_id} is DEGRADED - optimizing")

        # Reduce load
        await self.reduce_platform_load(platform_id, reduction=0.5)

        # Apply optimizations
        self.healing_actions.append({
            "platform": platform_id,
            "action": "reduce_load",
            "timestamp": datetime.now().isoformat()
        })

    async def migrate_tasks(self, from_platform: str, to_platform: str):
        """Migrate tasks from one platform to another"""

        logger.info(f"📦 Migrating tasks: {from_platform} → {to_platform}")

        # In production, this would actually move running tasks
        # For now, we simulate the migration

        migration = {
            "from": from_platform,
            "to": to_platform,
            "timestamp": datetime.now().isoformat(),
            "tasks_migrated": 0
        }

        # Simulate task migration
        await asyncio.sleep(2)
        migration["tasks_migrated"] = 10  # Simulated

        logger.info(f"✅ Migrated {migration['tasks_migrated']} tasks")

        return migration

    async def reduce_platform_load(self, platform_id: str, reduction: float = 0.75):
        """Reduce load on a platform"""

        logger.info(f"📉 Reducing load on {platform_id} by {reduction*100:.0f}%")

        # In production, this would actually reduce task allocation
        # For now, we track the action

        self.healing_actions.append({
            "platform": platform_id,
            "action": "reduce_load",
            "reduction": reduction,
            "timestamp": datetime.now().isoformat()
        })

    async def attempt_recovery(self, platform_id: str):
        """Attempt to recover a dead platform with exponential backoff"""

        logger.info(f"🔄 Attempting recovery for {platform_id}")

        # Track recovery attempts
        if platform_id not in self.recovery_attempts:
            self.recovery_attempts[platform_id] = 0

        self.recovery_attempts[platform_id] += 1
        attempt_count = self.recovery_attempts[platform_id]

        # Calculate exponential backoff: 1min, 2min, 4min, 8min, 16min, max 30min
        backoff_duration = min(60 * (2 ** (attempt_count - 1)), 1800)
        self.backoff_timers[platform_id] = backoff_duration

        logger.info(f"Recovery attempt #{attempt_count} - waiting {backoff_duration}s before retry")

        # Wait before attempting recovery
        await asyncio.sleep(backoff_duration)

        # Try to recover
        max_retries = 5
        for attempt in range(max_retries):
            health = await self.check_platform_health(platform_id)

            if health["status"] == PlatformHealth.HEALTHY:
                logger.info(f"🎉 Platform {platform_id} recovered after {attempt_count} attempts!")
                self.platforms[platform_id]["health"] = PlatformHealth.HEALTHY
                # Reset recovery counter
                self.recovery_attempts[platform_id] = 0
                self.backoff_timers.pop(platform_id, None)
                return

            logger.info(f"Recovery attempt {attempt+1}/{max_retries} failed")
            await asyncio.sleep(30)

        logger.error(f"❌ Failed to recover {platform_id} (attempt #{attempt_count})")
        self.platforms[platform_id]["health"] = PlatformHealth.DEAD

        # If too many attempts, increase backoff significantly
        if attempt_count >= 10:
            logger.warning(f"Platform {platform_id} has failed {attempt_count} times - reducing recovery frequency")

    def calculate_platform_health_score(self, platform_id: str) -> float:
        """Calculate comprehensive health score for a platform"""

        if not self.health_checker:
            # Fallback to simple calculation
            platform = self.platforms.get(platform_id, {})
            success = platform.get("success_count", 0)
            failure = platform.get("failure_count", 0)
            total = success + failure

            if total == 0:
                return 100.0

            return (success / total) * 100

        # Use advanced health checker
        if platform_id in self.health_checker.check_history:
            history = self.health_checker.check_history[platform_id]
            if history:
                # Return most recent health score
                return history[-1].health_score

        return 50.0  # Unknown/new platform

    def update_reliability_rankings(self):
        """Update platform rankings by reliability"""

        rankings = []

        for platform_id in self.platforms.keys():
            score = self.calculate_platform_health_score(platform_id)
            self.platform_scores[platform_id] = score
            rankings.append((platform_id, score))

        # Sort by score (highest first)
        rankings.sort(key=lambda x: x[1], reverse=True)
        self.reliability_rankings = rankings

        logger.info("\n📊 Platform Reliability Rankings:")
        for i, (platform_id, score) in enumerate(rankings, 1):
            platform_name = self.platforms[platform_id]["name"]
            logger.info(f"  {i}. {platform_name:25} - {score:.1f}/100")

    def intelligent_failover(self, failed_platform: str, task_type: str = None) -> Optional[str]:
        """Intelligently select best failover platform based on health scores"""

        logger.info(f"🔄 Initiating intelligent failover for {failed_platform}")

        # Update rankings first
        self.update_reliability_rankings()

        # Get candidate platforms
        candidates = []

        # If task type specified, use failover mappings
        if task_type and task_type in self.failover_mappings:
            for priority in ["primary", "secondary", "tertiary"]:
                platforms = self.failover_mappings[task_type].get(priority, [])
                for p in platforms:
                    if p != failed_platform and p in self.platforms:
                        candidates.append(p)
        else:
            # Use platform alternatives
            alternatives = self.platforms[failed_platform].get("alternatives", [])
            candidates.extend([p for p in alternatives if p in self.platforms])

        if not candidates:
            # Fallback: any healthy platform
            candidates = list(self.platforms.keys())

        # Filter by health and score
        healthy_candidates = []
        for p in candidates:
            if p == failed_platform:
                continue

            platform = self.platforms[p]
            score = self.platform_scores.get(p, 50.0)

            # Only consider platforms with good health
            if platform["health"] == PlatformHealth.HEALTHY and score > 60:
                healthy_candidates.append((p, score))

        if not healthy_candidates:
            logger.error("❌ No healthy platforms available for failover")
            return None

        # Sort by score and select best
        healthy_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_platform = healthy_candidates[0][0]
        selected_score = healthy_candidates[0][1]

        logger.info(f"✅ Selected {selected_platform} for failover (score: {selected_score:.1f})")

        # Record failover event
        self.failover_history.append({
            "timestamp": datetime.now().isoformat(),
            "from": failed_platform,
            "to": selected_platform,
            "reason": "intelligent_failover",
            "score": selected_score,
        })

        return selected_platform

    def detect_predictive_failures(self) -> List[str]:
        """Detect platforms likely to fail soon"""

        if not self.health_checker:
            return []

        warnings = []

        for platform_id in self.platforms.keys():
            warning = self.health_checker.detect_predictive_failures(platform_id)
            if warning:
                warnings.append(f"{platform_id}: {warning}")
                logger.warning(f"⚠️ Predictive failure alert - {platform_id}: {warning}")

        return warnings

    def update_health_dashboard(self):
        """Update health dashboard"""

        dashboard = """
╔══════════════════════════════════════════════════════════════╗
║              🏥 SELF-HEALING ORCHESTRATOR                    ║
╚══════════════════════════════════════════════════════════════╝

PLATFORM HEALTH STATUS
───────────────────────────────────────────────────────────────
"""

        for platform_id, platform in self.platforms.items():
            health_icon = {
                PlatformHealth.HEALTHY: "🟢",
                PlatformHealth.DEGRADED: "🟡",
                PlatformHealth.FAILING: "🟠",
                PlatformHealth.DEAD: "🔴",
                PlatformHealth.RECOVERING: "🔵"
            }.get(platform["health"], "⚪")

            uptime = platform["success_count"] / max(1, platform["failure_count"] + platform["success_count"])

            dashboard += f"{health_icon} {platform['name']:20} | Uptime: {uptime*100:.1f}% | Status: {platform['health'].value}\n"

        dashboard += f"""
───────────────────────────────────────────────────────────────

RECENT INCIDENTS: {len(self.incidents)}
HEALING ACTIONS: {len(self.healing_actions)}
───────────────────────────────────────────────────────────────
"""

        # Save dashboard
        with open("health_dashboard.txt", 'w') as f:
            f.write(dashboard)

        return dashboard

    def get_platform_alternatives(self, platform_id: str, task_type: str) -> List[str]:
        """Get alternative platforms for a specific task type"""

        failover_map = self.failover_mappings.get(task_type, {})
        alternatives = []

        for priority in ["primary", "secondary", "tertiary"]:
            platforms = failover_map.get(priority, [])
            for p in platforms:
                if p != platform_id and self.platforms.get(p, {}).get("health") == PlatformHealth.HEALTHY:
                    alternatives.append(p)

        return alternatives

    async def run_self_healing_demo(self):
        """Run a demo of self-healing capabilities"""

        logger.info("=" * 70)
        logger.info("🏥 SELF-HEALING ORCHESTRATOR DEMO")
        logger.info("=" * 70)

        # Check initial health
        health_report = await self.check_all_platforms()

        logger.info(f"Overall system health: {health_report['overall_health']}")

        # Simulate some failures
        logger.info("\n🔥 Simulating platform failures...")

        self.platforms["google_colab"]["health"] = PlatformHealth.DEAD
        self.platforms["gitpod"]["health"] = PlatformHealth.DEGRADED

        # Run healing
        await self.heal_unhealthy_platforms({
            "issues": [
                {"platform": "google_colab", "status": PlatformHealth.DEAD},
                {"platform": "gitpod", "status": PlatformHealth.DEGRADED}
            ]
        })

        # Show dashboard
        dashboard = self.update_health_dashboard()
        print(dashboard)

        logger.info("\n✅ Self-healing demonstration complete!")

async def main():
    """Main entry point"""
    orchestrator = SelfHealingOrchestrator()

    # Run demo
    await orchestrator.run_self_healing_demo()

    # Optionally run continuous monitoring
    # await orchestrator.monitor_health_continuous()

if __name__ == "__main__":
    asyncio.run(main())