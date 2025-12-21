#!/usr/bin/env python3
"""
Hugging Face Space: Autocoder Monitoring Dashboard
Unified monitoring for all autonomous workers

This dashboard aggregates data from all worker datasets and provides
real-time monitoring, analytics, and insights.
"""

import gradio as gr
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
import os

# Hugging Face integration
from huggingface_hub import HfApi
from datasets import load_dataset

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "autocoder")

# Dataset names
ANALYSIS_DATASET = f"{HF_USERNAME}/autocoder-analysis-results"
TEST_DATASET = f"{HF_USERNAME}/autocoder-test-planning-results"
IMPL_DATASET = f"{HF_USERNAME}/autocoder-implementation-results"

class MonitoringDashboard:
    """Unified Monitoring Dashboard for All Workers"""

    def __init__(self):
        self.hf_api = HfApi()
        self.last_update = None
        self.system_status = "Initializing..."

        # Worker status tracking
        self.workers = {
            "analysis": {
                "name": "Analysis Worker 🤖",
                "dataset": ANALYSIS_DATASET,
                "status": "Unknown",
                "iterations": 0,
                "last_updated": None
            },
            "test_planning": {
                "name": "Test Planning Worker 🧪",
                "dataset": TEST_DATASET,
                "status": "Unknown",
                "iterations": 0,
                "last_updated": None
            },
            "implementation": {
                "name": "Implementation Worker ⚙️",
                "dataset": IMPL_DATASET,
                "status": "Unknown",
                "iterations": 0,
                "last_updated": None
            }
        }

        # Aggregated insights
        self.total_iterations = 0
        self.insights = {
            "failure_modes": 0,
            "integration_points": 0,
            "edge_cases": 0,
            "tests_specified": 0,
            "implementations": 0
        }

    def check_worker_status(self, worker_name: str, dataset_name: str) -> Dict:
        """Check status of a specific worker"""
        try:
            # Try to load dataset
            dataset = load_dataset(dataset_name, split="train")

            if len(dataset) > 0:
                # Get latest entry
                latest = dataset[-1]

                return {
                    "status": "✅ Running",
                    "iterations": len(dataset),
                    "last_updated": latest.get("timestamp", "Unknown") if isinstance(latest, dict) else "Unknown",
                    "health": "Healthy"
                }
            else:
                return {
                    "status": "⚠️ No Data",
                    "iterations": 0,
                    "last_updated": None,
                    "health": "Waiting"
                }

        except Exception as e:
            return {
                "status": "❌ Offline",
                "iterations": 0,
                "last_updated": None,
                "health": f"Error: {str(e)[:50]}"
            }

    def get_system_overview(self) -> Dict:
        """Get overall system status"""

        if not HF_TOKEN:
            return {
                "status": "⚠️ No HF_TOKEN configured",
                "message": "Configure HF_TOKEN to enable monitoring",
                "workers_online": 0,
                "total_iterations": 0
            }

        # Check all workers
        workers_online = 0
        total_iterations = 0

        for worker_key, worker_info in self.workers.items():
            status = self.check_worker_status(worker_key, worker_info["dataset"])
            worker_info.update(status)

            if status["status"] == "✅ Running":
                workers_online += 1

            total_iterations += status["iterations"]

        self.total_iterations = total_iterations
        self.last_update = datetime.now().isoformat()

        # Determine overall system status
        if workers_online == 3:
            system_status = "🟢 All Systems Operational"
        elif workers_online >= 1:
            system_status = f"🟡 Partial Operation ({workers_online}/3 workers)"
        else:
            system_status = "🔴 System Offline"

        self.system_status = system_status

        return {
            "status": system_status,
            "workers_online": workers_online,
            "total_workers": 3,
            "total_iterations": total_iterations,
            "last_update": self.last_update
        }

    def get_worker_details(self):
        """Get detailed status for each worker"""
        # Return as list of lists for Gradio Dataframe
        details = []

        for worker_key, worker_info in self.workers.items():
            details.append([
                worker_info["name"],
                worker_info.get("status", "Unknown"),
                worker_info.get("iterations", 0),
                worker_info.get("last_updated") or "Never",
                worker_info.get("health", "Unknown")
            ])

        return details

    def get_aggregated_insights(self) -> Dict:
        """Get combined insights from all workers"""

        insights = {
            "Total Analysis Iterations": 0,
            "Total Test Planning Iterations": 0,
            "Total Implementation Iterations": 0,
            "Combined Iterations": self.total_iterations
        }

        if not HF_TOKEN:
            return insights

        # Try to get specific metrics from each worker
        try:
            # Analysis worker
            analysis_data = load_dataset(ANALYSIS_DATASET, split="train")
            insights["Total Analysis Iterations"] = len(analysis_data)
        except (FileNotFoundError, ValueError, ConnectionError):
            pass  # Dataset not available yet

        try:
            # Test planning worker
            test_data = load_dataset(TEST_DATASET, split="train")
            insights["Total Test Planning Iterations"] = len(test_data)

            # Try to extract specific metrics
            if len(test_data) > 0:
                latest = test_data[-1]
                if isinstance(latest, dict):
                    cumulative = latest.get("cumulative_insights", {})
                    insights["Failure Modes Identified"] = cumulative.get("failure_modes_identified", 0)
                    insights["Integration Points Mapped"] = cumulative.get("integration_points_mapped", 0)
                    insights["Edge Cases Discovered"] = cumulative.get("edge_cases_discovered", 0)
        except (FileNotFoundError, ValueError, ConnectionError, IndexError):
            pass  # Dataset not available yet

        try:
            # Implementation worker
            impl_data = load_dataset(IMPL_DATASET, split="train")
            insights["Total Implementation Iterations"] = len(impl_data)

            if len(impl_data) > 0:
                latest = impl_data[-1]
                if isinstance(latest, dict):
                    cumulative = latest.get("cumulative_progress", {})
                    insights["Implementations Generated"] = cumulative.get("implementations_generated", 0)
        except (FileNotFoundError, ValueError, ConnectionError, IndexError):
            pass  # Dataset not available yet

        return insights

    def get_latest_findings(self) -> str:
        """Get latest findings from all workers"""

        findings = "# Latest Worker Findings\n\n"

        if not HF_TOKEN:
            return "Configure HF_TOKEN to view findings"

        # Analysis Worker
        try:
            analysis_data = load_dataset(ANALYSIS_DATASET, split="train")
            if len(analysis_data) > 0:
                latest = analysis_data[-1]
                findings += f"## Analysis Worker (Latest)\n"
                findings += f"```json\n{json.dumps(latest, indent=2)[:500]}...\n```\n\n"
        except Exception as e:
            findings += f"## Analysis Worker\nError loading: {e}\n\n"

        # Test Planning Worker
        try:
            test_data = load_dataset(TEST_DATASET, split="train")
            if len(test_data) > 0:
                latest = test_data[-1]
                findings += f"## Test Planning Worker (Latest)\n"
                findings += f"```json\n{json.dumps(latest, indent=2)[:500]}...\n```\n\n"
        except Exception as e:
            findings += f"## Test Planning Worker\nError loading: {e}\n\n"

        # Implementation Worker
        try:
            impl_data = load_dataset(IMPL_DATASET, split="train")
            if len(impl_data) > 0:
                latest = impl_data[-1]
                findings += f"## Implementation Worker (Latest)\n"
                findings += f"```json\n{json.dumps(latest, indent=2)[:500]}...\n```\n\n"
        except Exception as e:
            findings += f"## Implementation Worker\nError loading: {e}\n\n"

        return findings

# Global dashboard instance
dashboard = MonitoringDashboard()

# Gradio Interface Functions
def get_system_status():
    """Get current system status"""
    overview = dashboard.get_system_overview()
    return {
        "System Status": overview["status"],
        "Workers Online": f"{overview.get('workers_online', 0)}/{overview.get('total_workers', 3)}",
        "Total Iterations": overview.get("total_iterations", 0),
        "Last Update": overview.get("last_update", "Never")
    }

def get_worker_table():
    """Get worker status table"""
    return dashboard.get_worker_details()

def get_insights():
    """Get aggregated insights"""
    return dashboard.get_aggregated_insights()

def get_findings():
    """Get latest findings"""
    return dashboard.get_latest_findings()

# Build Gradio UI
with gr.Blocks(title="Autocoder Monitoring Dashboard") as demo:
    gr.Markdown("""
    # 📊 Autocoder Monitoring Dashboard

    **Unified 24/7 Worker Monitoring**

    Real-time monitoring and analytics for all autonomous workers.

    ## System Overview
    """)

    system_status = gr.JSON(label="System Status", value=get_system_status)

    gr.Markdown("## Worker Status")
    worker_table = gr.Dataframe(
        label="Individual Workers",
        value=get_worker_table,
        headers=["Worker", "Status", "Iterations", "Last Updated", "Health"]
    )

    gr.Markdown("## Aggregated Insights")
    insights_display = gr.JSON(label="Combined Metrics", value=get_insights)

    gr.Markdown("## Latest Findings")
    findings_display = gr.Textbox(
        label="Recent Worker Findings",
        value=get_findings,
        lines=20
    )

    refresh_btn = gr.Button("🔄 Refresh All Data", variant="primary")

    gr.Markdown("""
    ## Workers Monitored

    - **Analysis Worker 🤖** - Continuous code analysis
    - **Test Planning Worker 🧪** - Test requirement analysis
    - **Implementation Worker ⚙️** - Code implementation

    ## Configuration

    Set environment variables:
    - `HF_TOKEN` - Your Hugging Face access token
    - `HF_USERNAME` - Your HF username (default: autocoder)

    ## Datasets

    - `autocoder-analysis-results`
    - `autocoder-test-planning-results`
    - `autocoder-implementation-results`

    ## Update Frequency

    - **System Status**: Every 10 seconds
    - **Worker Details**: Every 10 seconds
    - **Insights**: Every 30 seconds
    - **Findings**: Every 30 seconds

    ## Support

    For issues, check individual worker logs or visit the main repository.
    """)

    # Button actions
    refresh_btn.click(
        fn=lambda: (get_system_status(), get_worker_table(), get_insights(), get_findings()),
        outputs=[system_status, worker_table, insights_display, findings_display]
    )

    # Auto-refresh
    demo.load(fn=get_system_status, outputs=system_status, every=10)
    demo.load(fn=get_worker_table, outputs=worker_table, every=10)
    demo.load(fn=get_insights, outputs=insights_display, every=30)
    demo.load(fn=get_findings, outputs=findings_display, every=30)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
