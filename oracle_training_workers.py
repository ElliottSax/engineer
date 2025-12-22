#!/usr/bin/env python3
"""
Oracle Cloud Training Workers for Autocoder

Deploys training workers on Oracle Cloud to use $300 credit.

Instance Options:
- VM.Standard.A1.Flex (FREE tier): 4 OCPU, 24GB RAM - Always Free
- VM.Standard.E4.Flex: Flexible AMD - $0.01/OCPU/hr
- VM.GPU.A10.1: GPU instance - For heavy training

This script:
1. Creates/manages OCI compute instances
2. Deploys training workers on instances
3. Monitors training progress
4. Auto-scales based on workload
"""

import os
import json
import asyncio
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# OCI CONFIGURATION
# =============================================================================

@dataclass
class OCIConfig:
    """Oracle Cloud configuration"""
    tenancy_ocid: str = ""
    user_ocid: str = ""
    region: str = "us-chicago-1"
    compartment_ocid: str = ""
    fingerprint: str = ""
    key_file: str = "~/.oci/oci_api_key.pem"

    @classmethod
    def from_env(cls):
        return cls(
            tenancy_ocid=os.getenv("OCI_TENANCY_OCID", ""),
            user_ocid=os.getenv("OCI_USER_OCID", ""),
            region=os.getenv("OCI_REGION", "us-chicago-1"),
            compartment_ocid=os.getenv("OCI_COMPARTMENT_OCID", ""),
            fingerprint=os.getenv("OCI_FINGERPRINT", ""),
            key_file=os.getenv("OCI_KEY_FILE", "~/.oci/oci_api_key.pem")
        )


@dataclass
class InstanceSpec:
    """Compute instance specification"""
    name: str
    shape: str
    ocpus: int
    memory_gb: int
    is_free_tier: bool
    hourly_cost: float


# Available instance types
INSTANCE_SPECS = {
    "free": InstanceSpec(
        name="Training Worker (Free)",
        shape="VM.Standard.A1.Flex",
        ocpus=4,
        memory_gb=24,
        is_free_tier=True,
        hourly_cost=0.0
    ),
    "small": InstanceSpec(
        name="Training Worker (Small)",
        shape="VM.Standard.E4.Flex",
        ocpus=2,
        memory_gb=16,
        is_free_tier=False,
        hourly_cost=0.02
    ),
    "medium": InstanceSpec(
        name="Training Worker (Medium)",
        shape="VM.Standard.E4.Flex",
        ocpus=4,
        memory_gb=32,
        is_free_tier=False,
        hourly_cost=0.04
    ),
    "large": InstanceSpec(
        name="Training Worker (Large)",
        shape="VM.Standard.E4.Flex",
        ocpus=8,
        memory_gb=64,
        is_free_tier=False,
        hourly_cost=0.08
    ),
    "gpu": InstanceSpec(
        name="Training Worker (GPU)",
        shape="VM.GPU.A10.1",
        ocpus=15,
        memory_gb=240,
        is_free_tier=False,
        hourly_cost=2.50
    )
}


# =============================================================================
# WORKER DEPLOYMENT SCRIPT
# =============================================================================

WORKER_SETUP_SCRIPT = '''#!/bin/bash
# Oracle Cloud Training Worker Setup Script

set -e

echo "=== Setting up Training Worker ==="

# Update system
sudo apt-get update
sudo apt-get install -y python3 python3-pip git

# Clone the repository
cd /home/ubuntu
if [ ! -d "engineer" ]; then
    git clone https://github.com/ElliottSax/engineer.git
fi
cd engineer

# Install dependencies
pip3 install requests asyncio aiohttp

# Set environment variables
export DEEPSEEK_API_KEY="{deepseek_key}"
export GITHUB_TOKEN="{github_token}"
export HF_TOKEN="{hf_token}"

# Create workspace
mkdir -p /tmp/training_workspace
mkdir -p training_output

# Start continuous training
echo "Starting continuous training worker..."
nohup python3 multi_provider_trainer.py \\
    --workers 1 \\
    --iterations 1000 \\
    --delay 0.5 \\
    > training.log 2>&1 &

echo "Worker started. Check training.log for progress."
echo "PID: $!"
'''


# =============================================================================
# OCI MANAGER
# =============================================================================

class OCIManager:
    """Manages Oracle Cloud resources"""

    def __init__(self, config: OCIConfig):
        self.config = config
        self.instances: Dict[str, Dict] = {}

    def _run_oci_cmd(self, cmd: List[str]) -> Optional[Dict]:
        """Run OCI CLI command"""
        try:
            result = subprocess.run(
                ["oci"] + cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                return json.loads(result.stdout) if result.stdout else {}
            else:
                logger.error(f"OCI command failed: {result.stderr}")
                return None

        except FileNotFoundError:
            logger.error("OCI CLI not installed. Install with: pip install oci-cli")
            return None
        except Exception as e:
            logger.error(f"OCI command error: {e}")
            return None

    def check_oci_cli(self) -> bool:
        """Check if OCI CLI is available and configured"""
        try:
            result = subprocess.run(
                ["oci", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def list_instances(self) -> List[Dict]:
        """List all compute instances"""
        result = self._run_oci_cmd([
            "compute", "instance", "list",
            "--compartment-id", self.config.compartment_ocid,
            "--lifecycle-state", "RUNNING"
        ])

        if result and "data" in result:
            return result["data"]
        return []

    def create_instance(
        self,
        name: str,
        spec: InstanceSpec,
        ssh_key: str,
        subnet_id: str,
        image_id: str
    ) -> Optional[str]:
        """Create a new compute instance"""

        shape_config = json.dumps({
            "ocpus": spec.ocpus,
            "memoryInGBs": spec.memory_gb
        })

        result = self._run_oci_cmd([
            "compute", "instance", "launch",
            "--compartment-id", self.config.compartment_ocid,
            "--availability-domain", f"{self.config.region}-AD-1",
            "--display-name", name,
            "--shape", spec.shape,
            "--shape-config", shape_config,
            "--image-id", image_id,
            "--subnet-id", subnet_id,
            "--assign-public-ip", "true",
            "--ssh-authorized-keys-file", ssh_key,
            "--wait-for-state", "RUNNING"
        ])

        if result and "data" in result:
            instance_id = result["data"]["id"]
            self.instances[name] = result["data"]
            logger.info(f"Created instance: {name} ({instance_id})")
            return instance_id

        return None

    def get_instance_ip(self, instance_id: str) -> Optional[str]:
        """Get public IP of an instance"""
        result = self._run_oci_cmd([
            "compute", "instance", "list-vnics",
            "--instance-id", instance_id
        ])

        if result and "data" in result:
            for vnic in result["data"]:
                if vnic.get("public-ip"):
                    return vnic["public-ip"]
        return None

    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an instance"""
        result = self._run_oci_cmd([
            "compute", "instance", "terminate",
            "--instance-id", instance_id,
            "--force"
        ])
        return result is not None


# =============================================================================
# TRAINING WORKER MANAGER
# =============================================================================

class OracleTrainingManager:
    """Manages training workers on Oracle Cloud"""

    def __init__(self):
        self.config = OCIConfig.from_env()
        self.oci = OCIManager(self.config)
        self.workers: Dict[str, Dict] = {}
        self.output_dir = Path("oci_training_output")
        self.output_dir.mkdir(exist_ok=True)

    def estimate_costs(self, spec_name: str, hours: int) -> Dict:
        """Estimate costs for running workers"""
        spec = INSTANCE_SPECS.get(spec_name)
        if not spec:
            return {"error": "Unknown spec"}

        cost = spec.hourly_cost * hours

        return {
            "spec": spec_name,
            "shape": spec.shape,
            "ocpus": spec.ocpus,
            "memory_gb": spec.memory_gb,
            "hours": hours,
            "hourly_cost": f"${spec.hourly_cost:.2f}",
            "total_cost": f"${cost:.2f}",
            "is_free": spec.is_free_tier,
            "remaining_credit": f"${300 - cost:.2f}" if not spec.is_free_tier else "$300.00"
        }

    def deploy_worker(
        self,
        name: str,
        spec_name: str = "free",
        ssh_key_path: str = "~/.ssh/id_rsa.pub",
        subnet_id: str = "",
        image_id: str = ""
    ) -> bool:
        """Deploy a training worker on OCI"""

        spec = INSTANCE_SPECS.get(spec_name)
        if not spec:
            logger.error(f"Unknown spec: {spec_name}")
            return False

        logger.info(f"Deploying worker: {name}")
        logger.info(f"  Shape: {spec.shape}")
        logger.info(f"  OCPUs: {spec.ocpus}, Memory: {spec.memory_gb}GB")
        logger.info(f"  Cost: ${spec.hourly_cost}/hr")

        # Create instance
        instance_id = self.oci.create_instance(
            name=name,
            spec=spec,
            ssh_key=ssh_key_path,
            subnet_id=subnet_id,
            image_id=image_id
        )

        if not instance_id:
            return False

        # Wait for IP
        time.sleep(30)
        ip = self.oci.get_instance_ip(instance_id)

        if ip:
            self.workers[name] = {
                "instance_id": instance_id,
                "ip": ip,
                "spec": spec_name,
                "created_at": datetime.now().isoformat()
            }
            logger.info(f"Worker deployed: {name} @ {ip}")
            return True

        return False

    def setup_worker(self, name: str) -> bool:
        """Setup training environment on worker"""

        if name not in self.workers:
            logger.error(f"Worker not found: {name}")
            return False

        ip = self.workers[name]["ip"]

        # Generate setup script
        script = WORKER_SETUP_SCRIPT.format(
            deepseek_key=os.getenv("DEEPSEEK_API_KEY", ""),
            github_token=os.getenv("GITHUB_TOKEN", ""),
            hf_token=os.getenv("HF_TOKEN", "")
        )

        # Save script locally
        script_path = self.output_dir / f"setup_{name}.sh"
        with open(script_path, 'w') as f:
            f.write(script)

        # Copy and execute on worker
        try:
            # Copy script
            subprocess.run([
                "scp", "-o", "StrictHostKeyChecking=no",
                str(script_path),
                f"ubuntu@{ip}:/tmp/setup.sh"
            ], check=True, timeout=60)

            # Execute script
            subprocess.run([
                "ssh", "-o", "StrictHostKeyChecking=no",
                f"ubuntu@{ip}",
                "chmod +x /tmp/setup.sh && /tmp/setup.sh"
            ], check=True, timeout=300)

            logger.info(f"Worker setup complete: {name}")
            return True

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False

    def get_worker_status(self, name: str) -> Dict:
        """Get status of a worker"""

        if name not in self.workers:
            return {"error": "Worker not found"}

        ip = self.workers[name]["ip"]

        try:
            result = subprocess.run([
                "ssh", "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=5",
                f"ubuntu@{ip}",
                "tail -20 /home/ubuntu/engineer/training.log 2>/dev/null || echo 'No log yet'"
            ], capture_output=True, text=True, timeout=30)

            return {
                "name": name,
                "ip": ip,
                "status": "running" if result.returncode == 0 else "unknown",
                "log": result.stdout[-500:] if result.stdout else ""
            }

        except Exception as e:
            return {"name": name, "status": "unreachable", "error": str(e)}

    def save_state(self):
        """Save manager state"""
        state = {
            "workers": self.workers,
            "timestamp": datetime.now().isoformat()
        }

        with open(self.output_dir / "oci_state.json", 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Load manager state"""
        state_file = self.output_dir / "oci_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                self.workers = state.get("workers", {})


# =============================================================================
# LOCAL SIMULATION (for testing without OCI)
# =============================================================================

class LocalTrainingSimulator:
    """Simulates OCI training locally for testing"""

    def __init__(self):
        self.workers: Dict[str, asyncio.subprocess.Process] = {}
        self.output_dir = Path("training_output")
        self.output_dir.mkdir(exist_ok=True)

    async def start_worker(self, worker_id: int, iterations: int = 50) -> bool:
        """Start a local training worker"""

        logger.info(f"Starting local worker {worker_id}")

        cmd = [
            "python3", "multi_provider_trainer.py",
            "--workers", "1",
            "--iterations", str(iterations),
            "--delay", "1.0"
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path(__file__).parent)
        )

        self.workers[f"local_{worker_id}"] = process
        return True

    async def run_training(self, num_workers: int = 1, iterations: int = 20):
        """Run training with multiple local workers"""

        logger.info(f"Starting {num_workers} local training workers")

        # Start all workers
        for i in range(num_workers):
            await self.start_worker(i, iterations)

        # Wait for completion
        for name, process in self.workers.items():
            stdout, stderr = await process.communicate()
            logger.info(f"Worker {name} completed")
            if stdout:
                logger.info(stdout.decode()[-500:])


# =============================================================================
# MAIN
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Oracle Cloud Training Workers")
    parser.add_argument("--mode", choices=["local", "oci", "estimate"], default="local")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--spec", choices=list(INSTANCE_SPECS.keys()), default="free")
    parser.add_argument("--hours", type=int, default=24, help="Hours to run (for cost estimate)")
    args = parser.parse_args()

    print("=" * 60)
    print("ORACLE CLOUD TRAINING WORKERS")
    print("=" * 60)

    if args.mode == "estimate":
        # Cost estimation
        manager = OracleTrainingManager()
        estimate = manager.estimate_costs(args.spec, args.hours)

        print(f"\nCost Estimate for {args.hours} hours:")
        print(f"  Instance: {estimate['spec']}")
        print(f"  Shape: {estimate['shape']}")
        print(f"  OCPUs: {estimate['ocpus']}, RAM: {estimate['memory_gb']}GB")
        print(f"  Hourly: {estimate['hourly_cost']}")
        print(f"  Total: {estimate['total_cost']}")
        print(f"  Free tier: {'Yes' if estimate['is_free'] else 'No'}")
        print(f"  Remaining credit: {estimate['remaining_credit']}")

    elif args.mode == "local":
        # Run locally
        print(f"\nRunning {args.workers} local workers for {args.iterations} iterations")

        simulator = LocalTrainingSimulator()
        await simulator.run_training(args.workers, args.iterations)

    elif args.mode == "oci":
        # Deploy to OCI
        manager = OracleTrainingManager()

        if not manager.oci.check_oci_cli():
            print("\nOCI CLI not configured. Install with:")
            print("  pip install oci-cli")
            print("  oci setup config")
            return

        print(f"\nDeploying {args.workers} workers on Oracle Cloud")
        print(f"Spec: {args.spec}")

        for i in range(args.workers):
            name = f"training-worker-{i}"
            manager.deploy_worker(name, args.spec)
            manager.setup_worker(name)

        manager.save_state()
        print("\nWorkers deployed. Check status with:")
        print("  python oracle_training_workers.py --mode oci --status")


if __name__ == "__main__":
    asyncio.run(main())
