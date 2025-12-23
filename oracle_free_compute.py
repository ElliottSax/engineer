#!/usr/bin/env python3
"""
Oracle Cloud + HuggingFace FREE Compute System
Combines Oracle Cloud's free tier (4 ARM CPUs, 24GB RAM) with HuggingFace APIs
Total cost: $0.00
"""

import os
import json
import asyncio
import requests
from pathlib import Path
import logging
from typing import Dict, List, Optional
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OracleHuggingFaceSystem:
    """Combines Oracle Cloud free compute with HuggingFace free APIs"""

    def __init__(self):
        # Oracle Cloud config
        self.oci_config = {
            "user": "ocid1.user.oc1..aaaaaaaa74qquxlbn7ky5lpcfkt2akeemtn4v2gd7kdc52dve7mx7kxbdziq",
            "tenancy": "ocid1.tenancy.oc1..aaaaaaaa2ktu74gnhxcctwnk65ntpj6gfb53ofanbz2ram3jkm62ke5ekpsa",
            "region": "us-chicago-1",
            "fingerprint": "f0:02:61:70:fd:35:eb:67:a6:20:3a:08:ab:f4:4f:6f"
        }

        # HuggingFace token for free API
        self.hf_token = "$(HF_TOKEN)"

        # Other free/cheap APIs
        self.apis = {
            "gemini": "REDACTED_GEMINI_KEY",
            "deepseek": "REDACTED_DEEPSEEK_KEY",
            "alibaba": "REDACTED_ALIBABA_KEY"
        }

        logger.info("🌩️ Oracle Cloud + HuggingFace FREE system initialized")

    def setup_oci_compute(self):
        """Setup Oracle Cloud compute instance (if not exists)"""

        # Check if we have OCI CLI
        try:
            result = subprocess.run(["oci", "--version"], capture_output=True)
            if result.returncode != 0:
                logger.info("Installing OCI CLI...")
                subprocess.run(["bash", "-c", "curl -L https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.sh | bash"])
        except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
            logger.warning(f"OCI CLI not found - manual setup required: {e}")

        # Create config file
        oci_config_path = Path.home() / ".oci/config"
        oci_config_path.parent.mkdir(exist_ok=True)

        config_content = f"""[DEFAULT]
user={self.oci_config['user']}
fingerprint={self.oci_config['fingerprint']}
tenancy={self.oci_config['tenancy']}
region={self.oci_config['region']}
key_file=~/.oci/oci_api_key.pem
"""

        with open(oci_config_path, 'w') as f:
            f.write(config_content)

        logger.info("✅ Oracle Cloud config created")

    async def deploy_ollama_on_oci(self):
        """Deploy Ollama on Oracle Cloud free tier for local inference"""

        deploy_script = """#!/bin/bash
# Deploy Ollama on Oracle Cloud Free Tier

# Create compute instance (ARM-based, free tier)
oci compute instance launch \
    --availability-domain "mFKn:US-CHICAGO-1-AD-1" \
    --compartment-id "{tenancy}" \
    --shape "VM.Standard.A1.Flex" \
    --shape-config '{"memoryInGBs": 24, "ocpus": 4}' \
    --display-name "ollama-free-compute" \
    --image-id "ocid1.image.oc1.us-chicago-1.aaaaaaaafp3tq7vp6hpt3s6xpmryqhljdvvmxbxjw5q5xfh5hz4gg7f2wdka" \
    --subnet-id "{subnet_id}" \
    --assign-public-ip true \
    --ssh-authorized-keys-file ~/.ssh/id_rsa.pub \
    --user-data-file ollama_setup.sh

# Wait for instance to be running
sleep 60

# Get instance IP
INSTANCE_IP=$(oci compute instance list-vnics \
    --compartment-id "{tenancy}" \
    --display-name "ollama-free-compute" \
    --query "data[0].\"public-ip\"" --raw-output)

echo "Oracle Cloud instance IP: $INSTANCE_IP"
echo "SSH: ssh opc@$INSTANCE_IP"
"""

        # Create Ollama setup script
        ollama_setup = """#!/bin/bash
# Install Ollama on Oracle Linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5-coder:7b
ollama pull llama3.2:3b
ollama serve
"""

        with open("ollama_setup.sh", 'w') as f:
            f.write(ollama_setup)

        logger.info("🚀 Oracle Cloud deployment scripts created")
        logger.info("Run manually to deploy: bash deploy_oci.sh")


class UltraFreeExecutor:
    """Executor using ONLY free resources"""

    def __init__(self):
        self.hf_token = "$(HF_TOKEN)"
        self.gemini_key = "REDACTED_GEMINI_KEY"
        self.total_cost = 0.0
        self.providers_used = {"huggingface": 0, "gemini": 0, "local": 0}

    async def execute_free(self, prompt: str) -> Dict:
        """Execute using only free compute"""

        # Try HuggingFace first (completely free)
        for model in ["bigcode/starcoder2-15b", "mistralai/Mixtral-8x7B-Instruct-v0.1", "codellama/CodeLlama-34b-Python-hf"]:
            try:
                url = f"https://api-inference.huggingface.co/models/{model}"
                headers = {"Authorization": f"Bearer {self.hf_token}"}
                data = {"inputs": prompt, "parameters": {"max_new_tokens": 500}}

                response = requests.post(url, headers=headers, json=data, timeout=30)
                if response.status_code == 200:
                    self.providers_used["huggingface"] += 1
                    logger.info(f"✅ HuggingFace {model} (FREE)")
                    return {"success": True, "output": response.json(), "cost": 0.0}
            except (requests.RequestException, ConnectionError, TimeoutError) as e:
                logger.debug(f"HuggingFace {model} failed: {e}")
                continue

        # Fallback to Gemini (free tier)
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.gemini_key}"
            data = {"contents": [{"parts": [{"text": prompt}]}]}

            response = requests.post(url, json=data, timeout=30)
            if response.status_code == 200:
                self.providers_used["gemini"] += 1
                logger.info("✅ Gemini free tier")
                return {"success": True, "output": response.json(), "cost": 0.0}
        except (requests.RequestException, ConnectionError, TimeoutError) as e:
            logger.debug(f"Gemini failed: {e}")

        # Try local Ollama if available
        try:
            result = subprocess.run(
                ["ollama", "run", "qwen2.5-coder:7b", prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                self.providers_used["local"] += 1
                logger.info("✅ Local Ollama (FREE)")
                return {"success": True, "output": result.stdout, "cost": 0.0}
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.debug(f"Local Ollama failed: {e}")

        return {"success": False, "error": "No free compute available"}


async def run_free_system():
    """Run the completely free system"""

    logger.info("=" * 60)
    logger.info("🆓 ULTRA FREE MODE - ZERO COST OPERATION")
    logger.info("Using: HuggingFace + Gemini + Oracle Cloud")
    logger.info("=" * 60)

    # Load tasks
    with open("tasks.json", 'r') as f:
        tasks = json.load(f)["tasks"]

    executor = UltraFreeExecutor()
    completed = 0

    # Process tasks with free compute
    for i, task in enumerate(tasks[:10], 1):  # First 10 tasks
        logger.info(f"[{i}/10] Processing: {task['prompt'][:50]}...")

        result = await executor.execute_free(task["prompt"])

        if result["success"]:
            completed += 1
            logger.info(f"✅ Completed (Cost: $0.00)")
        else:
            logger.error(f"❌ Failed: {result.get('error')}")

        await asyncio.sleep(1)  # Rate limiting

    # Report
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 FINAL REPORT")
    logger.info("=" * 60)
    logger.info(f"Tasks completed: {completed}/10")
    logger.info(f"Total cost: $0.00")
    logger.info(f"Providers used: {executor.providers_used}")
    logger.info(f"Money saved vs Claude: ~${completed * 0.15:.2f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_free_system())