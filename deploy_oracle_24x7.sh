#!/bin/bash
#
# Oracle Cloud 24/7 Worker Deployment Script
#
# This script deploys the autonomous improvement system to Oracle Cloud.
# It handles instance creation, code deployment, and service setup.
#
# Usage:
#   ./deploy_oracle_24x7.sh              # Deploy to existing instance
#   ./deploy_oracle_24x7.sh --create     # Create new instance and deploy
#   ./deploy_oracle_24x7.sh --status     # Check deployment status
#   ./deploy_oracle_24x7.sh --ssh        # SSH into instance
#
# Requirements:
#   - OCI CLI configured (~/.oci/config)
#   - SSH key pair for instance access
#   - Python 3.9+ on local machine
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="autocoder-24x7"
INSTANCE_NAME="autocoder-worker"

# OCI Configuration (override with environment variables)
OCI_REGION="${OCI_REGION:-us-ashburn-1}"
OCI_COMPARTMENT="${OCI_COMPARTMENT:-}"
OCI_SHAPE="${OCI_SHAPE:-VM.Standard.A1.Flex}"  # Free tier ARM
OCI_OCPUS="${OCI_OCPUS:-2}"
OCI_MEMORY_GB="${OCI_MEMORY_GB:-12}"

# SSH Configuration
SSH_KEY="${SSH_KEY:-~/.ssh/id_rsa}"
SSH_USER="${SSH_USER:-ubuntu}"
REMOTE_DIR="/home/${SSH_USER}/autocoder"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Helper Functions
# =============================================================================

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check OCI CLI
    if ! command -v oci &> /dev/null; then
        log_error "OCI CLI not found. Install with: pip install oci-cli"
        exit 1
    fi

    # Check SSH key
    if [ ! -f "${SSH_KEY}" ]; then
        log_error "SSH key not found: ${SSH_KEY}"
        exit 1
    fi

    # Check OCI config
    if [ ! -f ~/.oci/config ]; then
        log_error "OCI config not found. Run: oci setup config"
        exit 1
    fi

    log_success "Prerequisites OK"
}

get_instance_ip() {
    # Try to get instance IP from OCI
    local ip=""

    if [ -n "${OCI_COMPARTMENT}" ]; then
        ip=$(oci compute instance list \
            --compartment-id "${OCI_COMPARTMENT}" \
            --display-name "${INSTANCE_NAME}" \
            --lifecycle-state RUNNING \
            --query 'data[0]."id"' \
            --raw-output 2>/dev/null || echo "")

        if [ -n "$ip" ] && [ "$ip" != "null" ]; then
            # Get VNIC attachments
            local vnic_id=$(oci compute vnic-attachment list \
                --compartment-id "${OCI_COMPARTMENT}" \
                --instance-id "$ip" \
                --query 'data[0]."vnic-id"' \
                --raw-output 2>/dev/null)

            if [ -n "$vnic_id" ] && [ "$vnic_id" != "null" ]; then
                ip=$(oci network vnic get \
                    --vnic-id "$vnic_id" \
                    --query 'data."public-ip"' \
                    --raw-output 2>/dev/null)
            fi
        fi
    fi

    # Fallback to environment variable or config file
    if [ -z "$ip" ] || [ "$ip" == "null" ]; then
        if [ -n "${INSTANCE_IP}" ]; then
            ip="${INSTANCE_IP}"
        elif [ -f "${SCRIPT_DIR}/.instance_ip" ]; then
            ip=$(cat "${SCRIPT_DIR}/.instance_ip")
        fi
    fi

    echo "$ip"
}

wait_for_ssh() {
    local ip=$1
    local max_attempts=30
    local attempt=0

    log_info "Waiting for SSH to be available..."

    while [ $attempt -lt $max_attempts ]; do
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i "${SSH_KEY}" "${SSH_USER}@${ip}" "echo OK" &>/dev/null; then
            log_success "SSH connection established"
            return 0
        fi
        attempt=$((attempt + 1))
        echo -n "."
        sleep 10
    done

    log_error "SSH connection timeout"
    return 1
}

# =============================================================================
# Deployment Functions
# =============================================================================

create_instance() {
    log_info "Creating OCI instance..."

    if [ -z "${OCI_COMPARTMENT}" ]; then
        log_error "OCI_COMPARTMENT not set. Export it or add to .env"
        exit 1
    fi

    # Get availability domain
    local ad=$(oci iam availability-domain list \
        --compartment-id "${OCI_COMPARTMENT}" \
        --query 'data[0].name' \
        --raw-output)

    # Get Ubuntu image
    local image_id=$(oci compute image list \
        --compartment-id "${OCI_COMPARTMENT}" \
        --operating-system "Canonical Ubuntu" \
        --operating-system-version "22.04" \
        --shape "${OCI_SHAPE}" \
        --query 'data[0].id' \
        --raw-output 2>/dev/null)

    if [ -z "$image_id" ] || [ "$image_id" == "null" ]; then
        log_error "Could not find Ubuntu image for shape ${OCI_SHAPE}"
        exit 1
    fi

    # Create cloud-init script
    local cloud_init=$(cat <<'CLOUDINIT'
#!/bin/bash
set -e

# Update system
apt-get update
apt-get upgrade -y

# Install dependencies
apt-get install -y python3 python3-pip python3-venv git htop tmux

# Create user directory
mkdir -p /home/ubuntu/autocoder
chown ubuntu:ubuntu /home/ubuntu/autocoder

# Install Python packages
sudo -u ubuntu pip3 install --user aiohttp requests pyyaml

echo "Cloud-init complete" > /tmp/cloud-init-done
CLOUDINIT
)

    local cloud_init_base64=$(echo "$cloud_init" | base64 -w 0)

    # Get or create VCN and subnet (simplified - use existing)
    local subnet_id="${OCI_SUBNET_ID}"

    if [ -z "$subnet_id" ]; then
        log_warn "OCI_SUBNET_ID not set. Using first available subnet."
        subnet_id=$(oci network subnet list \
            --compartment-id "${OCI_COMPARTMENT}" \
            --query 'data[0].id' \
            --raw-output 2>/dev/null)
    fi

    if [ -z "$subnet_id" ] || [ "$subnet_id" == "null" ]; then
        log_error "No subnet available. Create a VCN first."
        exit 1
    fi

    # Create instance
    log_info "Creating instance with shape ${OCI_SHAPE}..."

    local instance_id=$(oci compute instance launch \
        --compartment-id "${OCI_COMPARTMENT}" \
        --availability-domain "$ad" \
        --display-name "${INSTANCE_NAME}" \
        --image-id "$image_id" \
        --shape "${OCI_SHAPE}" \
        --shape-config "{\"ocpus\": ${OCI_OCPUS}, \"memoryInGBs\": ${OCI_MEMORY_GB}}" \
        --subnet-id "$subnet_id" \
        --assign-public-ip true \
        --metadata "{\"ssh_authorized_keys\": \"$(cat ${SSH_KEY}.pub)\", \"user_data\": \"${cloud_init_base64}\"}" \
        --query 'data.id' \
        --raw-output)

    if [ -z "$instance_id" ]; then
        log_error "Failed to create instance"
        exit 1
    fi

    log_success "Instance created: $instance_id"

    # Wait for instance to be running
    log_info "Waiting for instance to start..."
    oci compute instance get \
        --instance-id "$instance_id" \
        --wait-for-state RUNNING \
        --max-wait-seconds 300 >/dev/null

    # Get public IP
    sleep 10  # Wait for VNIC
    local ip=$(get_instance_ip)

    if [ -n "$ip" ]; then
        echo "$ip" > "${SCRIPT_DIR}/.instance_ip"
        log_success "Instance IP: $ip"
    else
        log_warn "Could not get instance IP automatically"
    fi
}

deploy_code() {
    local ip=$(get_instance_ip)

    if [ -z "$ip" ]; then
        log_error "Instance IP not found. Set INSTANCE_IP or run with --create"
        exit 1
    fi

    log_info "Deploying code to ${ip}..."

    # Wait for SSH
    wait_for_ssh "$ip"

    # Create remote directory
    ssh -i "${SSH_KEY}" "${SSH_USER}@${ip}" "mkdir -p ${REMOTE_DIR}"

    # Sync code (exclude large files and caches)
    rsync -avz --progress \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.git' \
        --exclude 'node_modules' \
        --exclude 'venv' \
        --exclude '.env' \
        --exclude '*.log' \
        --exclude 'training_output' \
        --exclude 'logs' \
        --exclude '.orchestrator_state' \
        -e "ssh -i ${SSH_KEY}" \
        "${SCRIPT_DIR}/"*.py \
        "${SCRIPT_DIR}/tests/" \
        "${SCRIPT_DIR}/utils/" \
        "${SSH_USER}@${ip}:${REMOTE_DIR}/"

    log_success "Code deployed"

    # Install dependencies
    log_info "Installing dependencies..."
    ssh -i "${SSH_KEY}" "${SSH_USER}@${ip}" "cd ${REMOTE_DIR} && pip3 install --user aiohttp requests pyyaml"

    log_success "Dependencies installed"
}

setup_service() {
    local ip=$(get_instance_ip)

    if [ -z "$ip" ]; then
        log_error "Instance IP not found"
        exit 1
    fi

    log_info "Setting up systemd service..."

    # Create systemd service file
    local service_file=$(cat <<SERVICE
[Unit]
Description=AutoCoder 24/7 Orchestrator
After=network.target

[Service]
Type=simple
User=${SSH_USER}
WorkingDirectory=${REMOTE_DIR}
ExecStart=/usr/bin/python3 ${REMOTE_DIR}/oracle_24x7_orchestrator.py --workers 3
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE
)

    # Upload and enable service
    echo "$service_file" | ssh -i "${SSH_KEY}" "${SSH_USER}@${ip}" "sudo tee /etc/systemd/system/autocoder.service > /dev/null"

    ssh -i "${SSH_KEY}" "${SSH_USER}@${ip}" "sudo systemctl daemon-reload && sudo systemctl enable autocoder && sudo systemctl start autocoder"

    log_success "Service installed and started"

    # Check status
    ssh -i "${SSH_KEY}" "${SSH_USER}@${ip}" "sudo systemctl status autocoder --no-pager" || true
}

show_status() {
    local ip=$(get_instance_ip)

    if [ -z "$ip" ]; then
        log_error "Instance IP not found"
        exit 1
    fi

    echo ""
    echo "=========================================="
    echo "ORACLE 24/7 DEPLOYMENT STATUS"
    echo "=========================================="
    echo ""
    echo "Instance IP: $ip"
    echo ""

    # Check SSH connectivity
    if ssh -o ConnectTimeout=5 -i "${SSH_KEY}" "${SSH_USER}@${ip}" "echo OK" &>/dev/null; then
        echo "SSH: ✓ Connected"

        # Check service status
        echo ""
        echo "Service Status:"
        ssh -i "${SSH_KEY}" "${SSH_USER}@${ip}" "sudo systemctl status autocoder --no-pager" 2>/dev/null || echo "  Service not installed"

        # Check orchestrator state
        echo ""
        echo "Orchestrator State:"
        ssh -i "${SSH_KEY}" "${SSH_USER}@${ip}" "cat ${REMOTE_DIR}/.orchestrator_state/orchestrator_state.json 2>/dev/null || echo '  No state file'" | head -30

        # Check logs
        echo ""
        echo "Recent Logs:"
        ssh -i "${SSH_KEY}" "${SSH_USER}@${ip}" "tail -20 ${REMOTE_DIR}/logs/orchestrator.log 2>/dev/null || echo '  No logs'"

    else
        echo "SSH: ✗ Cannot connect"
    fi

    echo ""
    echo "=========================================="
}

ssh_connect() {
    local ip=$(get_instance_ip)

    if [ -z "$ip" ]; then
        log_error "Instance IP not found"
        exit 1
    fi

    log_info "Connecting to ${ip}..."
    ssh -i "${SSH_KEY}" "${SSH_USER}@${ip}"
}

show_logs() {
    local ip=$(get_instance_ip)

    if [ -z "$ip" ]; then
        log_error "Instance IP not found"
        exit 1
    fi

    ssh -i "${SSH_KEY}" "${SSH_USER}@${ip}" "tail -f ${REMOTE_DIR}/logs/orchestrator.log"
}

restart_service() {
    local ip=$(get_instance_ip)

    if [ -z "$ip" ]; then
        log_error "Instance IP not found"
        exit 1
    fi

    log_info "Restarting service..."
    ssh -i "${SSH_KEY}" "${SSH_USER}@${ip}" "sudo systemctl restart autocoder"
    log_success "Service restarted"
}

full_deploy() {
    check_prerequisites
    deploy_code
    setup_service
    show_status
}

# =============================================================================
# Main
# =============================================================================

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --create      Create new OCI instance and deploy"
    echo "  --deploy      Deploy code to existing instance"
    echo "  --service     Setup systemd service only"
    echo "  --status      Show deployment status"
    echo "  --ssh         SSH into instance"
    echo "  --logs        Tail orchestrator logs"
    echo "  --restart     Restart the service"
    echo "  --help        Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  OCI_COMPARTMENT  OCI compartment OCID (required for --create)"
    echo "  OCI_REGION       OCI region (default: us-ashburn-1)"
    echo "  OCI_SHAPE        Instance shape (default: VM.Standard.A1.Flex)"
    echo "  INSTANCE_IP      Instance IP (if known)"
    echo "  SSH_KEY          SSH private key path (default: ~/.ssh/id_rsa)"
    echo ""
}

case "${1:-}" in
    --create)
        check_prerequisites
        create_instance
        deploy_code
        setup_service
        show_status
        ;;
    --deploy)
        check_prerequisites
        deploy_code
        ;;
    --service)
        setup_service
        ;;
    --status)
        show_status
        ;;
    --ssh)
        ssh_connect
        ;;
    --logs)
        show_logs
        ;;
    --restart)
        restart_service
        ;;
    --help)
        print_usage
        ;;
    "")
        # Default: full deploy to existing instance
        full_deploy
        ;;
    *)
        log_error "Unknown option: $1"
        print_usage
        exit 1
        ;;
esac
