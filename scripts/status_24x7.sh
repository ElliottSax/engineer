#!/bin/bash
#
# Quick status check for 24/7 orchestrator
# Run locally or on the Oracle Cloud instance
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/.."

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           AUTOCODER 24/7 ORCHESTRATOR STATUS                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check if running on Oracle Cloud
if [ -f /etc/oracle-cloud-agent/plugins/oci-hpc/oci-hpc-configure.sh ]; then
    echo -e "${BLUE}Environment:${NC} Oracle Cloud Instance"
else
    echo -e "${BLUE}Environment:${NC} Local"
fi

# Check systemd service (if on Linux)
if command -v systemctl &> /dev/null; then
    echo ""
    echo -e "${BLUE}Service Status:${NC}"
    if systemctl is-active --quiet autocoder 2>/dev/null; then
        echo -e "  Status: ${GREEN}● Running${NC}"
        echo -e "  $(systemctl show autocoder --property=ActiveEnterTimestamp 2>/dev/null | sed 's/ActiveEnterTimestamp=/Started: /')"
    else
        echo -e "  Status: ${RED}● Stopped${NC}"
    fi
fi

# Check orchestrator state file
STATE_FILE="${BASE_DIR}/.orchestrator_state/orchestrator_state.json"
if [ -f "$STATE_FILE" ]; then
    echo ""
    echo -e "${BLUE}Orchestrator State:${NC}"

    if command -v python3 &> /dev/null; then
        python3 << EOF
import json
from datetime import datetime

try:
    with open('$STATE_FILE') as f:
        state = json.load(f)

    stats = state.get('stats', {})
    workers = state.get('workers', {})
    queue = state.get('queue', {})

    print(f"  Uptime: {stats.get('uptime', 'N/A')}")
    print(f"  Tasks Completed: {stats.get('total_tasks_completed', 0)}")
    print(f"  Tasks Failed: {stats.get('total_tasks_failed', 0)}")
    print(f"  Success Rate: {stats.get('success_rate', 'N/A')}")
    print(f"  Improvements Made: {stats.get('improvements_made', 0)}")
    print()
    print(f"  Workers: {workers.get('running', 0)}/{workers.get('total_workers', 0)} running")
    print(f"  Queue: {queue.get('pending', 0)} pending, {queue.get('in_progress', 0)} in progress")
except Exception as e:
    print(f"  Error reading state: {e}")
EOF
    else
        echo "  (Install python3 for detailed stats)"
        cat "$STATE_FILE" | head -20
    fi
else
    echo ""
    echo -e "${YELLOW}No orchestrator state file found${NC}"
fi

# Check logs
LOG_FILE="${BASE_DIR}/logs/orchestrator.log"
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo -e "${BLUE}Recent Log Entries:${NC}"
    tail -10 "$LOG_FILE" | while read line; do
        if echo "$line" | grep -q "ERROR"; then
            echo -e "  ${RED}$line${NC}"
        elif echo "$line" | grep -q "SUCCESS\|completed"; then
            echo -e "  ${GREEN}$line${NC}"
        elif echo "$line" | grep -q "WARN"; then
            echo -e "  ${YELLOW}$line${NC}"
        else
            echo "  $line"
        fi
    done
fi

# Check disk space
echo ""
echo -e "${BLUE}Disk Usage:${NC}"
df -h "${BASE_DIR}" 2>/dev/null | tail -1 | awk '{print "  Used: "$3" / "$2" ("$5")"}'

# Check memory
echo ""
echo -e "${BLUE}Memory Usage:${NC}"
if [ -f /proc/meminfo ]; then
    free -h 2>/dev/null | grep Mem | awk '{print "  Used: "$3" / "$2}'
fi

# Check Python processes
echo ""
echo -e "${BLUE}Python Processes:${NC}"
ps aux 2>/dev/null | grep -E "oracle_24x7|multi_provider|unified_coding" | grep -v grep | while read line; do
    echo "  $line" | awk '{print "  PID:"$2" CPU:"$3"% MEM:"$4"% "$11}'
done

if ! ps aux 2>/dev/null | grep -E "oracle_24x7|multi_provider|unified_coding" | grep -v grep > /dev/null; then
    echo -e "  ${YELLOW}No orchestrator processes found${NC}"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""
