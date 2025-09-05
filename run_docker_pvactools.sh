#!/bin/bash
"""
Run PVACtools analysis using official Griffith Lab Docker image
Addresses previous Docker "no output" issues
"""

echo "🧬 Starting PVACtools Docker Analysis"
echo "====================================="

# Set project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project directory: $PROJECT_DIR"

# Make script executable and run
chmod +x "$PROJECT_DIR/scripts/tools/docker_pvactools_runner.py"
python3 "$PROJECT_DIR/scripts/tools/docker_pvactools_runner.py"

echo ""
echo "Analysis complete. Check results in: $PROJECT_DIR/results/pvacbind_docker/"
