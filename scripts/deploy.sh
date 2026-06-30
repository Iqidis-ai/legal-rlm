#!/bin/bash
# Auto-deploy script - pulls latest code and restarts services

set -e

cd /home/ubuntu/legal-rlm

echo "$(date): Starting deployment..."

# Pull latest changes
git fetch origin
git reset --hard origin/$(git rev-parse --abbrev-ref HEAD)

# Install any new dependencies
/home/ubuntu/legal-rlm/.venv/bin/pip install -r requirements.txt -q

# Restart services
sudo systemctl restart irys-rlm.service

# Check if UI service is enabled, restart if so
if systemctl is-enabled irys-rlm-ui.service &>/dev/null; then
    sudo systemctl restart irys-rlm-ui.service
fi

echo "$(date): Deployment complete!"
