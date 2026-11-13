#!/usr/bin/env bash
# Open the inference port (7000) to ONE address: aws/open_port.sh <client IP>   (default: this laptop's IP)
set -euo pipefail
source ~/.config/pi/instance
IP=${1:-$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')}
aws ec2 authorize-security-group-ingress --region $R --group-id $SGID --protocol tcp --port 7000 --cidr $IP/32 >/dev/null && echo "port 7000 open to $IP only"
