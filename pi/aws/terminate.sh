#!/usr/bin/env bash
# Delete the test machine now, then list anything still running in the region (should be nothing).
set -euo pipefail
source ~/.config/pi/instance
aws ec2 terminate-instances --region $R --instance-ids $ID --query 'TerminatingInstances[0].CurrentState.Name' --output text
aws ec2 wait instance-terminated --region $R --instance-ids $ID && echo "$ID terminated"
echo "still running in $R:"; aws ec2 describe-instances --region $R --filters Name=instance-state-name,Values=pending,running,stopping \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]' --output text
