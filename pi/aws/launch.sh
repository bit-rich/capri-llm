#!/usr/bin/env bash
# Launch one GPU machine for the private-inference tests, with a hard self-delete timer.
#   aws/launch.sh <instance type> <hours>        e.g. aws/launch.sh g4dn.metal 2
# - The machine shuts itself down after <hours> and is then deleted (shutdown behaviour = terminate),
#   even if this laptop or Claude goes away.
# - SSH (port 22) is open only to this laptop's current public IP. The inference port is closed until open_port.sh.
# - Writes the instance id and address to ~/.config/pi/instance.
set -euo pipefail
TYPE=${1:?instance type}; HOURS=${2:?hours}; R=${PI_REGION:-us-west-1}; KEY=pi-test; SG=pi-test
CONF=~/.config/pi; mkdir -p $CONF; chmod 700 $CONF
AMI=$(aws ec2 describe-images --region $R --owners amazon --filters \
  "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *" "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
if [ ! -f $CONF/$KEY.pem ]; then
  aws ec2 create-key-pair --region $R --key-name $KEY --query KeyMaterial --output text > $CONF/$KEY.pem; chmod 600 $CONF/$KEY.pem
elif ! aws ec2 describe-key-pairs --region $R --key-names $KEY >/dev/null 2>&1; then   # key pairs are per region: reuse ours
  aws ec2 import-key-pair --region $R --key-name $KEY --public-key-material fileb://<(ssh-keygen -y -f $CONF/$KEY.pem) >/dev/null
fi
SGID=$(aws ec2 describe-security-groups --region $R --filters Name=group-name,Values=$SG --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ -z "$SGID" ] || [ "$SGID" = None ]; then
  SGID=$(aws ec2 create-security-group --region $R --group-name $SG --description "private inference tests" --query GroupId --output text)
fi
MYIP=$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')
aws ec2 authorize-security-group-ingress --region $R --group-id $SGID --protocol tcp --port 22 --cidr $MYIP/32 >/dev/null 2>&1 || true
USERDATA=$(printf '#!/bin/bash\nshutdown -h +%d\n' $((HOURS * 60)))
ID=$(aws ec2 run-instances --region $R --image-id $AMI --instance-type $TYPE --key-name $KEY --security-group-ids $SGID \
  ${PI_SUBNET:+--subnet-id $PI_SUBNET} \
  --instance-initiated-shutdown-behavior terminate --user-data "$USERDATA" \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=pi-test}]' --query 'Instances[0].InstanceId' --output text)
echo "launched $ID ($TYPE in $R), self-deletes after $HOURS h; waiting for it to start"
aws ec2 wait instance-running --region $R --instance-ids $ID
IP=$(aws ec2 describe-instances --region $R --instance-ids $ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
printf 'ID=%s\nIP=%s\nR=%s\nSGID=%s\n' $ID $IP $R $SGID > $CONF/instance
echo "running at $IP"
