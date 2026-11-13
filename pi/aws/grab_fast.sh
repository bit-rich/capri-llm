#!/usr/bin/env bash
# Fast capacity grab: set up once, then one launch request every GAP seconds, rotating over zones and types.
#   aws/grab_fast.sh <hours limit> <gap s> <types...>      status line every 60 s; exits after a launch
HOURS=$1; GAP=$2; shift 2; TYPES=($@); R=us-west-2; KEY=pi-test; CONF=~/.config/pi
AMI=$(aws ec2 describe-images --region $R --owners amazon --filters \
  "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04) *" "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
SGID=$(aws ec2 describe-security-groups --region $R --filters Name=group-name,Values=pi-test --query 'SecurityGroups[0].GroupId' --output text)
MYIP=$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')
aws ec2 authorize-security-group-ingress --region $R --group-id $SGID --protocol tcp --port 22 --cidr $MYIP/32 >/dev/null 2>&1 || true
SUBS=($(aws ec2 describe-subnets --region $R --filters Name=default-for-az,Values=true --query 'Subnets[].SubnetId' --output text))
USERDATA=$(printf '#!/bin/bash\nshutdown -h +%d\n' $((HOURS * 60)))
STATE=/tmp/claude-1000/-home-xiphos-Documents-projects-HE-testing/45d8d881-a595-44b6-82f1-ae3e48890a6f/scratchpad/grab.state
N=0; BUSY=0; [ -f $STATE ] && source $STATE; last=$(date +%s); i=0
echo "$(date +%H:%M:%S) fast mode: one request every ${GAP}s over ${#SUBS[@]} zones x ${#TYPES[@]} types (so far: $BUSY/$N busy)"
while true; do
  T=${TYPES[$(( (i / ${#SUBS[@]}) % ${#TYPES[@]} ))]}; S=${SUBS[$(( i % ${#SUBS[@]} ))]}; i=$((i + 1)); N=$((N + 1))
  OUT=$(aws ec2 run-instances --region $R --image-id $AMI --instance-type $T --key-name $KEY --security-group-ids $SGID --subnet-id $S \
    --instance-initiated-shutdown-behavior terminate --user-data "$USERDATA" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=pi-test}]' --query 'Instances[0].InstanceId' --output text 2>&1)
  if [[ $OUT == i-* ]]; then
    aws ec2 wait instance-running --region $R --instance-ids $OUT
    IP=$(aws ec2 describe-instances --region $R --instance-ids $OUT --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    printf 'ID=%s\nIP=%s\nR=%s\nSGID=%s\n' $OUT $IP $R $SGID > $CONF/instance; rm -f $STATE
    echo "$(date +%H:%M:%S) GOT ONE after $N checks ($BUSY busy): $T $OUT at $IP"; exit 0
  fi
  if [[ $OUT == *InsufficientInstanceCapacity* ]]; then BUSY=$((BUSY + 1)); else echo "$(date +%H:%M:%S) unexpected: ${OUT:0:200}"; fi
  echo "N=$N BUSY=$BUSY" > $STATE
  if [ $(( $(date +%s) - last )) -ge 60 ]; then echo "$(date +%H:%M:%S) busy $BUSY/$N checks so far, nothing available"; last=$(date +%s); fi
  sleep $GAP
done
