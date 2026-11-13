#!/usr/bin/env bash
# Keep trying to launch until AWS has capacity; print one status line per round, then launch and exit.
#   aws/grab.sh <hours limit for the machine> <round interval s> <types...>
HOURS=$1; GAP=$2; shift 2; TYPES="$@"; R=us-west-2; N=0; BUSY=0; ROUND=0
STATE=/tmp/claude-1000/-home-xiphos-Documents-projects-HE-testing/45d8d881-a595-44b6-82f1-ae3e48890a6f/scratchpad/grab.state
[ -f $STATE ] && source $STATE
SUBS=$(aws ec2 describe-subnets --region $R --filters Name=default-for-az,Values=true --query 'Subnets[].SubnetId' --output text)
cd "$(dirname "$0")"
while true; do
  ROUND=$((ROUND + 1)); got=""
  for T in $TYPES; do for S in $SUBS; do
    N=$((N + 1))
    if PI_REGION=$R PI_SUBNET=$S bash launch.sh $T $HOURS > /tmp/grab_launch.log 2>&1 && grep -q "running at" /tmp/grab_launch.log; then got="$T"; break 2; fi
    grep -q InsufficientInstanceCapacity /tmp/grab_launch.log && BUSY=$((BUSY + 1)) || { echo "$(date +%H:%M) unexpected error: $(grep -o 'error occurred ([A-Za-z]*)' /tmp/grab_launch.log | head -1)"; }
  done; done
  echo "N=$N BUSY=$BUSY ROUND=$ROUND" > $STATE
  if [ -n "$got" ]; then source ~/.config/pi/instance; echo "$(date +%H:%M) GOT ONE after $N checks ($BUSY busy): $got $ID at $IP"; rm -f $STATE; exit 0; fi
  [ $((ROUND % ${PI_EVERY:-1})) = 0 ] && echo "$(date +%H:%M) round $ROUND: busy $BUSY/$N checks so far, nothing available"
  sleep $GAP
done
