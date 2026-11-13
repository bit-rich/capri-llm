#!/usr/bin/env bash
# Runs ON the AWS machine (detached): fetch the prepared weights from S3 and serve them to remote clients.
#   remote_serve.sh <s3 prefix> <repo>     e.g. remote_serve.sh s3://bucket/qwen3-32b Qwen/Qwen3-32B
# Needs ~/.aws/credentials (short-lived) and ~/.config/pi/token (the password, mode 600).
set -euo pipefail
PREFIX=$1; REPO=$2
source /opt/pytorch/bin/activate; cd ~/pi
step() { echo "[$(date -u +%H:%M:%S)] $*"; }
step "compile kernels"; gcc -O3 -march=native -shared -fPIC kernels.c -o libk.so
step "fetch prepared weights"; aws s3 sync $PREFIX/weights /data/prep/weights --only-show-errors; du -sh /data/prep/weights
rm -f ~/.aws/credentials                            # not needed any more
step "serve"; exec python server.py repo=$REPO bits=20 host=0.0.0.0 port=7000 weights=/data/prep/weights
