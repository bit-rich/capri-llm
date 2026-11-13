#!/usr/bin/env bash
# Runs ON the AWS machine (detached, so a dropped SSH connection doesn't stop it):
# download the model, prepare weights + bundle + one-time C1', then check the result with a local client.
#   remote_prep.sh <HF repo> <out dir>
set -euo pipefail
REPO=$1; OUT=$2
source /opt/pytorch/bin/activate; cd ~/pi
export HF_HOME=/data/hf
step() { echo "[$(date -u +%H:%M:%S)] $*"; }
step "compile kernels"; gcc -O3 -march=native -shared -fPIC kernels.c -o libk.so
step "download $REPO"; python -c "from protocol import qwen_path; print(qwen_path('$REPO'))"
step "prepare (calibrate, quantize, merge, C1')"; python server.py repo=$REPO bits=20 gpuload calib=24 prepare=$OUT
if grep -q '"output_gate_type": "swish"' $(python -c "from protocol import qwen_path; print(qwen_path('$REPO'))")/config.json; then
  step "gate check (config says output_gate_type=swish; HF uses sigmoid)"
  python gate_check.py $OUT/weights tokens=48 | tee /data/gate.log
  export PI_GATE=$(sort -t: -k2 -g <(sed -E 's/gate ([a-z]+) *: perplexity +([0-9.e+]+).*/\1:\2/' /data/gate.log) | head -1 | cut -d: -f1)
  step "using gate $PI_GATE"
fi
step "check: serve the prepared weights, client on this machine (CPU), compare with plain W8"
python server.py repo=$REPO bits=20 port=7001 weights=$OUT/weights > /data/server_check.log 2>&1 &
SP=$!; until grep -q listening /data/server_check.log; do sleep 5; kill -0 $SP; done
CUDA_VISIBLE_DEVICES= PI_TOKEN= python client.py port=7001 prompts=2 new=8 check cache=$OUT label=awscheck
kill $SP
step "done"
