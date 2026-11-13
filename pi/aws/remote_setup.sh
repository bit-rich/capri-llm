#!/usr/bin/env bash
# Runs ON the AWS machine: put the data on the fast local disks, make a Python environment, check the GPUs.
set -euo pipefail
if [ -d /opt/dlami/nvme ] && [ ! -e /data/.ok ]; then  # Deep Learning AMI: the instance disks are already one volume there
  sudo rmdir /data 2>/dev/null || true; sudo ln -sfn /opt/dlami/nvme /data; sudo chown $USER /opt/dlami/nvme; touch /data/.ok
fi
if [ ! -e /data/.ok ] && ! mountpoint -q /data; then                      # instance-store NVMe disks (g4dn.metal has two): one fast volume
  DEVS=$(lsblk -dpno NAME,MODEL | awk '/Instance Storage/ {print $1}')
  sudo mkdir -p /data
  if [ $(echo "$DEVS" | wc -w) -gt 1 ]; then sudo mdadm --create /dev/md0 --run --level=0 --raid-devices=$(echo "$DEVS" | wc -w) $DEVS; D=/dev/md0; else D=$DEVS; fi
  sudo mkfs.ext4 -q -F $D; sudo mount $D /data; sudo chown $USER /data
fi
source /opt/pytorch/bin/activate                    # the image's PyTorch environment, plus what the code needs
pip -q install numpy tokenizers huggingface_hub safetensors
python - <<'PY'
import torch
print("torch", torch.__version__, "GPUs", torch.cuda.device_count(), torch.cuda.get_device_name(0))
a = torch.randint(-128, 127, (64, 256), dtype=torch.int8, device="cuda"); b = torch.randint(-128, 127, (256, 32), dtype=torch.int8, device="cuda")
assert (torch._int_mm(a, b).cpu() == a.cpu().int() @ b.cpu().int()).all(); print("int8 GEMM on this GPU: ok")
PY
df -h /data | tail -1
