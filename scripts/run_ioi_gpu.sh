#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

JETSON_CUDA_LIBS="/tmp/jetson-cuda-libs/usr/lib/aarch64-linux-gnu:/tmp/jetson-cuda-libs/usr/local/cuda-12.6/targets/aarch64-linux/lib"
PY_CUDA_LIBS="/home/cog/.local/lib/python3.10/site-packages/nvidia/cufile/lib:/home/cog/.local/lib/python3.10/site-packages/nvidia/cufft/lib:/home/cog/.local/lib/python3.10/site-packages/nvidia/curand/lib:/home/cog/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/home/cog/.local/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:/home/cog/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/home/cog/.local/lib/python3.10/site-packages/nvidia/nvtx/lib:/home/cog/.local/lib/python3.10/site-packages/nvidia/cu12/lib"
export LD_LIBRARY_PATH="${JETSON_CUDA_LIBS}:${PY_CUDA_LIBS}:${LD_LIBRARY_PATH:-}"

command="${1:-preflight}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${command}" in
  -h|--help|help)
    echo "Usage: scripts/run_ioi_gpu.sh [preflight|smoke|stage0|stage1|agent] ..."
    ;;
  preflight)
    python scripts/ioi_gpu_preflight.py "$@"
    ;;
  smoke)
    python scripts/run_ioi_stage0_smoke.py "$@"
    ;;
  stage0)
    python scripts/ioi_stage0_bridge.py "$@"
    ;;
  stage1)
    python scripts/ioi_stage1_bridge.py "$@"
    ;;
  agent)
    python -m autointerp_agent.investigation "$@"
    ;;
  *)
    echo "Unknown command: ${command}" >&2
    echo "Usage: scripts/run_ioi_gpu.sh [preflight|smoke|stage0|stage1|agent] ..." >&2
    exit 2
    ;;
esac
