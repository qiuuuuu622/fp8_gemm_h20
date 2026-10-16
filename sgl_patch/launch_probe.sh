#!/bin/bash
# Launch the /prefill_model server (prefill-node args minus dynamo/PD/spec) and time startup phases.
#   launch_probe.sh <tag> <gpu> <port> <ours|dg> <cold|warm>
# ours: FP8MM_ENABLE=1 (+ skip DeepGEMM precompile); dg: stock DeepGEMM path.
# cold: fresh SGLANG_CACHE_DIR (all JIT caches empty); warm: reuse the tag's cache from a previous run.
set -u
TAG=$1; GPU=$2; PORT=$3; IMPL=$4; CACHE=$5
D=/root/fp8mm/startup
mkdir -p $D
CDIR=$D/cache_$TAG
[ "$CACHE" = cold ] && rm -rf "$CDIR"
mkdir -p "$CDIR"
export CUDA_VISIBLE_DEVICES=$GPU
export SGLANG_CACHE_DIR=$CDIR
export TRITON_CACHE_DIR=$CDIR/triton
if [ "$IMPL" = ours ]; then
  export FP8MM_ENABLE=1 FP8MM_DISABLE_DG_PRECOMPILE=1 FP8MM_HOME=/root/fp8mm
  export PYTHONPATH=/root/fp8mm/sgl_patch${PYTHONPATH:+:$PYTHONPATH}
fi
LOG=$D/launch_${TAG}_${CACHE}.log
START=$(date +%s.%N)
/opt/sglang/bin/python -m sglang.launch_server --model-path ${MODEL:-/dev/shm/prefill_model} --context-length 262144 \
  --host 127.0.0.1 --port $PORT --tensor-parallel-size 1 --mem-fraction-static 0.6 --served-model-name probe \
  --limit-mm-data-per-request '{"video":0}' --mm-process-config '{"image":{"min_pixels":3136,"max_pixels":2097152}}' \
  --enable-multimodal --page-size 16 --max-running-requests 16 \
  > >(while IFS= read -r l; do printf "%s %s\n" "$(date +%s.%N)" "$l"; done > $LOG) 2>&1 &
PID=$!
until curl -sf http://127.0.0.1:$PORT/health_generate > /dev/null 2>&1; do
  sleep 2
  kill -0 $PID 2>/dev/null || { echo "server died"; exit 1; }
done
READY=$(date +%s.%N)
python3 -c "print(\"READY_SEC %.1f\" % ($READY - $START))" | tee -a $LOG
echo $PID > $D/pid_$TAG
