#!/bin/bash
# How many networks, and nested against flat: 5-fold paper cross-validation (the task model's cv_abs.py, restarts 2,
# iters 1000, s 100, seed 0) of the patient-versus-control corpus. One worker per GPU, each starting a job only
# when its GPU is idle, so this waits for the task model's refit. Run on beast from work/patient-control-at.
cd /data2/james/construct-claims/work/patient-control-at || exit 1
export PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; CV=/data2/james/construct-claims/work/pilot-results/code/cv_abs.py
D=fits/ksweep; Q=$D/queue.txt; mkdir -p $D
if [ ! -f $Q ]; then
  { for k in 12 20 8 16 25 30 40; do echo "nested $k"; done; for k in 12 20; do echo "flat $k"; done; } > $Q
fi
idle() {
  for t in 1 2 3; do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)" -lt 1000 ] || return 1
    [ $t -lt 3 ] && sleep 40
  done
  return 0
}
worker() {
  local g=$1
  while true; do
    until idle $g; do sleep 120; done
    J=$(flock $Q.lock sh -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$J" ] && break
    read -r s k <<< "$J"
    CUDA_VISIBLE_DEVICES=$g $PY $CV data_all/corpus_pc_$s.json $k $D/cv_${s}_K$k.npy --restarts 2 --iters 1000 --s 100 --seed 0 --devices cuda:0 > $D/cv_${s}_K$k.log 2>&1
    echo "$(date -Is) gpu $g $s K$k exit $?"
  done
}
for g in 0 1 2 3; do worker $g & sleep 5; done; wait
echo "$(date -Is) K SWEEP DONE"
