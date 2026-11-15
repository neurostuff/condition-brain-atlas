#!/bin/bash
# One run of checks.py additive on a given corpus (the 2026-11-15 rebuild). usage: bash code/v2_job.sh GPU TAG CORPUS [checks.py options...]
cd /data2/james/construct-claims/work/patient-control-at || exit 1
g=$1; tag=$2; corpus=$3; shift 3; mkdir -p fits/v2
PC_THREADS=6 OMP_NUM_THREADS=6 PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src CUDA_VISIBLE_DEVICES=$g \
  /data/james/vbm-meta-20261108/cbmr-gpu/bin/python code/checks.py additive $corpus fits/v2/$tag.json --window-from foci --lam 20 --save-scores "$@" > fits/v2/$tag.log 2>&1
grep -a " paper: \| cell: \| future \| inner: \|Error\|error" fits/v2/$tag.log | sed "s/^/$tag: /" | cut -c1-560 >> fits/v2/log.txt
echo "$tag done" >> fits/v2/done.txt
