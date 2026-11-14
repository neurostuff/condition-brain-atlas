#!/bin/bash
# One cross-validation job of pc_model.py on a given GPU (may share the card: chunks are sized to its free memory).
# usage: bash code/cv_job.sh GPU CORPUS_STRUCTURE BLOCKS K [LAM]       run on beast from work/patient-control-at
cd /data2/james/construct-claims/work/patient-control-at || exit 1
export PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src CUDA_VISIBLE_DEVICES=$1
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; mkdir -p fits/cv; t=$2_$3_K$4${5:+_lam$5}
$PY code/pc_model.py cv data_all/corpus_pc_$2.json $4 fits/cv/$t.json --blocks $3 ${5:+--lam $5} > fits/cv/$t.log 2>&1
echo "$(date -Is) gpu $1 $t exit $?"
