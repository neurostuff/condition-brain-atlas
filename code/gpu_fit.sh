#!/bin/bash
# One Gibbs + EM fit of a patient-versus-control corpus on a given GPU (shared with another job: the model sizes
# its chunks to 60% of the card's free memory at start).
# usage: bash code/gpu_fit.sh GPU STRUCTURE K SEED [TAG]        run on beast from work/patient-control-at
cd /data2/james/construct-claims/work/patient-control-at || exit 1
export PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src CUDA_VISIBLE_DEVICES=$1 GIBBS_DEVICE=cuda:0
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; G=/data2/james/construct-claims/work/pilot-results/code/gibbs_init.py
d=fits/${5:-first}_$2_K$3; mkdir -p $d
$PY $G fit data_all/corpus_pc_$2.json $3 $d/gibbs_s$4 $4 1000 > $d/gibbs_s$4.log 2>&1
echo "$(date -Is) gpu $1 $2 K$3 seed $4 exit $?"
