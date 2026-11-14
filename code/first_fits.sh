#!/bin/bash
# First fits of the patient-versus-control corpus, on the CPU while the GPUs are on the task model's refit
# (2026-11-13). One Gibbs + EM fit (the task model's gibbs_init.py, 1000 EM iterations at most) per (structure, K).
# usage: bash code/first_fits.sh "nested flat" "10 15"      run on beast from work/patient-control-at
cd /data2/james/construct-claims/work/patient-control-at || exit 1
export PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src GIBBS_DEVICE=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12
PY=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; G=/data2/james/construct-claims/work/pilot-results/code/gibbs_init.py
for s in $1; do for k in $2; do
  d=fits/first_${s}_K$k; mkdir -p $d
  $PY $G fit data_all/corpus_pc_$s.json $k $d/gibbs_s100 100 1000 > $d/gibbs_s100.log 2>&1
  echo "$(date -Is) $s K$k exit $?"
done; done
echo "$(date -Is) FIRST FITS DONE"
