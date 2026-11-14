#!/bin/bash
# K search, one K at a time: both held-out tests (code/calib.sh) and one full-data fit (for concentration and unused
# networks). usage: bash code/ksearch.sh GPU SEED K [K ...]     (K = learned networks; the fit has K + 1 with the flat one)
# Setting (2026-11-14): counts in peaks, S 10, ETA 0.01 (lighter smoothing: conditions stay mixtures), warm-up 30.
cd /data2/james/construct-claims/work/patient-control-at || exit 1
g=$1; seed=$2; shift 2; B="--count-scale 0.0185 --S 10 --eta 0.01 --warm 30 --seed $seed"; mkdir -p fits/ksearch
for K in "$@"; do
  bash code/calib.sh $g $((K + 1)) ks_s$seed $B
  PC_THREADS=6 OMP_NUM_THREADS=6 PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src CUDA_VISIBLE_DEVICES=$g \
    /data/james/vbm-meta-20261108/cbmr-gpu/bin/python code/pc_model.py fit data_all/corpus_pc_multimodal_dir.json $((K + 1)) fits/ksearch/K${K}_s$seed \
    --blocks measure_dir --model window --flat-bg --window measure --restarts 1 --gpu-update $B > fits/ksearch/K${K}_s$seed.log 2>&1
  echo "$(date -Is) gpu $g ksearch K$K s$seed done" >> logs/cv_jobs.log
done
