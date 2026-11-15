#!/bin/bash
# One uncertainty.py fit on one GPU. usage: bash code/unc_job.sh GPU TEST PART MODE DRAWS [--save-coef]
cd /data2/james/construct-claims/work/patient-control-at || exit 1
g=$1; t=$2; p=$3; m=$4; n=$5; mkdir -p fits/unc
PC_THREADS=6 OMP_NUM_THREADS=6 PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src CUDA_VISIBLE_DEVICES=$g \
  /data/james/vbm-meta-20261108/cbmr-gpu/bin/python code/uncertainty.py fit data_all/corpus_pc_multimodal_fam.json fits/unc/${t}${p}_${m}.npz --test $t --part $p --mode $m --draws $n $6 > fits/unc/${t}${p}_${m}.log 2>&1
echo "${t}${p}_${m} done" >> fits/unc/done.txt
