#!/bin/bash
# One run of checks.py additive (MODEL_PLAN.md), one process per GPU at a time. usage: bash code/fast_job.sh GPU TAG [checks.py options...]
cd /data2/james/construct-claims/work/patient-control-at || exit 1
g=$1; tag=$2; shift 2
PC_THREADS=6 OMP_NUM_THREADS=6 PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src CUDA_VISIBLE_DEVICES=$g \
  /data/james/vbm-meta-20261108/cbmr-gpu/bin/python code/checks.py additive data_all/corpus_pc_multimodal_fam.json fits/fast/$tag.json --save-scores "$@" 2>&1 | grep -a " paper: \| cell: \| future \| ic: \| inner: \|Error\|error" | sed "s/^/$tag: /" | cut -c1-560 >> fits/fast/log.txt
echo "$tag done" >> fits/fast/done.txt
