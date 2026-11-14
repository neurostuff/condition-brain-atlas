#!/bin/bash
# One run of checks.py additive for the additions in MODEL_PLAN.md. usage: bash code/add_job.sh GPU TAG [checks.py options...]
cd /data2/james/construct-claims/work/patient-control-at || exit 1
g=$1; tag=$2; shift 2
PC_THREADS=6 OMP_NUM_THREADS=6 PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src CUDA_VISIBLE_DEVICES=$g \
  /data/james/vbm-meta-20261108/cbmr-gpu/bin/python code/checks.py additive data_all/corpus_pc_multimodal_fam.json fits/add/$tag.json --window-from foci "$@" 2>&1 | grep -a " paper: \| cell: \| future \| ic: \| inner: \|Error\|error" | sed "s/^/$tag: /" | cut -c1-520 >> fits/add/log.txt
echo "$tag done" >> fits/add/done.txt
