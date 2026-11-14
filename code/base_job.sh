#!/bin/bash
# One baseline-question run of checks.py additive (MODEL_PLAN.md). usage: bash code/base_job.sh GPU TAG [checks.py options...]
cd /data2/james/construct-claims/work/patient-control-at || exit 1
g=$1; tag=$2; shift 2
PC_THREADS=6 OMP_NUM_THREADS=6 PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src CUDA_VISIBLE_DEVICES=$g \
  /data/james/vbm-meta-20261108/cbmr-gpu/bin/python code/checks.py additive data_all/corpus_pc_multimodal_fam.json fits/base/$tag.json "$@" 2>&1 | grep -a "^full lam\|Error\|error" | sed "s/^/$tag: /" | cut -c1-420 >> fits/base/log.txt
echo "$tag done" >> fits/base/done.txt
