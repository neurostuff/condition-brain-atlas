#!/bin/bash
# One pc_model.py job on a given GPU (may share the card). usage: bash code/job.sh GPU TAG CMD CORPUS K [pc_model args...]
# writes fits/cv/TAG.json and .log; a line in logs/cv_jobs.log when it ends. Run on beast from work/patient-control-at.
cd /data2/james/construct-claims/work/patient-control-at || exit 1
export PC_THREADS=${PC_THREADS:-6} OMP_NUM_THREADS=${PC_THREADS:-6} MKL_NUM_THREADS=${PC_THREADS:-6} PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src CUDA_VISIBLE_DEVICES=$1
gpu=$1; tag=$2; cmd=$3; corpus=$4; K=$5; shift 5; mkdir -p fits/cv
/data/james/vbm-meta-20261108/cbmr-gpu/bin/python code/pc_model.py $cmd data_all/corpus_pc_$corpus.json $K fits/cv/$tag.json "$@" > fits/cv/$tag.log 2>&1
echo "$(date -Is) gpu $gpu $tag exit $?" >> logs/cv_jobs.log
