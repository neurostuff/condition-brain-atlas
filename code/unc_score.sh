#!/bin/bash
# scores the uncertainty fits and pools. usage: bash code/unc_score.sh OUTDIR MODES [score options]   e.g. fits/unc/r08 post --rho 0.08
cd /data2/james/construct-claims/work/patient-control-at || exit 1
o=$1; modes=$2; shift 2; mkdir -p $o
for m in ${modes//,/ }; do for f in fits/unc/*_$m.npz; do b=$(basename ${f%.npz})
  PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src CUDA_VISIBLE_DEVICES= /data/james/vbm-meta-20261108/cbmr-gpu/bin/python code/uncertainty.py score data_all/corpus_pc_multimodal_fam.json $f $o/${b}_score.json "$@" > $o/${b}_score.log 2>&1 &
done; done; wait
PYTHONPATH=code:/data/james/agents/dev/task-networks-repro/repo/src CUDA_VISIBLE_DEVICES= /data/james/vbm-meta-20261108/cbmr-gpu/bin/python code/unc_report.py $o $o/report.json $modes 2>&1 | grep -av "INFO\|Warn" > $o/report.txt
