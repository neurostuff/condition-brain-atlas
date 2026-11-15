#!/bin/bash
# scores the uncertainty fits and pools. usage: [CORPUS=...] [UNC=fits/unc_v2] bash code/unc_score.sh OUTDIR MODES [score options]   e.g. fits/unc_v2/cov boot
cd /data2/james/construct-claims/work/patient-control-at || exit 1
o=$1; modes=$2; shift 2; mkdir -p $o; C=${CORPUS:-data_v2/corpus_pc_multimodal_fam.json}; U=${UNC:-fits/unc_v2}
for m in ${modes//,/ }; do for f in $U/*_$m.npz; do b=$(basename ${f%.npz}); case $b in *_coef|all*) continue;; esac
  PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src CUDA_VISIBLE_DEVICES= /data/james/vbm-meta-20261108/cbmr-gpu/bin/python code/uncertainty.py score $C $f $o/${b}_score.json "$@" > $o/${b}_score.log 2>&1 &
done; done; wait
PYTHONPATH=code:/data/james/agents/dev/task-networks-repro/repo/src CUDA_VISIBLE_DEVICES= /data/james/vbm-meta-20261108/cbmr-gpu/bin/python code/unc_report.py $o $o/report.json $modes 2>&1 | grep -av "INFO\|Warn" > $o/report.txt
