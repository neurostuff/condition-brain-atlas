#!/bin/bash
# Refit the atlas model on the real corpus and on shuffled diagnoses (seeds 1-10 of each null, the relabellings of fits/null),
# saving coefficients for the per-cell peak test (null_peak.py). usage: bash code/null_coef.sh GPU OUT_DIR
cd /data2/james/construct-claims/work/patient-control-at || exit 1
g=$1; out=$2; py=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; mkdir -p $out
export PC_THREADS=6 OMP_NUM_THREADS=6 PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src CUDA_VISIBLE_DEVICES=$g SAVE_COEF=1
C="data_v2/corpus_pc_multimodal_fam.json data_v2/vocabulary.json $out"
$py code/null_atlas.py $C real 0 > $out/real.log 2>&1 || { echo "real failed" >> $out/done.txt; exit 1; }
for s in 1 2 3 4 5 6 7 8 9 10; do
  for mode in any class; do $py code/null_atlas.py $C $mode $s >> $out/$mode.log 2>&1 || echo "$mode $s failed" >> $out/done.txt; done
done
echo "null_coef done" >> $out/done.txt
