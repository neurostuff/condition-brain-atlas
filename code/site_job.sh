#!/bin/bash
# The atlas page's data with intervals: the within-paper correlation from the all-papers bootstrap, then the export.
# usage: bash code/site_job.sh GPU OUT_DIR      (after: bash code/unc_job.sh GPU all 0 boot 50, with --save-coef)
cd /data2/james/construct-claims/work/patient-control-at || exit 1
g=$1; out=$2; py=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; J=/data2/james/construct-claims/atlases/JHU
export PC_THREADS=6 OMP_NUM_THREADS=6 PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src CUDA_VISIBLE_DEVICES=$g
$py code/uncertainty.py score data_all/corpus_pc_multimodal_fam.json fits/unc/all0_boot.npz fits/unc/cov/all0_boot_score.json > fits/unc/cov/all0_boot_score.log 2>&1 || exit 1
$py code/export_site_hyp.py data_all/corpus_pc_multimodal_fam.json data_all/vocabulary.json $out $J/JHU-ICBM-labels-2mm.nii.gz $J/JHU-labels.xml fits/fast/dsm_pen5.json fits/fast/dsm_pen5_cell.npy \
  fits/unc/all0_boot_coef.npz fits/unc/cov/all0_boot_score.json fits/unc/cov/report.json > $out.log 2>&1
echo "site_job $out exit $?" >> fits/unc/done.txt
