#!/bin/bash
# The atlas page's data with intervals, on the corpus rebuilt on 2026-11-15 (data_v2).
# usage: bash code/site_job.sh GPU OUT_DIR
# after: bash code/v2_job.sh GPU prim data_v2/corpus_pc_multimodal_fam.json --variant full+dirsplit+meaning --pen fam=5 --tests paper,cell,future
#        bash code/unc_job.sh GPU all 0 boot 50 --save-coef ; the fold fits (paper 0-4, cell 0-2) ; bash code/unc_score.sh fits/unc_v2/cov boot
cd /data2/james/construct-claims/work/patient-control-at || exit 1
g=$1; out=$2; py=/data/james/vbm-meta-20261108/cbmr-gpu/bin/python; J=/data2/james/construct-claims/atlases/JHU; D=data_v2; U=fits/unc_v2
export PC_THREADS=6 OMP_NUM_THREADS=6 PYTHONHASHSEED=0 PYTHONPATH=/data/james/agents/dev/task-networks-repro/repo/src CUDA_VISIBLE_DEVICES=$g
$py code/uncertainty.py score $D/corpus_pc_multimodal_fam.json $U/all0_boot.npz $U/cov/all0_boot_score.json > $U/cov/all0_boot_score.log 2>&1 || exit 1
$py code/export_site_hyp.py $D/corpus_pc_multimodal_fam.json $D/vocabulary.json $out $J/JHU-ICBM-labels-2mm.nii.gz $J/JHU-labels.xml fits/v2/prim.json fits/v2/prim_cell.npy \
  $U/all0_boot_coef.npz $U/cov/all0_boot_score.json $U/cov/report.json > $out.log 2>&1
echo "site_job $out exit $?" >> $U/done.txt
