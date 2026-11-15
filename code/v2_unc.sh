#!/bin/bash
# Every bootstrap fit of the 2026-11-15 rebuild, four GPUs: all papers (with coefficients), paper folds 0-4, held-out-measure rounds 0-2.
cd /data2/james/construct-claims/work/patient-control-at || exit 1
(bash code/unc_job.sh 0 all 0 boot 50 --save-coef; bash code/unc_job.sh 0 paper 3 boot 50; bash code/unc_job.sh 0 cell 2 boot 50) &
(bash code/unc_job.sh 1 paper 0 boot 50; bash code/unc_job.sh 1 paper 4 boot 50) &
(bash code/unc_job.sh 2 paper 1 boot 50; bash code/unc_job.sh 2 cell 0 boot 50) &
(bash code/unc_job.sh 3 paper 2 boot 50; bash code/unc_job.sh 3 cell 1 boot 50) &
wait; echo "all fits done" >> fits/unc_v2/done.txt
