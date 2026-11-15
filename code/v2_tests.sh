#!/bin/bash
# The label comparison of the 2026-11-15 rebuild on GPUs 1 to 3 (GPU 0: v2_job.sh 0 all data_v2/all/... the earlier labels).
cd /data2/james/construct-claims/work/patient-control-at || exit 1
C=data_v2/corpus_pc_multimodal_fam.json; V="--variant full+dirsplit+meaning"
(bash code/v2_job.sh 1 prim $C $V --pen fam=5 --tests paper,cell,future; bash code/v2_job.sh 1 primi $C $V --pen fam=5 --tests inner) &
(bash code/v2_job.sh 2 also20 $C $V+also --pen fam=5 --tests paper,cell,future; bash code/v2_job.sh 2 also80 $C $V+also --pen fam=5,also=80 --tests paper,inner) &
(bash code/v2_job.sh 3 also5 $C $V+also --pen fam=5,also=5 --tests paper,inner; bash code/v2_job.sh 3 also20i $C $V+also --pen fam=5 --tests inner) &
wait
