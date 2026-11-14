#!/bin/bash
# Calibration options scored on both held-out tests (by paper: cv; condition x measure: cellcv), windowed model with a
# flat background network. usage: bash code/calib.sh GPU K NAME [pc_model options...]   (run on beast from work/patient-control-at)
g=$1; K=$2; name=$3; shift 3
W="--blocks measure_dir --model window --flat-bg --window measure --restarts 1 --gpu-update"
bash code/job.sh $g calib_K${K}_${name}_cv cv multimodal_dir $K $W "$@"
bash code/job.sh $g calib_K${K}_${name}_cell cellcv multimodal_dir $K $W --rounds 3 "$@"
