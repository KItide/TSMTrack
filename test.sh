#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SCRIPT_NAME="tsmtrack"
YAML_NAME="tsmtrack"
CONFIG_NAME="tsmtrack"
DATASET_NAME="VTUAVST"
THREADS=16
NUM_GPUS=2
START_EPOCH=10
END_EPOCH=14
THRESHOLD_START=0.45
THRESHOLD_END=0.45
THRESHOLD_STEP=0.05

# 可写多个值，例如：SPARSE_KEEP_RATIOS=(0.25 0.5 0.75 1.0)
SPARSE_KEEP_RATIOS=(0.75)

# for thr in $(seq "$THRESHOLD_START" "$THRESHOLD_STEP" "$THRESHOLD_END")
# do
#     for ((runid=START_EPOCH; runid<=END_EPOCH; runid++))
#     do
#         python ./RGBT_workspace/test_rgbt_mgpus.py \
#             --script_name "$SCRIPT_NAME" \
#             --yaml_name "$YAML_NAME" \
#             --config_name "$CONFIG_NAME" \
#             --dataset_name "$DATASET_NAME" \
#             --threads "$THREADS" \
#             --num_gpus "$NUM_GPUS" \
#             --epoch "$runid" \
#             --mode parallel \
#             --temporal_enter_threshold "$thr"
#         echo "Command for runid $runid threshold $thr executed successfully"
#     done
# done



CONFIG_NAME="tsmtrack"
DATASET_NAME="LasHeR"
THREADS=5
NUM_GPUS=1
START_EPOCH=20
END_EPOCH=25
THRESHOLD_START=0.45
THRESHOLD_END=0.45
THRESHOLD_STEP=0.05

for sparse_keep_ratio in "${SPARSE_KEEP_RATIOS[@]}"
do
    for thr in $(seq "$THRESHOLD_START" "$THRESHOLD_STEP" "$THRESHOLD_END")
    do
        for ((runid=START_EPOCH; runid<=END_EPOCH; runid++))
        do
            python ./RGBT_workspace/test_rgbt_mgpus.py \
                --script_name "$SCRIPT_NAME" \
                --yaml_name "$YAML_NAME" \
                --config_name "$CONFIG_NAME" \
                --dataset_name "$DATASET_NAME" \
                --threads "$THREADS" \
                --num_gpus "$NUM_GPUS" \
                --epoch "$runid" \
                --mode parallel \
                --temporal_enter_threshold "$thr" \
                --sparse_keep_ratio "$sparse_keep_ratio"
            echo "Command for runid $runid threshold $thr sparse_keep_ratio $sparse_keep_ratio executed successfully"
        done
    done
done


CONFIG_NAME="tsmtrack"
DATASET_NAME="RGBT210"
THREADS=5
NUM_GPUS=1
START_EPOCH=5
END_EPOCH=25
THRESHOLD_START=0.45
THRESHOLD_END=0.45
THRESHOLD_STEP=0.05

for sparse_keep_ratio in "${SPARSE_KEEP_RATIOS[@]}"
do
    for thr in $(seq "$THRESHOLD_START" "$THRESHOLD_STEP" "$THRESHOLD_END")
    do
        for ((runid=START_EPOCH; runid<=END_EPOCH; runid++))
        do
            python ./RGBT_workspace/test_rgbt_mgpus.py \
                --script_name "$SCRIPT_NAME" \
                --yaml_name "$YAML_NAME" \
                --config_name "$CONFIG_NAME" \
                --dataset_name "$DATASET_NAME" \
                --threads "$THREADS" \
                --num_gpus "$NUM_GPUS" \
                --epoch "$runid" \
                --mode parallel \
                --temporal_enter_threshold "$thr" \
                --sparse_keep_ratio "$sparse_keep_ratio"
            echo "Command for runid $runid threshold $thr sparse_keep_ratio $sparse_keep_ratio executed successfully"
        done
    done
done

# CONFIG_NAME="tsmtrack"
# DATASET_NAME="LasHeR"
# THREADS=10
# NUM_GPUS=1
# START_EPOCH=20
# END_EPOCH=25
# THRESHOLD_START=0.45
# THRESHOLD_END=0.45
# THRESHOLD_STEP=0.05

# for sparse_keep_ratio in "${SPARSE_KEEP_RATIOS[@]}"
# do
#     for thr in $(seq "$THRESHOLD_START" "$THRESHOLD_STEP" "$THRESHOLD_END")
#     do
#         for ((runid=START_EPOCH; runid<=END_EPOCH; runid++))
#         do
#             python ./RGBT_workspace/test_rgbt_mgpus.py \
#                 --script_name "$SCRIPT_NAME" \
#                 --yaml_name "$YAML_NAME" \
#                 --config_name "$CONFIG_NAME" \
#                 --dataset_name "$DATASET_NAME" \
#                 --threads "$THREADS" \
#                 --num_gpus "$NUM_GPUS" \
#                 --epoch "$runid" \
#                 --mode parallel \
#                 --temporal_enter_threshold "$thr" \
#                 --sparse_keep_ratio "$sparse_keep_ratio"
#             echo "Command for runid $runid threshold $thr sparse_keep_ratio $sparse_keep_ratio executed successfully"
#         done
#     done
# done

# CONFIG_NAME="tsmtrack"
# DATASET_NAME="RGBT210"
# THREADS=32
# NUM_GPUS=4
# START_EPOCH=20
# END_EPOCH=25
# THRESHOLD_START=0.4
# THRESHOLD_END=0.5
# THRESHOLD_STEP=0.05

# for thr in $(seq "$THRESHOLD_START" "$THRESHOLD_STEP" "$THRESHOLD_END")
# do
#     for ((runid=START_EPOCH; runid<=END_EPOCH; runid++))
#     do
#         python ./RGBT_workspace/test_rgbt_mgpus.py \
#             --script_name "$SCRIPT_NAME" \
#             --yaml_name "$YAML_NAME" \
#             --config_name "$CONFIG_NAME" \
#             --dataset_name "$DATASET_NAME" \
#             --threads "$THREADS" \
#             --num_gpus "$NUM_GPUS" \
#             --epoch "$runid" \
#             --mode parallel \
#             --temporal_enter_threshold "$thr"
#     done
# done

# CONFIG_NAME="tsmtrack"
# DATASET_NAME="LasHeR"
# THREADS=32
# NUM_GPUS=4
# START_EPOCH=17
# END_EPOCH=25
# THRESHOLD_START=0.3
# THRESHOLD_END=0.3
# THRESHOLD_STEP=0.05

# for thr in $(seq "$THRESHOLD_START" "$THRESHOLD_STEP" "$THRESHOLD_END")
# do
#     for ((runid=START_EPOCH; runid<=END_EPOCH; runid++))
#     do
#         python ./RGBT_workspace/test_rgbt_mgpus.py \
#             --script_name "$SCRIPT_NAME" \
#             --yaml_name "$YAML_NAME" \
#             --config_name "$CONFIG_NAME" \
#             --dataset_name "$DATASET_NAME" \
#             --threads "$THREADS" \
#             --num_gpus "$NUM_GPUS" \
#             --epoch "$runid" \
#             --mode parallel \
#             --temporal_enter_threshold "$thr"
#         echo "Command for runid $runid threshold $thr executed successfully"
#     done
# done


# CONFIG_NAME="tsmtrack"
# DATASET_NAME="RGBT234"
# THREADS=32
# NUM_GPUS=4
# START_EPOCH=20
# END_EPOCH=25
# THRESHOLD_START=0.3
# THRESHOLD_END=0.3
# THRESHOLD_STEP=0.05

# for thr in $(seq "$THRESHOLD_START" "$THRESHOLD_STEP" "$THRESHOLD_END")
# do
#     for ((runid=START_EPOCH; runid<=END_EPOCH; runid++))
#     do
#         python ./RGBT_workspace/test_rgbt_mgpus.py \
#             --script_name "$SCRIPT_NAME" \
#             --yaml_name "$YAML_NAME" \
#             --config_name "$CONFIG_NAME" \
#             --dataset_name "$DATASET_NAME" \
#             --threads "$THREADS" \
#             --num_gpus "$NUM_GPUS" \
#             --epoch "$runid" \
#             --mode parallel \
#             --temporal_enter_threshold "$thr"
#         echo "Command for runid $runid threshold $thr executed successfully"
#     done
# done
