# TSMTrack

This repository contains the model code and execution scripts for RGB-Thermal (RGBT) object tracking.

## Overview

TSMTrack processes visible and infrared frames with a shared tracking backbone. The current configuration uses temporal attention to combine search features from the current frame and selected historical frames. A confidence threshold controls which frames enter the history, while a sparse keep ratio controls how many search patches participate in temporal attention.

The repository provides the training configuration, checkpoint loading, benchmark testing script, and local path setup needed to run the model.

## Repository Structure

```text
.
|-- README.md
|-- install_tsmtrack.sh
|-- train.sh
|-- test.sh
|-- experiments/
|   `-- tsmtrack/tsmtrack.yaml
|-- tracking/
|   |-- create_default_local_file.py
|   `-- train.py
|-- RGBT_workspace/
|   `-- test_rgbt_mgpus.py
`-- lib/
    |-- models/
    |   |-- layers/temporal_attention.py
    |   `-- tsmtrack/tsmtrack.py
    |-- train/
    |   `-- dataset/lasher.py
    `-- test/
        `-- tracker/tsmtrack.py
```

## Code Description

- `experiments/tsmtrack/tsmtrack.yaml` defines the dataset, model, training, and testing settings.
- `lib/models/tsmtrack/tsmtrack.py` builds the tracker and applies sparse temporal attention to search features.
- `lib/train/` contains the training pipeline; `tracking/train.py` starts a training run.
- `lib/test/tracker/tsmtrack.py` runs tracking on a sequence; `RGBT_workspace/test_rgbt_mgpus.py` runs benchmark tests and saves predictions.

## Installation

Python 3.10 and a CUDA-enabled environment are used by the supplied installation script.

```bash
conda create -n TSMTrack python=3.10
conda activate TSMTrack
bash install_tsmtrack.sh
```

The script installs PyTorch 2.3.0 with CUDA 12.1 and requires `sudo` for `libturbojpeg0`. Install a Mamba/selective-scan backend compatible with your PyTorch and CUDA versions separately; it is not installed by this script.

## Data Format

The default training configuration uses the LasHeR training and validation splits. Each training sequence is expected under the configured LasHeR training root with visible and infrared frames and an `init.txt` annotation file. The benchmark testing script reads the LasHeR testing sequences from a separate root and expects `visible.txt` and `infrared.txt` annotations.

```text
LasHeR/
|-- TrainingSet/trainingset/<sequence>/
|   |-- visible/
|   |-- infrared/
|   `-- init.txt
`-- TestingSet/testingset/<sequence>/
    |-- visible/
    |-- infrared/
    |-- visible.txt
    `-- infrared.txt
```

Other supported test datasets are GTOT, RGBT210, RGBT234, VTUAV-ST, and VTUAV-LT. Set each dataset's `seq_home` in `RGBT_workspace/test_rgbt_mgpus.py` to its location on your machine. The script currently contains machine-specific paths.

## Pretrained Weights

The pretrained weight is `DropTrack.pth.tar`. Set its path in `MODEL.PRETRAIN_FILE` in `experiments/tsmtrack/tsmtrack.yaml`.

## Training

Generate the local path files and check that the LasHeR training root is correct:

```bash
python tracking/create_default_local_file.py \
  --workspace_dir . --data_dir ./data --save_dir ./output
```

```bash
bash train.sh
```

`train.sh` starts a two-GPU run in the background. It writes the console log to `train.log` and saves checkpoints under `output/checkpoints/train/tsmtrack/tsmtrack/`. Change `--nproc_per_node` in the script to match the number of GPUs you want to use.

## Testing

Update the dataset root in `RGBT_workspace/test_rgbt_mgpus.py` and confirm that the requested checkpoint exists. For a single LasHeR checkpoint:

```bash
python RGBT_workspace/test_rgbt_mgpus.py \
  --script_name tsmtrack \
  --yaml_name tsmtrack \
  --config_name tsmtrack \
  --dataset_name LasHeR \
  --epoch 20 \
  --threads 1 --num_gpus 1 --mode sequential \
  --temporal_enter_threshold 0.45 \
  --sparse_keep_ratio 0.75
```

The example saves one prediction file per sequence under `RGBT_workspace/results/LasHeR/LasHeR_0.45_skr0.75/tsmtrack_20/`. The sparse keep ratio must be in `(0, 1]`. Both testing arguments override the values in the YAML for that run.

To run the checkpoint ranges configured in `test.sh`:

```bash
bash test.sh
```

The current script tests LasHeR epochs 20–25 and RGBT210 epochs 5–25 with threshold `0.45` and sparse keep ratio `0.75`. Edit the dataset, epoch range, GPU count, and worker count in `test.sh` before running a different setup.
