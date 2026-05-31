# Master Thesis — *Leveraging Transformers for Multi-Feature Context-Aware Multi-Object Tracking*

This repository contains the code accompanying a master’s thesis investigating the limitations of learned association in online Multi-Object Tracking (MOT), using CAMELTrack as a baseline and extending it with new analysis tools and architectures.

---

# Abstract

Multi-Object Tracking (MOT) is a core computer vision task with applications in domains such as autonomous driving and sports analytics. While detection performance is now close to saturation, association—linking detections across frames—remains the primary source of errors in online tracking-by-detection (TbD) systems.

This thesis studies the limitations of learned association, specifically on the CAMELTrack tracker, through a systematic failure-case analysis. We identified two key issues: (i) a tracklet embedding drift under missing observations, that degrades identity recovery after long occlusions, and (ii) a limited exploitation of heterogeneous cues, due to early fusion into a single representation.

To address these limitations, we propose two contributions. (i) A temporal augmentation, Detection-Window Dropout that simulates long-term missing observations but improves robustness only in targeted scenarios and shows limited generalization. (ii) A Context-Cue-Aware Association Architecture Paradigm that preserves cue-specific representations and performs pair-dependent scoring, enabling adaptive cue re-weighting for each tracklet-detection pair. A proposed architecture following that paradigm achieves state-of-the-art performances by improving HOTA by +2.79% and +4.82% on SportsMOT and DanceTrack, evaluated on custom train/val/test splits.

Through this work, we demonstrate that substantial gains in online MOT can be achieved without improving detection, by focusing on robustness to missing observations, deeper representations, and pair-dependent association mechanisms

---

# Repository Scope

This repository serves as the **experimental backbone of the thesis**, containing:

- an extended version of the CAMELTrack pipeline (`camelv2`)
- oracle-based analysis tools to measure upper bounds on association
- a complete suite of failure-case diagnostics used to guide model design

The goal is not only to improve performance, but to provide **mechanisms to understand why association fails**, and how to fix it.

---

# CAMELv2: Association-Focused Extensions

The main implementation extends CAMELTrack with architectures designed to better exploit multi-cue information.

The key idea is to move away from **single-embedding association**, and instead:

- preserve **cue-specific representations** as long as possible
- perform **pair-dependent scoring** between tracklets and detections

This results in a paradigm where the association decision is no longer fixed, but adapts to the specific pair and context.

In practice, this repository includes:

- cue-preserving encoders (e.g., cue-wise transformers)
- learned pairwise association modules
- modifications of Association-Centric Training (ACT)

---

# Oracle Modules

To understand the limits of the system independently of model design, this repository includes two oracle implementations:

### Association Oracle
Simulates perfect association using ground truth assignments to establish an **upper bound on achievable performance** given detections.

### Cue-Fusion Oracle
Evaluates the best possible association performance achievable by combining cues with optimal weights, revealing whether limitations arise from:
- the cues themselves
- or the association architecture

These tools are essential to interpret experimental results and justify architectural changes.

---

# Failure Analysis Toolkit

A central part of this repository is a collection of tools designed to dissect tracking behavior beyond aggregate metrics.

These include:

- **Per-sequence error analysis**, breaking down FN, FP, and ID switches
- **Spatial error heatmaps**, revealing where failures occur in the frame
- **Identity persistence timelines**, showing when identities break
- **t-SNE visualizations**, inspecting structure in embedding space
- **Embedding similarity diagnostics**, tracking representation drift over time

Together, these tools enable a detailed understanding of how and why association fails, and directly inform model improvements.

---

# ⚙️ Quick Installation Guide from originl CAMELTrack repository
CAMELTrack is built on top of [TrackLab](https://github.com/TrackingLaboratory/tracklab), a research framework for Multi-Object Tracking.

![Installation demo](media/cameltrack-demo3.gif)
### Clone the Repository & Install

First git clone this repository : 

```bash
git clone https://github.com/TrackingLaboratory/CAMELTrack.git
```

You can then choose to install using either [uv](https://docs.astral.sh/uv/getting-started/installation/)
or directly using pip (while managing your environment yourself).

#### [Recommended] Install using uv
1. Install uv : https://docs.astral.sh/uv/getting-started/installation/
2. Create a new virtual environment with a recent python version (>3.9) : 
```bash
cd cameltrack
uv venv --python 3.12
```

> [!NOTE]
> To use the virtual environment created by uv,
> you need to prefix all commands with `uv run`, as shown in the examples below.
> Using `uv run` will automatically download the dependencies the first time it is run. 

#### Install using pip
1. Move into the directory
```bash
cd cameltrack
```
2. Create a virtual environment (using by example: `conda`)
3. Install the dependencies inside the virtual environment :
```bash
pip install -e .
```

> [!NOTE]
> The following instructions use the uv installation, but you can just remove `uv run`
> from all commands.

### First Run

To demonstrate CAMELTrack, a default video will be automatically output during the first run:
```bash
uv run tracklab -cn cameltrack
```

### First Run

To demonstrate CAMELTrack, a default video will be automatically output during the first run:
```bash
uv run tracklab -cn cameltrack
```

### Updating
Please make sure to check the official GitHub regularly for updates.
To update this repository to its latest version, run `git pull` on the repository or `uv run -U tracklab -cn cameltrack` to update the dependencies.

### Data Preparation

You can use tracklab directly on `mp4` videos or image folders.
Or also download the desired datasets [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), 
[DanceTrack](https://drive.google.com/drive/folders/1ASZCFpPEfSOJRktR8qQ_ZoT9nZR0hOea), [SportsMOT](https://github.com/MCG-NJU/SportsMOT?tab=readme-ov-file#download),
[BEE24](https://holmescao.github.io/datasets/BEE24), or [PoseTrack21](https://github.com/anDoer/PoseTrack21) and place them in the `data/` directory.

### Off-the-shelf Model Weights and Outputs

#### Detections
The YOLOX detector weights used in the paper are available from [DiffMOT](https://github.com/Kroery/DiffMOT/releases). 
You can also directly use the detection text files from [DiffMOT](https://github.com/Kroery/DiffMOT) by placing them in the correct data directories.

#### Saved off-the-shelf model results
We also provide precomputed outputs (`Tracker States`) for various datasets in `Pickle` format on [Hugging Face](https://huggingface.co/trackinglaboratory/CAMELTrack/tree/main/states), so you don’t need to run the models yourself.

#### Off-the-shelf models
TrackLab also offers several ready-to-use models (detectors, pose estimators, reid and other trackers). To see all available configurations and options, run:
```bash
uv run tracklab --help
```

### 🏋️‍♀ CAMELTrack Model Weights
The pre-trained weights used to achieve state-of-the-art results in the paper are listed below. They are automatically downloaded when running CAMELTrack.

| Dataset     |     Appearance     |      Keypoints      |  HOTA  | Weights                                                                                                                                                                                                                                                                                                                              |
|:------------|:------------------:|:-------------------:|:------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DanceTrack  | :white_check_mark: |                     |  66.1  | [camel_bbox_app_dancetrack.ckpt](https://huggingface.co/trackinglaboratory/CAMELTrack/blob/main/camel_bbox_app_dancetrack.ckpt)                                                                                                                                                                                                      |
| DanceTrack  | :white_check_mark: | :white_check_mark:  |  69.3  | [camel_bbox_app_kps_dancetrack.ckpt](https://huggingface.co/trackinglaboratory/CAMELTrack/blob/main/camel_bbox_app_kps_dancetrack.ckpt)                                                                                                                                                                                              |
| SportsMOT   | :white_check_mark: | :white_check_mark:  |  80.3  | [camel_bbox_app_kps_sportsmot.ckpt](https://huggingface.co/trackinglaboratory/CAMELTrack/blob/main/camel_bbox_app_kps_sportsmot.ckpt)                                                                                                                                                                                                |
| MOT17       | :white_check_mark: | :white_check_mark:  |  62.4  | [camel_bbox_app_kps_mot17.ckpt](https://huggingface.co/trackinglaboratory/CAMELTrack/blob/main/camel_bbox_app_kps_mot17.ckpt)                                                                                                                                                                                                    |
| PoseTrack21 | :white_check_mark: | :white_check_mark:  |  66.0  | [camel_bbox_app_kps_posetrack24.ckpt](https://huggingface.co/trackinglaboratory/CAMELTrack/blob/main/camel_bbox_app_kps_posetrack24.ckpt)                                                                                                                                                                                                                                                                                              |
| BEE24       |                    |                     |  50.3  | [camel_bbox_bee24.ckpt](https://huggingface.co/trackinglaboratory/CAMELTrack/blob/main/camel_bbox_bee24.ckpt)                                                                                                                                                                                                                                                                                                            |

We also provide (by default) the weights [camel_bbox_app_kps_global.ckpt](https://huggingface.co/trackinglaboratory/CAMELTrack/blob/main/camel_bbox_app_kps_global.ckpt) trained jointly on MOT17, DanceTrack, SportsMOT, and PoseTrack21, suitable for testing purposes.

## 🎯 Tracking

Run the following command to track, for example, on DanceTrack, with the checkpoint obtained from training, or the provided
model weights (pretrained weights are downloaded automatically when using the name from the table above) :

```bash
uv run tracklab -cn cameltrack dataset=dancetrack dataset.eval_set=test modules.track.checkpoint_path=camel_bbox_app_kps_dancetrack.ckpt
```

By default, this will create a new directory inside `outputs/cameltrack` which will contain a visualization of the
output for each sequence, in addition to the tracking output in MOT format.

## 💪 Training

### Training on a default dataset

You first have to run the complete tracking pipeline (without tracking, with a pre-trained
CAMELTrack or with a SORT-based tracker, like oc-sort), on train, validation (and testing) sets
for the dataset you want to train, and save the "Tracker States":
```bash
uv run tracklab -cn cameltrack dataset=dancetrack dataset.eval_set=train
uv run tracklab -cn cameltrack dataset=dancetrack dataset.eval_set=val
uv run tracklab -cn cameltrack dataset=dancetrack dataset.eval_set=test
```
By default, they are saved in the `states/` directory.

You can also use the Tracker States we provide for the
common MOT datasets [on huggingface](https://huggingface.co/trackinglaboratory/CAMELTrack/tree/main/states).

Once you have the Tracker States, you can put them in the dataset directory
(in `data_dir`, by default `./data/$DATASET`) under the `states/` directory, with the following names :
```text
data/
    DanceTrack/
        train/
        val/
        test/
        states/
            train.pklz
            val.pklz
            test.pklz
```

Once you have the Tracker States, run the following command to train on a specific dataset
(by default, DanceTrack) : 
```bash
uv run tracklab -cn cameltrack_train dataset=dancetrack
```


> [!NOTE]
> You can always modify the configuration in [cameltrack.yaml](cameltrack/configs/cameltrack.yaml), and in the
> other files inside this directory, instead of passing these values in the command line.
> 
> For example, to change the dataset for training, you can modify [camel.yaml](cameltrack/configs/modules/track/camel.yaml).

By default, this will create a new directory inside `outputs/cameltrack_train`, which will contain the checkpoints
to the created models, which can then be used for tracking and evaluation, by setting
the `modules.track.checkpoint_path` configuration key in [camel.yaml](cameltrack/configs/modules/track/camel.yaml#L4).

### Training on a custom dataset
To train on a custom dataset, you'll have to integrate it in tracklab, either by using the MOT format, or by implementing
a new dataset class. Once that's done, you can modify [cameltrack.yaml](cameltrack/configs/cameltrack.yaml), to point to
the new dataset.

# CAMELTrack
This is build on top of CAMELTrack and TrackLab:
[CAMELTrack](https://arxiv.org/abs/2505.01257):
```
@misc{somers2025cameltrackcontextawaremulticueexploitation,
      title={CAMELTrack: Context-Aware Multi-cue ExpLoitation for Online Multi-Object Tracking}, 
      author={Vladimir Somers and Baptiste Standaert and Victor Joos and Alexandre Alahi and Christophe De Vleeschouwer},
      year={2025},
      eprint={2505.01257},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.01257}, 
}
```

[TrackLab](https://github.com/TrackingLaboratory/tracklab):
```
@misc{Joos2024Tracklab,
	title = {{TrackLab}},
	author = {Joos, Victor and Somers, Vladimir and Standaert, Baptiste},
	journal = {GitHub repository},
	year = {2024},
	howpublished = {\url{https://github.com/TrackingLaboratory/tracklab}}
}
```
