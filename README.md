# causal-imt
This is the official implementation of the paper "A Causality-Inspired Model for Intima-Media Thickening Assessment in Ultrasound Videos". The code will be released soon！

## Introduction

Carotid atherosclerosis represents a significant health risk, with its early diagnosis primarily dependent on ultrasound-based assessments of carotid intima-media thickening. However, during carotid ultrasound screening, significant view variations cause style shifts, impairing content cues related to thickening, such as lumen anatomy, which introduces spurious correlations that hinder assessment. Therefore, we propose a novel causal-inspired method for assessing carotid intima-media thickening in frame-wise ultrasound videos, which focuses on two aspects: eliminating spurious correlations caused by style and enhancing causal content correlations. Specifically, we introduce a novel Spurious Correlation Elimination (SCE) module to remove non-causal style effects by enforcing prediction invariance with style perturbations. Simultaneously, we propose a Causal Equivalence Consolidation (CEC) module to strengthen causal content correlation through adversarial optimization during content randomization. Simultaneously, we design a Causal Transition Augmentation (CTA) module to ensure smooth causal flow by integrating an auxiliary pathway with text prompts and connecting it through contrastive learning. The experimental results on our in-house carotid ultrasound video dataset achieved an accuracy of 86.93\%, demonstrating the superior performance of the proposed method.

![image](https://github.com/xielaobanyy/causal-imt/blob/main/models/fig1-new.jpg)


## Using the code:

The code is stable while using Python 3.9.0, CUDA >= 11.3

- Clone this repository:
```bash
git clone https://github.com/xielaobanyy/causal-imt
cd causal-imt
```

To install all the dependencies :

```bash
conda env create causal-imt python==3.9.0
conda activate causal-imt
pip install -r requirements.txt
```
## Data Format

```
dataset
└── captions
    ├── cap.bifur.train.json
    ├── cap.bifur.val.json
    ├── cap.bifur.test.json
└── image_splits
    ├── split.bifur.train.json
    ├── split.bifur.val.json
    ├── split.bifur.test.json
└── images
      |   ├── 20190720094945_0953510 frame_0000.jpg
      |   ├── 20190720094945_0953510 frame_0001.jpg
      |   ├── 20190720094945_0953510 frame_0002.jpg
      |   ├── ...
```

## Training and Validation

1. Train the model.
```
python .\src\clip_fine_tune.py
```
2. Evaluate.
```
python .\src\test.py   
```

### Acknowledgements：
This code-base uses certain code-blocks and helper functions from CLIP4Cir [Link](https://github.com/ABaldrati/CLIP4Cir).

