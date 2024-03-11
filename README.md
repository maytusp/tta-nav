## An official implementation of "TTA-Nav: Test-time Adaptive Reconstruction for Point-Goal Navigation under Visual Corruptions"
Read the preprint version at: https://arxiv.org/abs/2403.01977


## Requirements
1. Navigation agent is based on Habitat-Lab and Habitat-Baselines. Please follow the official instruction [here](https://github.com/facebookresearch/habitat-lab)
  Here we use: habitat-sim 0.3.0, habitat-lab 0.3.0, and habitat-baselines 0.3.0.

2. Top-down decoder is independent of habitat and can be trained using standard pytorch and pytorch lighting. \
   Here we adpat the code from [DiffuseVAE](https://github.com/kpandey008/DiffuseVAE) to train our top-down decoder.
   Dependency: pytorch-lightning 1.4.9, torch 1.11.0 (could work with latest versions)

## Description
1. The main code for TTA-Nav is located at [this folder](https://github.com/maytusp/tta-nav/tree/main/habitat-lab/habitat-baselines/habitat_baselines/rl/ppo/ae)
2. TTA-Nav method requires minor edit of the main habitat-baseline code: modify [habitat_evaluator.py](https://github.com/maytusp/tta-nav/blob/main/habitat-lab/habitat-baselines/habitat_baselines/rl/ppo/habitat_evaluator.py).
because it adapts during the test time.


## Train Top-down Deocder
First of all, we have to pre-pare an image dataset by running habitat evaluation with [pre-defined episodes](https://github.com/maytusp/tta-nav/tree/main/habitat-lab/data/datasets/gibson_collect_obs/v1/val).
Before running this, we have to download Gibson Scene dataset from the official website. Please see [Datasets](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md) for the guideline.
Our work uses v1 dataset.

Then, we can run the train and test scripts.

### Training Script: `train_ae.sh`

```
bash scripts/train_ae.sh
```

### Test Script: `test_ae.sh`

```
bash scripts/test_ae.sh
```

## Run TTA-Nav Agent
Our method is test on Habitat-lab. Please check the official website for installation.
The code below is to run TTA-Nav agent under Lighting corruption. Other corruptions can be run with the similar command but changing .yaml files.
```
python -u -m habitat_baselines.run habitat_baselines.evaluate=True --config-name pointnav/tta-nav/adapt/lighting_5.yaml
```

We only have to modify [habitat_evaluator.py](https://github.com/maytusp/tta-nav/blob/main/habitat-lab/habitat-baselines/habitat_baselines/rl/ppo/habitat_evaluator.py) and add ae folder to the ppo folder.

## Checkpoints
1. The main agent: [DD-PPO (BatchNorm version)](https://drive.google.com/file/d/10CBT_N6vXyw_g8I0uG5pIRvlb9UgDFwy/view?usp=share_link)
2. Top-down Decoder: [Autoencoder](https://drive.google.com/file/d/1XHr-5BvnbC4TXOK-yPzFgdKoVe9oGdmG/view?usp=share_link)

## Checkpoints
Any enquiry can be made via maytusp [at] gmail [dot] com
