# AdaBoost.SDM: Similarity and Dissimilarity-based Manifold Regularized Adaptive Boosting Algorithm

This repository is the official implementation of [AdaBoost.SDM: Similarity and Dissimilarity-based Manifold Regularized Adaptive Boosting Algorithm].

## Requirements

This code has been developed under `Python 3.9.16`, `scikit-learn 1.3.0`, `Intel(R) Xeon(R) W-2145, (3.7 GHz)`, `64 GB of RAM`, `Nvidia TITAN RTX GPU (RAM = 24 GB)`, and `CUDA 11.1`, on `Windows 10`.

To install requirements:

```setup
pip install -r requirements.txt
```

## Usage

To reproduce the results presented in the paper, run this command:

```
python main.py --data climate --model AdaBoostSDM
```

## Data

Datasets are colleclted from OpenML(https://openml.org/) dataset repositories.
