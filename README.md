# AdaBoost.SDM: Similarity and Dissimilarity-based Manifold Regularized Adaptive Boosting Algorithm

This repository is the official implementation of [AdaBoost.SDM: Similarity and Dissimilarity-based Manifold Regularized Adaptive Boosting Algorithm](https://doi.org/10.1016/j.patrec.2025.05.016).

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

## Citation

```
@article{MUKHAMEDIYA202566,
  author = {Azamat Mukhamediya and Amin Zollanvari},
  title = {AdaBoost.SDM: Similarity and dissimilarity-based manifold regularized adaptive boosting algorithm},
  journal = {Pattern Recognition Letters},
  volume = {196},
  pages = {66-71},
  year = {2025},
  issn = {0167-8655},
  doi = {https://doi.org/10.1016/j.patrec.2025.05.016},
  url = {https://www.sciencedirect.com/science/article/pii/S0167865525002090},
}
```
