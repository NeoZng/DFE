# DFE training on ETH3d and KITTI odom dataset

This is a submodule of Epipolar Evaluation.

To use the repo, download ETH3d and KITTI odom dataset and place them in the `dataset` folder.  

The repo has been tested on Ubuntu 22.04 with miniconda.

The environment.yml file has been updated to support Python3 (PyTorch does not support Python2 anymore).

---

# DISCONTINUATION OF PROJECT #  
This project will no longer be maintained by Intel.  
Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.  
Intel no longer accepts patches to this project.  
 If you have an ongoing need to use this project, are interested in independently developing it, or would like to maintain patches for the open source software community, please create your own fork of this project.  
  
# Deep Fundamental Matrix Estimation

This repository contains code for training a network as described in our [paper](http://vladlen.info/papers/deep-fundamental.pdf):

>Deep Fundamental Matrix Estimation  
René Ranftl and Vladlen Koltun  
European Conference on Computer Vision (ECCV), 2018

The trained network can be used to compute a fundamental matrix based on noisy point correspondences.

## Setup


Create and activate conda environment:

```shell
conda env create -f environment.yml
conda activate dfe
```

## Train

Training can be performed with the following command:

```shell
python train.py --dataset [dataset folders]
```

Dataset folders are assumed to be in COLMAP format. Multiple folders can be listed by separating them with a whitespace.

An example dataset can be found here: [Family](https://drive.google.com/open?id=1b4lb5La3dzn_D87sy-fpgCAbEnGRSLrL).

To get help:

```shell
python train.py -h
```


## Test

```shell
python test.py --dataset [dataset folders] --model [path to model]
```

A pre-trained model that was trained on Tanks and Temples can be found in the models folder.


## Creating your own dataset

The dataloader assumes that a dataset folder contains the following files:

```shell
reconstruction.db 
sparse/0/cameras.bin
sparse/0/images.bin

```

`cameras.bin` and `images.bin` are SfM reconstruction in COLMAP format. 

`reconstruction.db` contains descriptors, matches, etc.

These files are automatically created when performing reconstruction using COLMAP. We found that better results are achieved with less aggressive filtering for feature matching than the default COLMAP settings. The recommended way to create a training set is:

1) Run COLMAP to produce `cameras.bin` and `images.bin` and initial `reconstruction.db`
2) Run `get_features.sh` to produce a training set with higher outlier ratios to replace `reconstruction.db`

## Citation

Please cite our paper if you use this code in your research:

```bibtex
@InProceedings{Ranftl2018,
    author = {Ranftl, Rene and Koltun, Vladlen},
    title = {Deep Fundamental Matrix Estimation},
    booktitle = {The European Conference on Computer Vision (ECCV)},
    year = {2018}
}
```

## License

MIT License
