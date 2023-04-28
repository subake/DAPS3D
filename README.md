# DAPS3D
DAPS3D: Domain Adaptive Projective Segmentation of 3D LiDAR Point Clouds

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Abstract
LiDARs are one of the key sources of reliable environmental ranging information for autonomous vehicles. Segmentation of 3D scene elements (roads, buildings, people, cars, etc.) based on LiDAR point clouds has limitations. On the one hand, point- and voxel-based segmentation neural networks have insufficiently high speed. On the other hand, modern labeled datasets mainly contain street scenes recorded for driverless cars and few datasets for mobile delivery robots or cleaners that must work in parks and yards with heavy pedestrian traffic. This article aims to overcome these limitations.
We have proposed a novel approach called DAPS3D to training a deep neural networks for 3D semantic segmentation based on a spherical projection of a point cloud and LiDAR-specific masks, which keeps the model working when changing the type of LiDAR.
First of all, we proposed various high-speed multi-scale spherical projection segmentation models, including convolutional, recurrent, and transformer architectures.
Secondly, we proposed a number of original augmentations of spherical projections of LiDAR data, including FoV, flip and rotation augmentation, as well as a special T-Zone cutout, which provide an increase in the model invariance  when changing the data domain.
Finally, we introduce a new method to generate synthetic datasets for a domain adaptation problem. According this we have developed two new data sets for validating 3D scene outdoor segmentation algorithms: the DAPS-1 dataset based on the augmentation of the reconstructed 3D semantic map, the DAPS-2 LiDAR dataset, collected by the on-board sensors of the cleaning robot in the park area.
Particular attention is paid to the performance of the developed models, which demonstrates the possibility of their functioning in real time. 

![SalsaNetRec](images/SalsaNetRec.png)
![RecBlock](images/RecBlock.png)

## ToDo List

- [x] Release code
- [x] Document for Installation
- [x] Document for Dataset Preparation
- [x] Document for Testing and Training
- [ ] Evaluation
- [ ] Release trained models
- [x] Release datasets: DAPS-1 & DAPS-2

## Installation Instructions
- We use conda environments 
- We use different Python and PyTorch versions for our models
- For complete installation instructions, please see [Installation](INSTALL.md).

## Dataset Preparation
- We release our [DAPS-1](DATASET.md#daps-1) and [DAPS-2](DATASET.md#daps-2) datasets.
- We also experiment on SemanticKITTI and RELLIS-3D datasets.
- Please see [Datasets Preparation](DATASET.md#datasets-preparation) for complete instructions for preparing the datasets.

## Execution Instructions

### Training

- Please see [Getting Started](GETTING_STARTED.md) for training commands.

### Evaluation

- Please see [Getting Started](GETTING_STARTED.md) for evaluation commands.

## Results
You can find our pretrained models in [Getting Started](GETTING_STARTED.md).

## Citation
If you found DAPS3D useful in your research, please consider starring ⭐ us on GitHub and citing 📚 us in your research!

## Licence
This repository is released under MIT License (see LICENSE file for details).

