# Datasets

### [DAPS-1](https://drive.google.com/file/d/1mB8uWu4lLU-y5GZlecgK8sgmbWYAM65U/view?usp=sharing)
The natural approach to overcome the domain gap is to capture or synthesize realistic data from a similar domain. This idea was implemented in this semi-synthetic dataset, which was generated based on the SemanticKITTI. Its main advantage is that the process can be reproduced for any setup with only a 3D model of the robot and LiDAR parameters. 

The pipeline consists of $5$ steps:
- Convert data to the 3D semantic mesh using [Kimera-Semantic](https://github.com/MIT-SPARK/Kimera-Semantics) approach. 
- Separate the layers of the map in the open-source program [Mesh-Lab](https://www.meshlab.net/). 
- Manually add some human models, that were generated in the [MakeHuman](http://www.makehumancommunity.org/) program, on the track. It is important, because [Kimera-Semantic](https://github.com/MIT-SPARK/Kimera-Semantics) works only with static objects. 
- Simulate in [Gazebo](https://gazebosim.org/home) with the semantic layers and desired LiDAR configuration with the open-source plugin. 
- Reproduce the original locations of the robot and capture the data to the rosbag archives.

DAPS-1 contains 11 sequences with more than 23 000 labeled LiDAR scans in total.

**Full DAPS-1 Dataset** [(Download 9 GB)](https://drive.google.com/file/d/1mB8uWu4lLU-y5GZlecgK8sgmbWYAM65U/view?usp=sharing)

### [DAPS-2](https://drive.google.com/file/d/1kwe6UZJzrLBOfua4BZGQcizY63XKVOEp/view?usp=sharing)
This dataset was recorded during a real field trip of the cleaning robot to the territory of the VDNH Park in Moscow in the summer of 2021. The robot model contained a configuration of 3 LiDARrs (central and 2 side ones). DAPS-2 contains several robot scenes in different parts of the park with different pedestrian fillings, with all points from the main central LiDAR.

DAPS-2 contains 3 sequences with 109 labeled LiDAR scans in total.

**Full DAPS-2 Dataset** [(Download 18 MB)](https://drive.google.com/file/d/1kwe6UZJzrLBOfua4BZGQcizY63XKVOEp/view?usp=sharing)

### [SemanticKITTI](http://www.semantic-kitti.org/index.html)
It is one of the most popular large-scale datasets for segmentation methods evaluation. This dataset provides dense annotations for each individual scan of 11 sequences, which enables the usage of multiple sequential scans for semantic scene interpretation, like semantic segmentation and semantic scene completion.

### [RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D)
Is is a  relevant off-road dataset that consists of fully-annotated LiDAR scans from off-road environment with a lot of scenes containing lawns, dirt roads, various vegetation. At the same time, this dataset contains a relatively small number of labeled people and vehicles.

## Datasets Statistics
We provide statistics on the proportion of points belonging to the five joined categories among the total number of points in LiDAR scans for datasets listed above.

| Dataset | Resolution | Scans / Maps | LiDAR | Vehicle | Human | Surface | Static Obstacle | Unlabeled |
|   :---:| :---:   |  :---: |    :---:   |    :---:   |    :---:   |    :---:   |    :---:   |    :---:   |
| [DAPS-1](#daps-1) | 1024x64 | 23,056 | 64-lines Ouster OS0 | 3.89% | 0.46% | 70.70% | 24.89% | 0.03% |
| [DAPS-2](#daps-2) | 1024x64 | 109 | 64-lines Ouster OS0 | 0.01% | 2.00% | 12.00% | 11.00% | 74.99% |
| [SemanticKITTI](#semantickitti) | 2048x64 | 19,130 | Velodyne HDL-64E | 0.01% | 0.01% | 43.91% | 47.54% | 8.52% | 
| [RELLIS-3D](#rellis-3d) | 1024x32 | 13,556 | 64-lines Ouster OS1 | 0.02% | 2.28% | 23.41% | 32.64% | 41.70% |


## Datasets Preparation

### Folder Structure
Our `Dataset` folder contains all datasets used for training and evaluation of our models. 

Expected structure for `Dataset` folder:
```bash
Dataset/ 
├── daps-1/               
│   ├── data/             
│   │   ├── 00            
│   │   │   ├── clouds/ <-- xxx.bin 
│   │   │   └── labels/ <-- xxx.label 
│   │   └── 08
│   │       ├── clouds/ <-- xxx.bin  
│   │       └── labels/ <-- xxx.label
│   └── params.yml      <-- parameters of the lidar sensor
├── daps-2/
├── kitti/
├── rellis/
├── <other datasets>
```

### Configuration Files

There are four dataset configuration files inside [`./cfgs`](./cfgs/):
```bash
cfgs/
├── navigation.yml      <-- shows datasets for train/val
├── dataset-params.yml  <-- parameters of the dataset lidar sensors
├── data_stats.yml      <-- point cloud statistics 
├── label_params.yml    <-- category mapping info
```

- Select datasets and sequences for training and validation inside [`navigation.yml`](./cfgs/navigation.yml).

- Add information about lidar sensors for choosen datasets inside [`dataset-params.yml`](./cfgs/dataset_params.yml).

- `SalsaNet` models require data stats like _Expected Value_ and _Standard Deviation_ for each of the channels of the input scan: `(x, y, z, range)`. You can calculate them by running [`dataset_stat.py`](./dataset_utils/dataset_stat.py) script and put them inside [`data_stats.yml`](./cfgs/data_stats.yml).

```bash
python dataset_stat.py /Dataset/ ../cfgs/
```
- [`label_params.yml`](./cfgs/label_params.yml) config contains category description and mapping for training. Before training update `content` with the results calculated by [`dataset_stat.py`](./dataset_utils/dataset_stat.py) based on selected datasets. This is needed for category weights calculation.