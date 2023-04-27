# Datasets Preparation

## [DAPS-1](https://drive.google.com/file/d/1mB8uWu4lLU-y5GZlecgK8sgmbWYAM65U/view?usp=sharing)
The natural approach to overcome the domain gap is to capture or synthesize realistic data from a similar domain. This idea was implemented in this semi-synthetic dataset, which was generated based on the SemanticKITTI. Its main advantage is that the process can be reproduced for any setup with only a 3D model of the robot and LiDAR parameters. 

The pipeline consists of $5$ steps:
- First, the data was converted to the 3D semantic mesh using [Kimera-Semantic](https://github.com/MIT-SPARK/Kimera-Semantics) approach. 
- Then the layers of the map are separated in the open-source program [Mesh-Lab](https://www.meshlab.net/). 
- It is important, that [Kimera-Semantic](https://github.com/MIT-SPARK/Kimera-Semantics) works only with static objects, that is why we manually added some human models, that were generated in the [MakeHuman](http://www.makehumancommunity.org/) program, on the track. 
- The next step is to simulate in [Gazebo](https://gazebosim.org/home) with the semantic layers and desired LiDAR configuration with the open-source plugin. 
- Finally, we reproduce the original locations of the robot and capture the data to the rosbag archives.

## [DAPS-2](https://drive.google.com/file/d/1kwe6UZJzrLBOfua4BZGQcizY63XKVOEp/view?usp=sharing)
This dataset was recorded during a real field trip of the cleaning robot to the territory of the VDNH Park in Moscow in the summer of 2021. The robot model contained a configuration of 3 LiDARrs (central and 2 side ones). DAPS-2 contains several robot scenes in different parts of the park with different pedestrian fillings, with all points from the main central LiDAR.

## [SemanticKITTI](http://www.semantic-kitti.org/index.html)
It is one of the most popular large-scale datasets for segmentation methods evaluation. This dataset provides dense annotations for each individual scan of 11 sequences, which enables the usage of multiple sequential scans for semantic scene interpretation, like semantic segmentation and semantic scene completion.

## [RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D)
Is is a  relevant off-road dataset that consists of fully-annotated LiDAR scans from off-road environment with a lot of scenes containing lawns, dirt roads, various vegetation. At the same time, this dataset contains a relatively small number of labeled people and vehicles.