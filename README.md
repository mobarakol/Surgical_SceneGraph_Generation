# Surgical_Graph

This repository contains the code implementation of the MICCAI 2020 paper [Learning and Reasoning with the Graph Structure Representation in Robotic Surgery](https://arxiv.org/pdf/2007.03357.pdf)<br>


The base interaction model "Graph Parsing Neural Networks" is adopted from [repository](https://github.com/SiyuanQi/gpnn) and integrated attention to improve the performance. <br>

To improve the model calibration, we integrated label smoothing by following this [repository](https://github.com/seominseok0429/label-smoothing-visualization-pytorch) 

# Dataset

## Training Data
Will be releasing soon!! 

## Evaluation Data
Download from **[[`Dataset Link`](https://drive.google.com/file/d/1OwWfgBZE0W5grXVaQN63VUUaTvufEmW0/view?usp=sharing)]** and place it inside the repository root and unzip 

The features of the evaluation data is not from this paper. You can follow our [ICRA2022 paper](**[ [```arXiv```](<https://arxiv.org/abs/2201.11957>) ]** |**[ [```Paper```](<https://ieeexplore.ieee.org/document/9695281>) ]** ) to get information about feature extraction.


## Citation
The paper can be cited by using below bibtex.

```
@inproceedings{islam2020learning,
  title={Learning and reasoning with the graph structure representation in robotic surgery},
  author={Islam, Mobarakol and Seenivasan, Lalithkumar and Ming, Lim Chwee and Ren, Hongliang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={627--636},
  year={2020},
  organization={Springer}
}
```
