# Surgical_Graph

This repository contains the code implementation of the MICCAI 2020 paper [Learning and Reasoning with the Graph Structure Representation in Robotic Surgery](https://arxiv.org/pdf/2007.03357.pdf)<br>


The base interaction model "Graph Parsing Neural Networks" is adopted from [repository](https://github.com/SiyuanQi/gpnn) and integrated attention to improve the performance. <br>

To improve the model calibration, we integrated label smoothing by following this [repository](https://github.com/seominseok0429/label-smoothing-visualization-pytorch) 

# Dataset

Tool-tissue interaction graph annotation on [2018 Robotic Scene Segmentation Challenge dataset](https://arxiv.org/abs/2001.11190) can be downloaded from [google drive](https://drive.google.com/file/d/16G_Pf4E9KjVq7j_7BfBKHg0NyQQ0oTxP/view). The xml contains both bounding box annotation and intraction class annotation.
Please note that video sequences of 1, 5 and 16 are used for validation and remaining sequences for training.

An interactive Colab notebook can be found on how to read the annotation of bounding box and tool-tissue interaction class [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/mobarakol/Surgical_SceneGraph_Generation/blob/main/reading_dataset.ipynb)

For more information you can check our also. [ICRA+RA-L2022 paper](**[ [```arXiv```](<https://arxiv.org/abs/2201.11957>) ]** |**[ [```Paper```](<https://ieeexplore.ieee.org/document/9695281>) ]** ).


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
