# Surgical_Graph

This repository contains the code implementation of the MICCAI 2020 paper [Learning and Reasoning with the Graph Structure Representation in Robotic Surgery](https://arxiv.org/pdf/2007.03357.pdf)<br>


The base interaction model "Graph Parsing Neural Networks" is adopted from [repository](https://github.com/SiyuanQi/gpnn) and integrated attention to improve the performance. <br>

To improve the model calibration, we integrated label smoothing by following this [repository](https://github.com/seominseok0429/label-smoothing-visualization-pytorch) 


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