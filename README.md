# Form-Context Model

This repository contains the code and supplementary material for the paper [Learning Semantic Representations for Novel Words: Leveraging Both Form and Context](https://arxiv.org/abs/1811.03866). 
The contents of each subdirectory are as follows:

### crw-dev

This directory contains the CRW development dataset. For more info, refer to the original paper.

### fcm

This directory contains the actual form-context model. Information on how to use it can be found in the README file of this directory.

**Important**: The code found in this directory is a beautified, more readable version of the original form-context model. Due to random parameter initialization, results may slightly deviate from the ones reported in the paper. If you want to *use* the form-context model, this is the right version for you. If, instead, you want to *reprocude* the original results, please contact me via `timo.schick<at>sulzer.de` to obtain the original files.  

### preprocessing

This directory contains a preprocessing script that is required for training the form-context model. Information on how to use it can be found in the README file of this directory.

## Citation

If you make use of the CRW development set or the code in this repository, please cite the following paper:
```
@inproceedings{schick2018learning,
  title={Learning Semantic Representations for Novel Words: Leveraging Both Form and Context},
  author={Schick, Timo and Sch{\"u}tze, Hinrich},
  url="https://arxiv.org/abs/1811.03866",
  booktitle={Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence},
  year={2019}
}
```