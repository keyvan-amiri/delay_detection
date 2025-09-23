# Imbalanced Regression for Remaining Time Prediction in Business Processes

## Installation
To install all packages please use  [imbalanced_regression.txt](https://github.com/keyvan-amiri/delay_detection/blob/main/imbalanced_regression.txt) which includes all packages required for experiments with DALSTM and PGTNet models. If we decided to not include PGTNet, we can remove unnecessary packages later. 

## Running Experiments
To execute the pipeline for a dataset (e.g., BPIC20PTC), a model (e.g., DALSTM) and a imbalanced regression technique (e.g., BMSE) run the following:

```
python main.py --dataset BPIC20PTC --model DALSTM --IR BMSE
```
If no imbalanced regression technique is parsed (--IR) the Vanilla model is trained. CSW (Cost Sensitive re-Weighting) and EAL (Error-Aware Loss) can be combined with Label Distribution Smooting (LDS) and/or Feature Distribution Smoothing (FDS). Therfore, the pipeline include running experiments with four different configurations (wos: without smooting, LDS, FDS, LDS+FDS). For more information please refer to  [Delving into Deep Imbalanced Regression](https://proceedings.mlr.press/v139/yang21m.html) and its corresponding [GitHub repository](https://github.com/YyzHarry/imbalanced-regression). 
Balanced MSE (BMSE) cannot be combined with LDS, but the authors suggested that FDS should be complementary to their technique. Therefore, the piepline includes experiments with two configurations (wos and FDS). For more information please refer to [Balanced MSE for Imbalanced Visual Regression](https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Balanced_MSE_for_Imbalanced_Visual_Regression_CVPR_2022_paper.html) and its its corresponding [GitHub repository](https://github.com/jiawei-ren/BalancedMSE). The pipeline includes the same two configurations for Squared Error Relevance Area (SERA). For more information please refer to [Model Optimization in Imbalanced Regression](https://link.springer.com/chapter/10.1007/978-3-031-18840-4_1) and [Imbalanced regression and extreme value prediction](https://link.springer.com/article/10.1007/s10994-020-05900-9). The original implementation of SERA is provided in R (in [this package](https://github.com/nunompmoniz/IRon/blob/master/R/phi.R)), and in our implementation is implemented in Python. 
