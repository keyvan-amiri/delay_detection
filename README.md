# Mind the Long Tail: Understanding the Difficulty of Delay Detection in Business Processes

This is the supplementary githob repository of the paper: "Mind the Long Tail: Understanding the Difficulty of Delay Detection in Business Processes", submitted to BPM 2026.

## Supplementary Report
The supplementary report of the paper is accessible [here](https://github.com/keyvan-amiri/delay_detection/blob/main/Supplementary_Report.pdf)  

### Installation
Clone this GitHub repository to your local machine. To install and set up the required environment on a Linux system, run the following commands:

```bash
conda create -n delay python=3.11
conda activate delay
pip install -r imbalanced_regression.txt
conda clean --all
```

## Running Experiments
To execute the pipeline for a dataset (e.g., BPIC20PTC) and a imbalanced regression technique (e.g., BMSE) run the following:

```
python main.py --dataset BPIC20PTC --IR BMSE
```
If no imbalanced regression technique is parsed (--IR) the Vanilla model is trained.

CSW (Cost Sensitive re-Weighting) and EAL (Error-Aware Loss) can be combined with Label Distribution Smooting (LDS) and/or Feature Distribution Smoothing (FDS). Therfore, the pipeline include running experiments with four different configurations (wos: without smooting, LDS, FDS, LDS+FDS). For more information please refer to  [Delving into Deep Imbalanced Regression](https://proceedings.mlr.press/v139/yang21m.html) and its corresponding [GitHub repository](https://github.com/YyzHarry/imbalanced-regression). 

Balanced MSE (BMSE) cannot be combined with LDS, but the authors suggested that FDS should be complementary to their technique. Therefore, the piepline includes experiments with two configurations (wos and FDS). For more information please refer to [Balanced MSE for Imbalanced Visual Regression](https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Balanced_MSE_for_Imbalanced_Visual_Regression_CVPR_2022_paper.html) and its its corresponding [GitHub repository](https://github.com/jiawei-ren/BalancedMSE). 

The pipeline includes the same two configurations (wos and FDS) for Squared Error Relevance Area (SERA). For more information please refer to [Model Optimization in Imbalanced Regression](https://link.springer.com/chapter/10.1007/978-3-031-18840-4_1) and [Imbalanced regression and extreme value prediction](https://link.springer.com/article/10.1007/s10994-020-05900-9). The original implementation of SERA is provided in R (in [this package](https://github.com/nunompmoniz/IRon/blob/master/R/phi.R)), and in our implementation is implemented in Python. 

To tain the uncertainty-aware approach based on survival analysis, --IR argument must be set to 'survival'. It is also possible to train a uncertainty-aware model based on quantile regression using 'quantile' for --IR argument.

* All event logs are collected [here](https://github.com/keyvan-amiri/delay_detection/tree/main/data).
* All congifurations that are used for hyper-parameter optimization and training are collected [here](https://github.com/keyvan-amiri/delay_detection/tree/main/cfg). You need to adjust cdg.data.path in the cfg file in order to determine the path to the XES or CSV file.

Once the survival model is trained, the second step for uncertainty-aware classification (and training the point-estimate deterministic baseline) for a dataset (e.g., BPIC20PTC) can be achived by running the followin:

```
python delay_analysis.py --dataset BPIC20PTC 
```
