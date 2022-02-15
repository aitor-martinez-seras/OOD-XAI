# OOD_XAI
This GihHub repository contains the code of the paper _____.

## Requirements
To install all the libraries needed, use `` pip install -r requirements.txt ``  

In this project, the tensorflow-gpu 2.7.0 version has been used. Please be aware that in order to use your GPU, you will need the following drivers and libraries installed on your computer, with the specific versions showed. [This link of the tensorflow library](https://www.tensorflow.org/install/gpu) has all the details, and below are the versions of the libraries with the links to the installation guides:
- [CUDA Toolkit 11.2](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
- [cuDNN 8.1.0](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)

## Usage 
There are two options for running the tests on the detector, either using the Jupyter Notebook in Google Colab or running it locally with the main.py.

### Google Colab
This is the most interactive and guided options, as you will run the algorithm step by step.

The instructions are included within the notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aitor-martinez-seras/OOD_XAI/blob/main/jupyter_notebook/OOD_Explainable_AI.ipynb)

### Run locally
The program is executed by running the main.py file, using the following parameters:

- **run_all**: If used, this argument will make the program ignore all other arguments and run all the tests included in the paper.
- **ind**: the In-Distribution datasets that will be tested.
- **ood**: the Out-of-Distribution datasets that will be tested.
- **model_arch**: the architectures that will be tested.
- **load_or_train**: determines whether you want to use the pretrained weights or to train the model first.
- **agg_function**: determines the way to compute the aggregation of the heatmaps of each cluster.
- **ood_function**: determines whether to use the expression (1) or expression (2) of the paper for the OoD score function.
- **seed**: the seed that determines the random sample taken from the training instances to create the clusters. 
If not included, the default of the paper is used.
- **n_heatmaps**: the size of the random subsample of training instances per class whose heatmaps 
are going to form the clusters.