# syntheticQCEW
The purpose of this project is to give users the tools needed to generate synthetic data with rows for each establishment and columns for monthly employment, quarterly wages, industry codes, and geographic codes to match the structure of the Quarterly Census on Employment and Wages (QCEW). Users can specify parameters and generating models in `config.yaml` to generate data matching their needs.

## Setup
To get started with the repository
### Dependancies
This project requires the following Python packages, which can be installed via pip using the command:
```bash
pip install pandas numpy scipy statsmodels scikit-learn patsy formulaic pyyaml tqdm
```

**Notes:**
* Standard Library Dependencies:
  * The following dependencies are included in Python and do not require separate installation: `re, random, time, os, sys, pathlib, multiprocessing`
* Operating System support
  * Some standard libraries such as `os` and `sys` have differing behaviors across operating systems. Our program has been tested to work on Unix-like systems (Linux/MacOS) and may not behave correctly on Windows.
  * *If you are running the program on Windows, I reccomend using a virtual machine running some Linux distrobution or using The Windows Subsystem for Linux (WSL) from PowerShell https://learn.microsoft.com/en-us/windows/wsl/install*

### Installation
## Directories Overview
## Pre-generated Data
For those that just want to use the synthetic dataset without specifying any parameters, you can find a pre-generated dataset which uses the default values specified in `config.yaml` at `syntheticQCEW/Datasets/FinalMicroda.zip` which can be extracted using 7z using the command
```bash
7z x ./Datasets/FinalMicrodata.zip
```
## Usage
## Studies and Justifications

