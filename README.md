
# syntheticQCEW
The purpose of this project is to give users the tools needed to generate synthetic data with rows for each establishment and columns for monthly employment, quarterly wages, industry codes, and geographic codes to match the structure of the Quarterly Census on Employment and Wages (QCEW). Users can specify parameters and generating models in `config.yaml` to generate data matching their needs.


## Setup
To get started with the repository run the following commands to clone the project locally:
```bash
git clone https://github.com/krd5520/syntheticQCEW.git
cd syntheticQCEW
```
### Dependencies
This project requires the following Python packages, which can be installed via pip using the command:
```bash
pip install pandas numpy scipy statsmodels scikit-learn patsy formulaic pyyaml tqdm
```
**Notes:**
* Standard Library Dependencies:
  * The following dependencies are included in Python and do not require separate installation: `re, random, time, os, sys, pathlib, multiprocessing`
* Operating System support
  * Some standard libraries such as `os` and `sys` have differing behaviors across operating systems. Our program has been tested to work on Unix-like systems (Linux/MacOS) and may not behave correctly on Windows.
  * *If you are running the program on Windows, I recommend using a virtual machine running some Linux distrobution or using The Windows Subsystem for Linux (WSL) https://learn.microsoft.com/en-us/windows/wsl/install*

## Directories Overview
Here is a brief visual overview of the project repository:
```
syntheticQCEW/
├── Datasets/               		# Contains pre-generated datasets
│   ├── combineQWIandCBP.csv    	# The output of preprocess_combine.py
│   └── FinalMicrodata/				# Full synthetic datasets generated using predetermined defaults
├── Studies/						# Contains jupyter-notebooks with studies and justifications
├── SyntheticDataGenerator/			# The main directory with code and required libraries used to generate the synthetic data.
│   ├── CBP_QWI_download.py
│   ├── preprocess_combine.py
│   ├── DataDiag/					# Contains intermediate datasets and diagnostic files
│   ├── NAICS6_Pyfunctions/			# Contains python helper libraries
│   ├── generateMicrodata.py		# The main script
│   └── config.yaml					# Contains configurable parameters and model selections
└── README.md
```

## Usage
To use the synthetic data generator follow these steps:
### Initial Setup
1. Request an API key from: https://api.census.gov/data/key_signup.html
	* Add this to `config.yaml` under `generalConfig`
	```
	generalConfig:
		API_KEY: < "Place API key here" >
		YEAR: 2016
		QTR: 1
	```
2. Download all required Census datasets:
**Note:** If you have trouble obtaining an API key, you can check the section `#Alternative Method for downloading datasets` for alternative download steps.
	* Run the python script `CBP_QWI_download.py`	which will automatically download the required County Business Patterns (CBP) and Quarterly Workforce Indicators (QWI) datasets and place them in the directories specified in `config.yaml`
	* Download the CBP dataset imputed by Eckert Et al. https://doi.org/10.3886/E117464V1
		1. Select `Imputed-CBP-Files/efsy_cbp_2016.zip`
		2. Extract the the archive using `7z`
		4. Place the extracted csv file in the `ImputeCBP/` directory specified in `config.yaml`
### Changing parameters and model formulas
Change parameters and models defined in `config.yaml` to suit your needs
### Combining the Census Datasets
1. Run the python script `preprocess_combine.py` which combines all of the datasets gathered in the previous step, creates a new directory called `PythonPreprocessOut` in the location specified in `config.yaml` and saves `combineQWIandCBP.csv` in the new directory.
### Generating Synthetic QCEW data:
1. Now you have everything configured and prepared to run the main pipeline. To generate the dataset, simply run `generateMicrodata.py`
	* This may take a few hours to run

**Note:**  The script `preprocess_combine` can sometimes generate data that is not compatible with the wage imputation step of the pipeline. If you are getting errors at this step you can replace the `combineQWIandCBP.csv` file with the one provided in the `Datasets` directory.

## Pre-generated Data
For those that just want to use the synthetic dataset without specifying any parameters, you can find a pre-generated dataset which uses the default values specified in `config.yaml` at `syntheticQCEW/Datasets/FinalMicroda.zip` which can be extracted using `7z`

## Studies and Justifications
See the `Studies/` directory for:
* Model selection justifications
* sdffds
## Alternative Method for downloading Census datasets

