

# Synthetic QCEW Data Generator
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
pip install pandas numpy scipy statsmodels scikit-learn patsy formulaic pyyaml tqdm matplotlib seaborn
```
**Notes:**
* Standard Library Dependencies:
  * The following dependencies are included in Python and do not require separate installation: `re, random, time, os, sys, pathlib, multiprocessing`
* Operating System support
  * Some standard libraries such as `os` and `sys` have differing behaviors across operating systems. Our program has been tested to work on Unix-like systems (Linux/MacOS) and may not behave correctly on Windows.
  * *If you are running the program on Windows, I recommend using a virtual machine running some Linux distrubution or using The Windows Subsystem for Linux (WSL) https://learn.microsoft.com/en-us/windows/wsl/install*

## Directories Overview
Here is a brief visual overview of the project repository:
```
syntheticQCEW/
├── Datasets/               		# Contains pre-generated datasets
│   ├── combineQWIandCBP.csv    	# The output of preprocess_combine.py
│   └── FinalMicrodata/				# Full synthetic datasets generated using predetermined defaults
├── Studies/						# Contains jupyter-notebooks with studies and justifications
├── SyntheticDataGenerator/			# The main directory with code and required libraries used to generate the synthetic data.
│   ├── DataDiag/					# Contains intermediate datasets and diagnostic files
│   ├── NAICS6_Pyfunctions/			# Contains python helper libraries
│   ├── GUIcode/			# Work in-progress GUI creation code to walk a user through creating the config file
│   ├── generateMicrodata.py		# The main script
│   ├── main.py		              # The main script
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
	* With an API key, the `main.py` will automatically download the required County Business Patterns (CBP) and Quarterly Workforce Indicators (QWI) datasets and place them in the directories specified in `config.yaml`
	* The `main.py` will also automatically download the required Quarterly Census of Employment and Wages from the Bureau of Labor Statistics and place it in the directory specified in `config.yaml`
	* Download the CBP dataset imputed by Eckert Et al. https://doi.org/10.3886/E117464V1 and place the csv file(s) in the `ImputeCBP/` directory specified in `config.yaml (See options below)
	
  Download Imputed CBP through Data Lumos:
    1. Access the data through Data Lumos https://doi.org/10.3886/E117464V1
	  2. Select `Imputed-CBP-Files/efsy_cbp_2016.zip`
	  3. Extract the the archive using `7z`
	  4. Place the extracted csv file in the `ImputeCBP/` directory specified in `config.yaml`
  Download Imputed CBP through Eckert webpage:
	  1. Alternatively (or in addition to) download the relevant data from https://fpeckert.me/cbp/
	  2. Extract the the archive using `7z`
	  3. Locate the imputed data csv file in the archive `Final_Imputed/efsy_cbp_2016.csv`
	  4. Place the extracted csv file in the `ImputeCBP/` directory specified in `config.yaml

3. Download other data files
  
  1. Download a csv file with the first column as the 5 digit state and county FIPS code.
      * Such a file can be found at https://www.bls.gov/cew/classifications/areas/qcew-area-titles.htm
      * The location of this file and its name should be used as the input for 'FULL_FIPS_FILE' under 'generalConfig' in the configuration file.
      * This file is **ONLY** used to automate the download of the QCEW and is not needed otherwise.
  2. Download a csv file which has the 6-digit NAICS as the second column and includes "naics_sector", "domain", and "super_sector". 
      * Such a file can be found at https://www.bls.gov/cew/classifications/industry/industry-supersectors.htm but may require some preprocessing.
      * The location of this file and its name should be used as the input for 'BLS_NAICS_CROSSWALK' under 'generalConfig' in the configuration file.
      * This file allows the use of BLS classifications such as super-sector and domain in the imputation models and includes these classifications in the final file output.
  3. Download a csv where the second column is of all 2-, 3-, 4-, 5-, and 6-digit NAICS codes including the NAICS-2 codes which are grouped, such as 31-33.
      * Such a file can be found at https://www.naics.com/search/ under 'Historical NAICS Reference Files'
      * The location of this file and its name should be used as the input for 'NAICS_FILE' under 'generalConfig' in the configuration file.
      * This file is primarily used to deal with the grouped NAICS-2 codes that appear with dashes. If it is not provided, the 2012 versions will be used as default.
      * Make sure to get the historical reference file that corresponds to a year that is the same as your generalConfig "YEAR" input or is before that year.
  4. Download a csv with the first column as the 2-digit FIPS state code, the second column as the state name, and the third as the state abbreviation. 
      * The first two columns can be downloaded from https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt. The state abbreviations can be added manually to the file.
      * The location of this file and its name should be used as the input for 'FIPS_STATE_FILE' under 'generalConfig' in the configuration file.
      * Throughout the code, the default is set to be the 2010 state fips codes. 


	
### Changing parameters and model formulas
Change parameters and models defined in `config.yaml` to suit your needs
### Generating Synthetic QCEW data:
1. Now you have everything configured and prepared to run the main pipeline. To generate the dataset, simply run `main.py`
	* This may take a few hours to run
	* Several checkpoint files are saved based on config inputs such as 'COMBINED_DATA' and 'NAICS6_FILE' under 'generalConfig'. 
	* If the run is interrupted, the `main.py` will use the checkpoint files instead of starting over. If there is already a file in the location you specified in the configuration file. `main.py` will **NOT** override it.


## Pre-generated Data
For those that just want to use the synthetic dataset without specifying any parameters, you can find a pre-generated dataset which uses the default values specified in `config.yaml` at `syntheticQCEW/Datasets/FinalMicroda.zip` which can be extracted using `7z`

## Studies and Justifications
See the `Studies/` directory for Jupyter Notebooks on:
* Default Model/parameter selection justifications
* Comparison to true QCEW data
## Alternative Method for downloading Census datasets
If for some reason you are unable to obtain an API key from census.gov, you may follow these steps to download the CBP and QWI datasets manually

**Downloading Quarterly Workforce Indicators (QWI):**
1. Navigate to https://ledextract.ces.census.gov/qwi/all
2. Make the following selections:
	1. Geography:
		* Geography Level: Choose some state/territory
			* Geography Type: Counties
				* Counties: Select All
	2. Firm Characteristics:
		* Industry Detail: NAICS 4-digit Industries
			* NAICS 4-digit Industries: Select All
		* Other Firm Characteristics: No Firm Age/Size Detail
		* Firm Ownership: All Private Ownership
	3. Worker Characteristics: No Worker Characteristics Detail
	4. Indicators: 
		* Employment: Emp, EmpEnd, EmpS
		* Earnings: EarnBeg, EarnHirAS
	5. Quarters: Select 2016 Q1
	6. Select another state/territory under Geography and select all counties
	7. Repeat for all states
		* ***Note:*** *It is best to split the the states into some number of groups to keep download sizes low.*
	8. Summary and Export: Submit Request, then download the csv.
	9. Save these files into the `QWIdata` directory specified in `config.yaml` as `qwi_co#.csv` where `#` indexes your grouping of states.

**Downloading County Business Patterns (CBP):**
1. Navigate to https://www.census.gov/data/datasets/2016/econ/cbp/2016-cbp.html
2. Select County File
3. Extract the archive and save as `cbp16co.txt` in the `CBPdataRaw` directory specified in `config.yaml`   



