import requests
import pandas as pd
import numpy as np
import time
import os
import yaml
with open('config.yaml','r') as configFile:
    config = yaml.safe_load(configFile)
    preprocessConfig = config['preprocessConfig']
    generalConfig = config['generalConfig']
fipscodes_df = pd.read_csv("./DataDiag/FIPSstatecodename.txt")
api_key = generalConfig['API_KEY'] 

'''

The purpose of this file is to automate the download process of the raw Quarterly Worforce Indicators (QWI)
and County Business Patterns (CBP) datasets. This is done via the Census Bureau API.

You can request a key at https://api.census.gov/data/key_signup.html

You will also need to manually download the Imputed CBP file. The process is described below.

Alternatively, you can download the files manually from the https://census.gov website.
------------------ QWI FILES ------------------

QWI files from the U.S. Census website should be retrieved using 
the LED Extraction Tool (https://ledextract.ces.census.gov/qwi/all).
****************
To Retrieve:
	Get County-Level by 4-digit NAICS aggregate values
		a. Get County Repeat steps b-f with different geography selections.
			i) Select Geography Level=[some state, DC, or Puerto Rico], 
				then select all Counties
			ii) Select another state/territory and select all Counties.
					(I split the 52 state/territories into 
					   4 subsets to download separately)
			iii) Repeat steps 1a to 1f for each group of states
Save these files in folder "DataDiag/DataIn/QWIdata" with county-level files 
named qwi_co#.csv and state-level file as qwi_states.csv
------------------ Raw CBP --------------------

From https://www.census.gov/data/datasets/2016/econ/cbp/2016-cbp.html, 
Download 'Complete County File' and save as "cbp16co.txt" in "DataDiag/DataIn/CBPdataRAW/" folder
------------------ Imputed CBP --------------------

Download the imputed data from from https://doi.org/10.3886/E117464V1 and save as 
	"efsy_cbp_2016.csv" in "DataIn/Impute/" folder. 

Imputed data created by Eckert et al. The code for the imputation can be found on GitHub (https://github.com/fpeckert/cbp_database_public/tree/master) and the corresponding paper can be found (https://www.nber.org/system/files/working_papers/w26632/w26632.pdf).

Eckert, Fabian, Fort, Teresa C., Schott, Peter K., and Yang, Natalie J. County Business Patterns Database. Ann Arbor, MI: Inter-university Consortium for Political and Social Research [distributor], 2020-01-31. https://doi.org/10.3886/E117464V1 

'''
def format_naics(naics_code):
    if naics_code == "00":
        return "------"
    elif (len(naics_code) == 5 and naics_code[2] == '-'):
        return naics_code[:2].ljust(6,'-')
    elif len(naics_code) == 2:
        return naics_code.ljust(6, '-')
    elif len(naics_code) == 3:
        return naics_code.ljust(6, '/')
    elif len(naics_code) == 4:
        return naics_code.ljust(6, '/')
    elif len(naics_code) == 5:
        return naics_code.ljust(6, '/')
    else:
        return naics_code


########## QWI Download ##########
url = "https://api.census.gov/data/timeseries/qwi/sa"
num_split = 4
state_groups = np.array_split(fipscodes_df, num_split)
print("Downloading QWI county data from https://api.census.gov/data/timeseries/qwi/sa")
print("Alternatively, you can visit https://ledextract.ces.census.gov/qwi/all and follow the instructions in the documentation.")

for i, group in enumerate(state_groups, start=1):
    fulldf_pergroup = []
    for _, row in group.iterrows():
        fips_code = f"{int(row['FIPScode']):02d}"
        params = {
            "get": "Emp,EmpEnd,EmpS,EarnBeg,EarnHirAS,sEmp,sEmpEnd,sEmpS,sEarnBeg,sEarnHirAS,geography,ind_level,geo_level",
            "for": "county:*",
            "in": f"state:{fips_code}",
            "year": "2016",
            "quarter": "1",
            "industry": "",
            "key": api_key
        }
        
        max_retries = 3
        for retry in range(max_retries):
            response = requests.get(url, params=params)
            if response.ok:
                fulldf_pergroup.extend(response.json()[1:])
                print(f"Successfully fetched data for {row['name']} (FIPS: {fips_code})")
                break
            else:
                print(f"Attempt {retry + 1} failed for {row['name']} (FIPS: {fips_code}) - Status: {response.status_code}")
                time.sleep(2) # Wait. Sometimes you hit rate limits and need to wait a second before making more API calls
        
        if not response.ok:
            print(f"Failed after {max_retries} retries for {row['name']} (FIPS: {fips_code})")
    
    if fulldf_pergroup:
        directory = f"{preprocessConfig['DATA_IN_FOLDER']}{preprocessConfig['QWIDATA']}"
        filename = f"{directory}qwi_co{i}.csv"
        os.makedirs(directory, exist_ok=True)
        df = pd.DataFrame(fulldf_pergroup, columns=response.json()[0])
        df.to_csv(filename, index=False)
        print(f"Saved part {i} with {len(group)} states to {filename}")
    else:
        print(f"No data was fetched for part {i}")

########## CBP Download ##########
url = "https://api.census.gov/data/2016/cbp"
print("Downloading CBP county data from https://api.census.gov/data/2016/cbp")
print("Alternatively, you can visit https://www.census.gov/data/datasets/2016/econ/cbp/2016-cbp.html and selecting 'county file'")
params = {
    'get': 'ESTAB,PAYQTR1,PAYQTR1_N_F', 
    'for': 'county:*',                    
    'in': 'state:*',
    'NAICS2012': '*',  
    'key': api_key                    
}
response = requests.get(url, params=params)
if response.ok:#status_code == 200:
    data = response.json()
    headers = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)
    df.rename(columns={"county":"fipscty",
                       "state":"fipstate",
                        "NAICS2012":"naics",
                        "ESTAB":"est",
                        "PAYQTR1":"qp1",
                        "PAYQTR1_N_F":"qp1_nf"},inplace=True)
    df['naics']=df['naics'].astype(str)
    df['naics'] = df['naics'].apply(format_naics)
    directory = f"{preprocessConfig['DATA_IN_FOLDER']}{preprocessConfig['CBPDIR']}"
    filename= f"{preprocessConfig['DATA_IN_FOLDER']}{preprocessConfig['CBPDATA']}"
    os.makedirs(directory, exist_ok=True)
    df.to_csv(filename, index=False)
    print("Data successfully downloaded and saved to ./DataDiag/DataIn/CBPdataRAW/cbp16co.txt")
    print("\nSample data:")
    print(df.head())
else:
    print(f"Error: {response.status_code}")
    print(response.text)
