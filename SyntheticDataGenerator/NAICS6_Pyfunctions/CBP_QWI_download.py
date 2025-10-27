import requests
import pandas as pd
import numpy as np
import time
import os
import yaml
pd.set_option("display.max_columns", None)


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
def format_naics(naics_code): #for CBP download
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

#subset fipscodes_df to only requested states and process savechunks into num_splits
def state_selector(fipscodes_df,states,savechunks):
    if savechunks is None:
        savechunks=1
    if states is not None and len(states)>0: #if states are inputted
        if isinstance(states,str) and not isinstance(states,list): #if STATES is a string
            states=list(states.upper()) #standardize format
            if states=="ALL": #if ALL, do not subset
                return fipscodes_df,savechunks
        try: #if iterable, standardize format
            states=[str(x).upper() for x in states]
        except:
            print('STATES must be iterable object. It was '+str(type(states)))
        if "ALL" in states: #if ALL do not subset
            return fipscodes_df, savechunks
        fipscode = fipscodes_df.iloc[:, 0].astype(str).str.upper() #standardize format col1
        stateselect=(fipscode.isin(states)) #initialize the logical to select states
        if len(fipscodes_df.columns)>1: #if there are more than 1 column
            for colname in fipscodes_df.columns: #iterate columns to create logical
                fipscode=fipscodes_df[colname].astype(str).str.upper()
                stateselect=(stateselect)|(fipscode.isin(states))
        if len(states)==1: #if only one state, then only 1 chunk to save
            savechunks=1
        fipscodes_df = fipscodes_df[stateselect] #subset
    return fipscodes_df, savechunks

########## QWI Download ##########
## internal function to iterate across splits if necessary
def sub_downloadQWI(i,group,url,generalConfig,preprocessConfig,max_retries=3):
    fulldf_pergroup = []
    for _, row in group.iterrows(): #for row in group
        fips_code = f"{int(row['FIPScode']):02d}" #fipscode
        paramsqwi = {
            "get": "Emp,EmpEnd,EmpS,EarnBeg,sEmp,sEmpEnd,sEmpS,sEarnBeg,geography,ind_level,geo_level",
            "for": "county:*",
            "in": f"state:{fips_code}",
            "year": generalConfig["YEAR"],
            "quarter": generalConfig["QTR"],
            "industry": "",
            "key": generalConfig["API_KEY"]
        }
        for retry in range(max_retries): #try a couple times in case fails out
            response = requests.get(url, params=paramsqwi)
            if response.ok:
                fulldf_pergroup.extend(response.json()[1:])
                print(f"Successfully fetched data for {row['name']} (FIPS: {fips_code})")
                break
            else:
                print(
                    f"Attempt {retry + 1} failed for {row['name']} (FIPS: {fips_code}) - Status: {response.status_code}")
                time.sleep(
                    2)  # Wait. Sometimes you hit rate limits and need to wait a second before making more API calls
        if not response.ok:
            print(f"Failed after {max_retries} retries for {row['name']} (FIPS: {fips_code})")

    if fulldf_pergroup is not None and len(fulldf_pergroup)>0:
        response = requests.get(url, params=paramsqwi)
        directory = f"{preprocessConfig['DATA_IN_FOLDER']}{preprocessConfig['QWIDIR']}"
        filename = f"{directory}qwi_co{i}.csv"
        os.makedirs(directory, exist_ok=True)
        df = pd.DataFrame(fulldf_pergroup, columns=response.json()[0])
        df.to_csv(filename, index=False)
        print(f"Saved part {i} with {len(group)} states to {filename}")
    else:
        df=None
        print(f"No data was fetched for part {i}")
    return(df)

## download QWI based on configs and a state fipscode dataframe
def download_QWI(fipscodes_df,preprocessConfig,generalConfig,savechunks=4,max_retries = 3):
    url = "https://api.census.gov/data/timeseries/qwi/sa"
    api_key = generalConfig['API_KEY']
    state_abbr = {
        "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca", "08": "co", "09": "ct", "10": "de",
        "11": "dc", "12": "fl", "13": "ga", "15": "hi", "16": "id", "17": "il", "18": "in", "19": "ia", "20": "ks",
        "21": "ky", "22": "la", "23": "me", "24": "md", "25": "ma", "26": "mi", "27": "mn", "28": "ms", "29": "mo",
        "30": "mt", "31": "ne", "32": "nv", "33": "nh", "34": "nj", "35": "nm", "36": "ny", "37": "nc", "38": "nd",
        "39": "oh", "40": "ok", "41": "or", "42": "pa", "44": "ri", "45": "sc", "46": "sd", "47": "tn", "48": "tx",
        "49": "ut", "50": "vt", "51": "va", "53": "wa", "54": "wv", "55": "wi", "56": "wy"
    }
    states=state_abbr.keys()
    if "STATES" in generalConfig:
        if generalConfig['STATES'] is not None:
            if "ALL" not in generalConfig["STATES"]:
                states=generalConfig["STATES"]

    fipscodes_df,num_split=state_selector(fipscodes_df=fipscodes_df, states=states, savechunks=savechunks)

    print("Downloading QWI county data from https://api.census.gov/data/timeseries/qwi/sa")
    print("Alternatively, you can visit https://ledextract.ces.census.gov/qwi/all and follow the instructions in the documentation.")
    if num_split>0: #if we are splitting the data
        state_groups = np.array_split(fipscodes_df, num_split)
        fulldf_pergroup = []
        for i, group in enumerate(state_groups, start=1):
            subdf=sub_downloadQWI(i=i, group=group, url=url, generalConfig=generalConfig, preprocessConfig=preprocessConfig, max_retries=max_retries)
            if fulldf_pergroup is not None:
                fulldf_pergroup=fulldf_pergroup.append(subdf)
            else:
                fulldf_pergroup=[subdf]
        if fulldf_pergroup is not None:
            outdf=pd.concat(fulldf_pergroup)
    else:
        outdf = sub_downloadQWI(i=1, group=fipscodes_df, url=url, generalConfig=generalConfig,
                                preprocessConfig=preprocessConfig, max_retries=max_retries)
    directory = f"{preprocessConfig['DATA_IN_FOLDER']}{preprocessConfig['QWIDIR']}"
    print("QWI Data successfully downloaded and saved to " + directory)
    print("\nSample data:")
    print(outdf.head())
    return(outdf)


########## CBP Download ##########
def download_rawCBP(fipscodes_df,preprocessConfig,generalConfig,max_retries = 3):
    year=generalConfig["YEAR"]
    api_key = generalConfig['API_KEY']
    url = "https://api.census.gov/data/"+str(year)+"/cbp"
    state_abbr = {
        "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca", "08": "co", "09": "ct", "10": "de",
        "11": "dc", "12": "fl", "13": "ga", "15": "hi", "16": "id", "17": "il", "18": "in", "19": "ia", "20": "ks",
        "21": "ky", "22": "la", "23": "me", "24": "md", "25": "ma", "26": "mi", "27": "mn", "28": "ms", "29": "mo",
        "30": "mt", "31": "ne", "32": "nv", "33": "nh", "34": "nj", "35": "nm", "36": "ny", "37": "nc", "38": "nd",
        "39": "oh", "40": "ok", "41": "or", "42": "pa", "44": "ri", "45": "sc", "46": "sd", "47": "tn", "48": "tx",
        "49": "ut", "50": "vt", "51": "va", "53": "wa", "54": "wv", "55": "wi", "56": "wy"
    }
    states=state_abbr.keys()
    if "STATES" in generalConfig:
        if generalConfig['STATES'] is not None:
            if "ALL" not in generalConfig["STATES"]:
                states=generalConfig["STATES"]
    fipscodes_df, num_split = state_selector(fipscodes_df=fipscodes_df, states=states, savechunks=1)

    print("Downloading CBP county data from "+url)
    print("Alternatively, you can visit https://www.census.gov/data/datasets/"+str(year)+"/econ/cbp/"+str(year)+"-cbp.html and selecting 'county file'")
    nyear=np.multiply(np.trunc((year-1997)//5),5)+1997
    if nyear<2012:
        raise Exception(f"Error: YEAR={year}. This code only supports API download after 2011 for CBP. ")

    naicstype="NAICS"+str(nyear.astype(int))

    if states is None or len(states)<1: #if using all states
        params = {
            'get': 'ESTAB,PAYQTR1,PAYQTR1_N_F,EMP,EMP_N_F,PAYANN,PAYANN_N_F,YEAR',
            'for': 'county:*',
            'in': 'state:*',
            naicstype: '*',
            'key': api_key
        }
    else: #otherwise use specific states
        statestr=fipscodes_df.iloc[:,0].astype(str).str.zfill(2) #fix state format
        params = {
            'get': 'ESTAB,PAYQTR1,PAYQTR1_N_F,EMP,EMP_N_F,YEAR',
            'for': 'county:*',
            'in': 'state:'+statestr.str.cat(sep=","),
            naicstype: '*',
            'key': api_key
        }
    for retry in range(max_retries): #try a couple times
        response = requests.get(url, params=params)
        if response.ok:#status_code == 200:
            data = response.json()
            print(f"Successfully fetched CBP data")
            break
        else:
            print(
                f"Attempt {retry + 1} failed for CBP data - Status: {response.status_code}")
            time.sleep(
                2)  # Wait. Sometimes you hit rate limits and need to wait a second before making more API calls
    headers = data[0]
    rows = data[1:]
    if not response.ok:
        print(f"Failed after {max_retries} retries for {row['name']} (FIPS: {fips_code})")
    else:
        df = pd.DataFrame(rows, columns=headers)
        df.rename(columns={"county":"fipscty", #rename columns
                           "state":"fipstate",
                           naicstype:"naics",
                           "ESTAB":"est",
                           "PAYQTR1":"qp1",
                           "PAYQTR1_N_F":"qp1_nf",
                           "EMP":"emp",
                           "EMP_N_F":"emp_nf"},inplace=True)
        df['year']=generalConfig["YEAR"]
        df['naics']=df['naics'].astype(str)
        df['naics'] = df['naics'].apply(format_naics) #formate the naics codes
        directory = f"{preprocessConfig['DATA_IN_FOLDER']}{preprocessConfig['CBPDIR']}"
        filename= f"{preprocessConfig['DATA_IN_FOLDER']}{preprocessConfig['CBPDATA']}"
        os.makedirs(directory, exist_ok=True)
        df.to_csv(filename, index=False)
        print("CBP Data successfully downloaded and saved to "+filename)
        print("\nSample data:")
        print(df.head())
    return(df)


#with open('config_pre2017.yaml','r') as configFile:
#    config = yaml.safe_load(configFile)
#    preprocessConfig = config['preprocessConfig']
#    generalConfig = config['generalConfig']
#fipscodes_df = pd.read_csv("./DataDiag/FIPSstatecodename.txt")
#print(fipscodes_df.iloc[:,0].isin(["10","11"]))
#api_key = generalConfig['API_KEY']
#print(generalConfig)
#tempCBP=download_rawCBP(fipscodes_df=fipscodes_df,preprocessConfig=preprocessConfig,generalConfig=generalConfig,max_retries = 3)
#tempQWI=download_QWI(fipscodes_df=fipscodes_df,preprocessConfig=preprocessConfig,generalConfig=generalConfig,max_retries = 3,savechunks=2)
#
#print(tempQWI['sPayroll'].value_counts())
#print(tempQWI.head())
# print(tempQWI.columns)
#print(tempCBP['fipstate'].value_counts())
# print(tempCBP.columns)