import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import statsmodels.api as sm
import sys
import os
import yaml
import random
# Load config file
with open('config.yaml', 'r') as configFile:
    config=yaml.safe_load(configFile)
generalConfig=config['generalConfig']
getNAICS6Config=config['getNAICS6Config']
employmentConfig=config['employmentConfig']
microdataConfig=config['microdataConfig']
sys.path.append(os.path.abspath(getNAICS6Config['FUNCTIONSDIR']))
from getAggLevelSummaries import *
from EmploymentFunctions import *
from WageFunctions import *
from NAICS6functions import *
from get_microdata import *
from MicrodataPostprocessing import *
pd.set_option("display.max_columns", None)  
onROAR=True

##################  Dataframes set up  #######################
# Load main dataset. Location is set in the config.yaml
# Tell the user where the dataset is located
print('---------- Loading Dataset ----------\n')
print(f"Dataset location: {getNAICS6Config['DATASET']}")
df = pd.read_csv(getNAICS6Config['DATASET'], dtype=str)#, nrows=100000)
df['year'] = 2016
df['quarter'] = 1
df = df.iloc[:, 1:] # Remove index
dfsave = df.copy()
df = pd.concat([df, pd.DataFrame([{
    "state": "29", "cnty": "198", "emp": "1", "geoindkey": "29189_525990",
    "qp1_nf": "Impute", "qp1": "0", "estnum": "2", "geo_level": "C",
    "geography": "29189", "ind_level": "6", "industry": "525990",
    "year": "2016", "quarter": "1",
    "EarnBeg": None, "EarnHirAS": None, "Emp": None, "EmpEnd": None, "EmpS": None,
    "sEarnBeg": None, "sEarnHirAS": None, "sEmp": None, "sEmpEnd": None, "sEmpS": None
}])], ignore_index=True)
df.loc[df['geoindkey'].str.contains("29189_5259//", regex=True), 'qp1_nf'] = "Impute"
# Extract 6-digit NAICS
df6 = df[df['geoindkey'].str.contains("_[0-9]{6}", regex=True)].copy()
df6['geo4naics'] = df6['geoindkey'].str[:-2]
df6['geo5naics'] = df6['geoindkey'].str[:-1]
df6 = df6[['geoindkey', 'geo4naics', 'geo5naics', 'state', 'cnty', 'estnum', 'qp1', 'qp1_nf', 'emp']]
# Get summary counts for distribution
count6dig = get_codes_summary(df, groupbydigits=4, levelgrouped=6)
# Filter and prepare 4-digit NAICS data
df4 = df[df['industry'] != "------"].copy()
df4 = df[df['industry'].notna()].copy() 
df4 = df4[df4['industry'].str.match(r"^[0-9]{4}[^0-9]{2}", na=False)]
# Create derived columns
df4['sector'] = df4['industry'].str[:2]
df4['state'] = df4['geography'].str[:-3]
df4['geo4naics'] = df4['geoindkey'].str[:-2]
df4['geo3naics'] = df4['geoindkey'].str[:-3]
# Merge with summary data
df4 = df4.merge(count6dig, on='geo4naics', how='left')
df4['wagediff'] = df4['qp1'].astype(float) - df4['wageCBP_sum6by4'].astype(float)
columns_to_convert = ['emp', 'qp1', 'estnum', 'year', 'quarter', "sEmp"]
df4[columns_to_convert] = df4[columns_to_convert].astype(float)

############## Employment Counts ################
print('---------- Employment Configuration ----------')
# Display all current employmentConfig settings
for key, value in employmentConfig.items():
    print(f"{key}: {value}")
# Step 1: Create employment prediction model
print('---------- Imputing Employment Data ----------')
m1empfit = get_m1emp_model(df=df4)
# Step 2: Generate monthly employment counts
empMat = get_employmentCounts4(
    df4,
    m1emp_model=m1empfit,
    m2emp_noisecoef=employmentConfig['M2EMP_NOISECOEF'],
    rseed=employmentConfig['RSEED'],
    include_m1emp_indicator=True
)
# Step 3: Adjust to match county totals
adjustdf = pd.DataFrame(empMat)
adjustdf = adjustdf.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.name and 'm' in col.name else col)
adjustm1emp = adjust_countytotal_qwi(valdf=adjustdf, sumdf=df[df["industry"] == "------"])
empMatA = empMat.copy() 
empMatA.iloc[:, 1] = adjustm1emp # Update with adjusted values

################## WAGES #######################
print('---------- Wage Configuration ----------')
# Display all current wageConfig settings
for key, value in wageConfig.items():
    print(f"{key}: {value}")
# Step 1. Create wage prediction model
print('---------- Imputing Wage Data ----------')
wagefit_sub= get_wage_model(df=df4, emp_mat_adj = empMatA)
# Step 2. Get min/max bounds
wage_maxmin=get_maxmindf(df4dig=df4,fulldf=df,emp_mat_adj = empMatA)
# Prepare employment matrix
empMatwage = pd.DataFrame({
    "m1emp": empMatA.iloc[:, 1],
    "m2emp": empMatA.iloc[:, 2],
    "m3emp": empMatA.iloc[:, 3]
})
# Step 3: Impute wage values
wagesout=get_wages4(df4=df4,empmat=empMatwage,wagemodel=wagefit_sub,useEarnQWI = True, maxmindf=wage_maxmin)
# Prepare final 4 digit output
df4imp = wagesout.copy().assign(
    EmpScale = lambda x: np.where(x['m3emp'] == 0, 1, x['emp']/x['m3emp'].astype(float)),
    geo4naics = lambda x: x['geoindkey'].str[:-2],
    m1empFromModel = empMatA.iloc[:, 4].astype(float)  # 5th column (0-based index 4)
)

################## Get NAICS6 By County Aggregates #######################
print('---------- Getting NAICS6 by County Aggregates ----------')
print('This may take a while, please be patient...')
# Distribute values from 4-digit to 6-digit NAICS
naics6df = get_6naics_all(df=df6,df4n=df4imp,codes4summary = count6dig)
# Final formatting and output
naics6df = naics6df.iloc[:, :5].join(naics6df[['m1emp', 'm3emp', 'wage']])
naics6df.head(50)
naics6df.to_csv('DataNAICS6.csv')

################## Get Microdata #######################
# Display all current microdataConfig settings
print('---------- Microdata Configuration ----------')
for key, value in microdataConfig.items():
    print(f"{key}: {value}")
print('---------- Making Synthetic Microdata ----------')
print('This may take a while, please be patient...')
naics6df = pd.read_csv("./DataNAICS6.csv")#, nrows=10000)
random.seed(employmentConfig['RSEED'])
tempdf = make_syn_microdata(naics6df,numchunk=microdataConfig['NUMCHUNK'],outfoldername=microdataConfig['OUTPATH'])

################## Microdata Postprocessing #######################
print('---------- Generating Final Microdata ----------')
combine_and_split_iterative(yr=generalConfig['YEAR'], qtr=generalConfig['QTR'])
