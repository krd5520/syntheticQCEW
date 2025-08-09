import pandas as pd
import os
import yaml
#load the config file
with open('config.yaml', 'r') as configFile:
    config = yaml.safe_load(configFile)
preprocessConfig = config['preprocessConfig']
###################################################################################
############## Functions for Preprocessing and combining 3 Establishment Datasets
###################################################################################
## Focus on Quarter 1: 2016
#         1. CBP Raw data (see https://www.census.gov/data/datasets/2016/econ/cbp/2016-cbp.html)
#         2. CBP Imputed data
#                   Eckert, Fabian, Fort, Teresa C., Schott, Peter K., and Yang, Natalie J.
#                   County Business Patterns Database. Ann Arbor, MI: Inter-university Consortium
#                   for Political and Social Research [distributor], 2020-01-31. https://doi.org/10.3886/E117464V1
#        3. QWI data from U.S. Census
#                downloaded using Data Extraction Tool at https://ledextract.ces.census.gov/
#                   get every county for each of the 50 states, DC, and Puerto Rico
#                   no firm or demographic information
#                   all 4 digit NAICS codes
#                   private ownership
#                   measures: Emp,EmpEnd,EmpS,EarnHirAS,EarnBeg
#                   for quarter 1 2016
####
#### All data will be combined on a created key which is [geography code]_[industry code]
#### geography code format in QWI: [state][leading zeros and county code (must be 3 characters)]
#### industry code form in CBP: '------' if total over all industries, '##----' for sector,
####                            otherwise NAICS code with trailing '/' to be six characters
###############################################################################################
###############################################################################################

####
## takes CBP format of fips state and county codes as 2 columns to
## the QWI format of [state code] + [3 characters: leading zeros and county code]
# INPUT: pandas dataframe with columns 'fipstate' & 'fipscty' for state & county codes
# OUTPUT: pandas series of QWI format geography
def fips_to_geography(df):
    # combine fipstate and fipscty to match geography format of QWI files.
    # state + leading zeros to make county number 3 characters
    df[['fipstate', 'fipscty']] = df[['fipstate', 'fipscty']].astype(str)
    df['fipscty'] = df['fipscty'].str.zfill(3)
    return(df['fipstate'] + df['fipscty'])

## NOT CURRENTLY USED
def get_state_totals_emp(df):
    #subset to only necessary columns
    dfsub = df[['state','emp','industry','estnum']]
    #group by state and industry code
    dfsub.groupby(['state','industry']).sum()
    # make indentifying key
    dfsub['geoindkey'] = dfsub['state']+"_"+dfsub['industry']
    #unify column names
    #dfsub.rename(columns={"est":"estnum",
    #                      "fipstate":"state"},inplace=True)
    dfsub['cnty'] = "---"
    dfsub['geography'] = dfsub['state']
    dfsub['geo_level'] = "S"
    return(dfsub)

####
# Take imputed CBP data and add geography variable to match QWI format and unique key
# INPUT: pandas data frame the imputed CBP data from:
# #     Eckert, Fabian, Fort, Teresa C., Schott, Peter K., and Yang, Natalie J.
# #     County Business Patterns Database. Ann Arbor, MI: Inter-university Consortium
# #     for Political and Social Research [distributor], 2020-01-31. https://doi.org/10.3886/E117464V1
# OUTPUT: pandas dataframe with  columns "state","cnty","geography","key","industry", and "emp"
def preprocess_imputedCBP(imputeCBP):
   imputeCBP['geography'] =fips_to_geography(imputeCBP)
   #unique identifier
   imputeCBP['geoindkey'] = imputeCBP['geography'] + "_" + imputeCBP['naics']
   #unify column names
   imputeCBP.rename(columns={"fipstate":"state",
                              "fipscty":"cnty",
                              "naics":"industry"},inplace=True)
   return(imputeCBP)

####
## preprocess the raw CBP data (county level files)
## by creating identifing key, and subsetting to include
## quarter 1 wages, flag for quarter 1 wages, number of establishments
# INPUT: pandas dataframe of the raw county-level CBP data
#           needs to include columns: fipstate,fipscty,naics,qp1,qp1_nf,est
# OUTPTU: pandas dataframe with columns:
#           indentifing key (geoindkey),
#           quarter 1 wages (qp1),
#           flag for quarter 1 wages (qp1_nf),
#           number of establishments (est)
def preprocess_rawCBPcnty(raw):
    raw['geography'] = fips_to_geography(raw)
    raw['geoindkey'] = raw['geography']+"_"+raw['naics']
    return(raw[['geoindkey','qp1_nf','qp1','est']])

#    rawdf =pd.read_csv(raw_co_file)
#    imputedf = pd.read_csv(imputefile)

## Combine the Raw and Imputed CBP data to have a dataset of employment count,
## quarterly wages, number of establishments, by county and industry level for Q1 2016
# INPUT: 2 unprocessed CBP pd.dataframes. rawdf and imputedf
#       to be fed to preprocess_rawCBPcnty() and preprocess_imputedCBP() respectively
#       rawdf should include 50 states, Puerto Rico, and Washington DC
# OUTPUT: pd.dataframe of CBP data with employment inputted and Q1 wages possibly suppressed.
def combine_CBP_raw_imputed(rawdf,imputedf,forcombine=False):
    #preprocess the dataframes
    rawdf = preprocess_rawCBPcnty(rawdf)
    imputedf = preprocess_imputedCBP(imputedf)
    #combine on geoindkey (keep all rows from imputted value
    cbpdf = imputedf.merge(rawdf,on=['geoindkey','geoindkey'],how="left",indicator=True)
    #unify column names
    cbpdf.rename(columns={"est":"estnum"},inplace=True)
    if not forcombine:
        cbpdf['geo_level']="C" #county level for all of these

    misscount = sum(cbpdf['_merge']=="left_only")
    if misscount>0: #if inputted data missing a combination, something has gone wrong and should be investigated
        raise Warning("Some county industry code combinations in imputed, but not raw CBP data: "+str(misscount))

    cbpdf = cbpdf.drop(columns=['_merge'])
    return(cbpdf)

## preprocess QWI data for combination with CBP data
# INPUT: pd.dataframe of QWI data should contain columns:
#       Emp,EmpEnd,EmpS,EarnHirAS,EarnBeg, industry, geography, year, quarter
# OUTPUT: pd.dataframe with unique key and reduced to key columns
def preprocess_qwi(qwi,forcombine=False):
    #drop unneccessary columns (come with all downloads of QWI data from US Census site)
    qwi = qwi.drop(columns=['periodicity','seasonadj','agegrp','race','ethnicity','education',
              'sex','ownercode','firmage','firmsize','version'],errors='ignore')

    #change industry format to match CBP
    # If sum over all "------", otherwise naisc with trailing '/' to be 6 characters
    qwi[['geography','industry']] = qwi[['geography','industry']].astype(str)
    qwi.loc[qwi.ind_level=="A",'industry'] = "------"
    qwi['industry'] = qwi['industry'].str.rjust(6,"/")

    # make unique indentifier
    qwi['geoindkey'] = qwi['geography']+"_"+qwi['industry']
    if forcombine:
        qwi.drop(columns=['industry','geography'],inplace=True)
    return(qwi)

## Loop through qwi files with 'co' in file name
##  (my file names have been manually changes to reflect "co" for county level, and "st" for state level
# INPUT: str which is the path to the folder including QWI data files
# OUTPUT: pd.dataframe with all qwi county files (labeled with "co" in file name) preprocessed and combined.
def read_qwi_co(folderpath,forcombine=False):
    lsdf = [] #initialize list

    fileexists=False
    #for files in the specified folder path
    for file in os.listdir(folderpath):
       # if the file name includes "co" in it, read data, preprocess it, add it to the list
        if "co" in file:
            fileexists=True
            df = pd.read_csv(folderpath+str(file))
            df = preprocess_qwi(df,forcombine=forcombine)
            dfnew = df.loc[df['geo_level']=="C"] #make sure this is just county data
            lsdf.append(dfnew)
    if not fileexists:
        print("No county files with 'co' in the name in QWI folder.")
    qwidf = pd.concat(lsdf,axis=0,ignore_index=True) #combine all of these
    return(qwidf)

## Combine QWi and CBP Data
# INPUTS:
#       rawfile: str for file path of raw cbp county data
#       imputedfile: str for file path of imputed cbp data
#       qwifolder: str for folder path of folder that includes qwi files with county-level names including "co"
#       printdiagnostics: logical, if True print shapes of CBP and QWI data and count row in imputed CBP not in QWI
# OUTPUTS: pd.dataframe of combined CBP and QWI
def combine_qwi_cbp(rawfile, imputedfile,qwifolder,outfilename="combineQWIandCBP.csv",printdiagnostics=False,outfilepath="PythonPreprocessOut"):
    if not os.path.isfile(rawfile):
        print(f"Cannot locate file named {rawfile} ")
    if not os.path.isfile(imputedfile):
        print(f"Cannot locate file named {imputedfile} ")
    if not os.path.exists(qwifolder):
        print(f"Cannot locate directory named {qwifolder} ")
    imputeCBP = pd.read_csv(imputedfile) #read imputed data file (only includes Mid March Employment)
    raw = pd.read_table(rawfile,sep=",") #read raw CBP file for counties in 50 states

    cbp = combine_CBP_raw_imputed(raw,imputeCBP,forcombine=True) #combine imputted and raw

    qwi =read_qwi_co(qwifolder,forcombine=True) #read all qwi county files

    #combine QWI and CBP
    combinedf = cbp.merge(qwi,on=['geoindkey', 'geoindkey'],how="left",indicator=True)

    misscount1 = sum(combinedf['_merge'] == "right_only")
    misscount2 = sum(combinedf['_merge'] == "left_only")
    #possible diagnostics
    if not os.path.exists(outfilepath):
        os.mkdir(outfilepath)
    if printdiagnostics:
        with open(outfilepath+'diagnosticsCombines.txt','w') as f:
            print("Number of rows,columns in CBP: ",cbp.shape,file=f)
            print("Number of rows,columns in QWI: ", qwi.shape, file=f)
            print("Number of rows in CBP but not QWI: ", misscount2, file=f)
            print("Number of rows in QWI but not CBP: ", misscount1, file=f)
    #print(list(combinedf.columns))
    #if misscount>0:
    #    raise Warning("Some county industry code combinations in imputed, but not QWI data: "+str(misscount))
    #    print(combinedf['geoindkey'][[combinedf['_merge'] == "left_only"]])
    combinedf = combinedf.drop(columns=['_merge'])
    combinedf.to_csv(outfilepath+outfilename)
    return(combinedf)

#
foldername = preprocessConfig['DATA_IN_FOLDER']
combine_qwi_cbp(rawfile=foldername + preprocessConfig['CBPDATA'],
                   imputedfile=foldername + preprocessConfig['IMPUTECBP'],
                   qwifolder=foldername + preprocessConfig['QWIDATA'],
                   outfilename='combineQWIandCBP.csv',
                   printdiagnostics=True,
                   outfilepath = preprocessConfig['OUTPATH'])
