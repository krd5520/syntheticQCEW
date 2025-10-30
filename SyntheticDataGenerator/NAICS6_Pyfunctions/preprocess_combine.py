import numpy as np
import pandas as pd
import os
import yaml
import sys

sys.path.append(os.path.abspath("./NAICS6_Pyfunctions/"))
from hierarchy_geoindkey import *
from GeneralFunctions import *
from investigate_preprocess import *

pd.set_option("display.max_columns", None)
pd.options.mode.copy_on_write = True

excluded_cbp=["92----","111///","112///","482///","491///","814///","525110", "525120","525190","525920","541120"]

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

#### OLD: NOT IN USE
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
   imputeCBP["state"]=imputeCBP["state"].astype(int).astype(str)
   imputeCBP["cnty"]=imputeCBP["cnty"].astype(int).astype(str).str.zfill(3)
   return(imputeCBP)


####
####
# Take imputed CBP data and add geography variable to match QWI format and unique key
# INPUT: file name and path for imputed CBP data from:
# #     Eckert, Fabian, Fort, Teresa C., Schott, Peter K., and Yang, Natalie J.
# #     County Business Patterns Database. Ann Arbor, MI: Inter-university Consortium
# #     for Political and Social Research [distributor], 2020-01-31. https://doi.org/10.3886/E117464V1
# #  if there is another file by that name but with a _lw_up.csv suffix (i.e. lower and upper bounds
# #     from https://fpeckert.me/cbp/, this function will read that file in and merge them
# OUTPUT: pandas dataframe with  columns "state","cnty","geography","key","industry", and "emp"
def preprocess_imputedCBP_file(imputefile,prefer_lw_up=True):
    imputeCBP=pd.read_csv(imputefile)
    imputeCBP['geography'] = fips_to_geography(imputeCBP)
    # unique identifier
    imputeCBP['geoindkey'] = imputeCBP['geography'] + "_" + imputeCBP['naics']
    lwupfile=imputefile.replace(".csv","_lw_up.csv")
    if os.path.exists(lwupfile):
        #read in file and make geoindkey
        lwupdf=pd.read_csv(lwupfile)
        lwupdf['geography'] = fips_to_geography(lwupdf)
        # unique identifier
        lwupdf['geoindkey'] = lwupdf['geography'].astype(str) + "_" + lwupdf['naics']

        #merge with other imputeCBP data
        imputeCBP=imputeCBP.merge(lwupdf,on=['geoindkey','geoindkey'],how="outer",suffixes=['_raw',""],indicator=True,validate="one_to_one")
        #for rows only in other imputeCBP, fill in the geographic and industry features
        left_indic=imputeCBP['_merge']=="left_only"
        if left_indic.sum()>0:
            for colstr in ['fipstate','fipscty','naics','geography']:
                imputeCBP.loc[left_indic,colstr]=imputeCBP.loc[left_indic,colstr+'_raw']
        eqlbubdf=imputeCBP.loc[imputeCBP["lb"]==imputeCBP['ub'],:]
        if "emp_raw" in imputeCBP.columns:
            eqhasemp=eqlbubdf.loc[~eqlbubdf['emp_raw'].isna(),:]
            empcolname="emp_raw"
        else:
            eqhasemp = eqlbubdf.loc[~eqlbubdf['emp'].isna(), :]
            empcolname="emp"
        checkeqdf=eqhasemp.loc[eqhasemp.loc[:,empcolname]==eqhasemp['lb'],:]
        if len(checkeqdf)!=len(eqhasemp):
            print("Lower and upper bounds are inconsistent with employment. "+str(len(checkeqdf))+"/"+str(len(eqhasemp))+" are consistent.")
        else:
            imputeCBP.loc[imputeCBP["lb"] == imputeCBP['ub'], empcolname] = imputeCBP.loc[
                imputeCBP["lb"] == imputeCBP['ub'], "lb"]
        if len(eqlbubdf)!=len(imputeCBP):
            print("In lw_up file, there are "+str(len(eqlbubdf))+"/"+str(len(lwupdf))+" rows where the lower bound=upper bound.")
        imputeCBP.loc[(imputeCBP.loc[:,empcolname].isna())&(imputeCBP['lb']==imputeCBP['ub']),empcolname]=imputeCBP.loc[(imputeCBP.loc[:,empcolname].isna())&(imputeCBP['lb']==imputeCBP['ub']),'lb']
        imputeCBP.loc[(imputeCBP.loc[:,empcolname].isna()) & (imputeCBP['lb'] != imputeCBP['ub']), empcolname] = imputeCBP.loc[(imputeCBP.loc[:,empcolname].isna()) & (imputeCBP['lb'] != imputeCBP['ub']),:].apply(random_midpoint,axis=1)
    elif "emp" not in imputeCBP.columns:
        imputeCBP['emp']=imputeCBP['lb']
        imputeCBP.loc[imputeCBP['lb'] != imputeCBP['ub'], 'emp'] = imputeCBP.loc[imputeCBP['lb'] != imputeCBP['ub'],:].apply(random_midpoint, axis=1)

    #unify column names
    imputeCBP.rename(columns={"fipstate":"state",
                              "fipscty":"cnty",
                              "naics":"industry"},inplace=True)
    imputeCBP["state"] = imputeCBP["state"].astype(int).astype(str)
    imputeCBP["cnty"] = imputeCBP["cnty"].astype(int).astype(str).str.zfill(3)
    imputeCBP=imputeCBP[imputeCBP['cnty']!="999"]
    if "emp_raw" in imputeCBP.columns:
        imputeCBP.rename(columns={"emp_raw": "emp"}, inplace=True)
    if os.path.exists(lwupfile):
        numnaemp=imputeCBP['emp'].isna().sum()
        print(f"imputeCBP: Number of NAs in emp column is: {numnaemp}")
        return(imputeCBP[['geoindkey','state','cnty','industry','geography','emp','lb','ub']])
    else:
        numnaemp = imputeCBP['emp'].isna().sum()
        print(f"imputeCBP: Number of NAs in emp column is: {numnaemp}")
        return (imputeCBP[['geoindkey', 'state', 'cnty', 'industry', 'geography', 'emp']])


####
## preprocess the raw CBP data (county level files)
## by creating identifing key, and subsetting to include
## quarter 1 wages, flag for quarter 1 wages, number of establishments
# INPUT: pandas dataframe of the raw county-level CBP data
#           needs to include columns: fipstate,fipscty,naics,qp1,qp1_nf,est
#        withemp is boolean for whether employment column should be in outputted data
#        supptab is boolean for whether the table of suppression counts/percents should be outputted
# OUTPTU: pandas dataframe with columns:
#           indentifing key (geoindkey),
#           quarter 1 wages (qp1),
#           if withemp==True, march employment (emp),
#           flag for quarter 1 wages (qp1_nf),
#           number of establishments (est)
#       if supptab==True, then a dataframe for the suppression counts with columns:
#               aggregate level code (agglvl_code),
#               number of cells in aggregate level (n)
#               count of cells where wage is suppressed (wage_suppressed)
#               percent of cells where wage is suppressed (wage_prop)"
##               count of cells where employment is suppressed (emp_suppressed)
#               percent of cells where employment is suppressed (emp_prop)"
def preprocess_rawCBPcnty(raw,supptab=False,suppressionflags=["S", "D"]):
    raw['geography'] = fips_to_geography(raw)
    raw['geoindkey'] = raw['geography']+"_"+raw['naics']
    #raw.drop(columns=["geography"],inplace=True)
    suppressemp= raw['emp_nf'].isin(suppressionflags)  # |tempraw['emp_nf']=="D"
    suppresswage = raw['qp1_nf'].isin(suppressionflags)  # tempraw['qp1_nf']=="D"
    raw.loc[suppressemp,'emp']=np.nan
    raw.loc[suppresswage,'qp1']=np.nan

    #unify column names
    raw.rename(columns={"fipstate": "state",
                        "fipscty": "cnty",
                        "naics": "industry"}, inplace=True)
    raw["state"] = raw["state"].astype(int).astype(str)
    raw["cnty"] = raw["cnty"].astype(int).astype(str).str.zfill(3)
    raw = raw[raw['cnty'] != "999"]

    dignum = raw['industry'].str.count(r'\d')  # number of digits in naics
    dignum[dignum==0]=-1 #adjust for state level
    #cntyindic = raw['cnty'].str.contains(r'\d').map(
    #    {True: 70, False: 50})  # [re.match("[0-9]*",fips) for fips in tempraw['fipscty'].tolist]
    raw['agglvl_code'] = [x+72 for x in dignum]

    if supptab: #make suppression table
        tempraw=raw #save as copy
        #dignum=tempraw.industry
        #get agglvl_code, 51+# digits in naics if state level, +20 more if cnty level.


        #get indicator of emp or wage being suppressed
        tempraw['suppressemp']=tempraw['emp_nf'].isin(suppressionflags)
        tempraw["suppresswage"]= tempraw['qp1_nf'].isin(suppressionflags)

        supptabwage=pd.crosstab(tempraw['agglvl_code'], tempraw['suppresswage']).join(pd.crosstab(tempraw['agglvl_code'], tempraw['suppresswage'],normalize="index"),
                                                                                on="agglvl_code",how='outer',lsuffix='_count',rsuffix='_prop',sort=True)

        supptabemp = pd.crosstab(tempraw['agglvl_code'], tempraw['suppressemp'])\
            .join(
            pd.crosstab(tempraw['agglvl_code'], tempraw['suppressemp'], normalize="index"),
            on="agglvl_code", how='outer', lsuffix='_count', rsuffix='_prop', sort=True)
        supptabfull=pd.DataFrame({'agglvl_code':supptabwage.index.values,'n':supptabwage['True_count'].add(supptabwage['False_count']).values,
                                  'wage_suppressed':supptabwage['True_count'].values,"wage_prop":supptabwage["True_prop"].multiply(100).round(0).values,
                                 'emp_suppressed':supptabemp['True_count'].values,"emp_prop":supptabemp["True_prop"].multiply(100).round(0).values})#.join(supptabemp,on='agglvl_code',how='outer',lsuffix='_wage',rsuffix='_emp')

    raw=raw[['geoindkey', 'qp1_nf', 'qp1', 'est','emp','geography',"industry","state","cnty",'emp_nf','agglvl_code']]
    if supptab: #if returning suppression table
        return (raw,supptabfull)
    else:
        return (raw)



## Combine the Raw and Imputed CBP data to have a dataset of employment count,
## quarterly wages, number of establishments, by county and industry level for Q1 2016
# INPUT: 2 unprocessed CBP pd.dataframes. rawdf and imputedf
#       to be fed to preprocess_rawCBPcnty() and preprocess_imputedCBP() respectively
#       rawdf should include 50 states, Puerto Rico, and Washington DC
#       if imputedf is not provided it is assumed that rawdf have complete emp
#       onlyraw indicates if only rawdf is used, supptab indicates if suppression table should be outputted
# OUTPUT: pd.dataframe of CBP data with employment inputted and Q1 wages possibly suppressed.
def combine_CBP_raw_imputed(rawdf,imputedf=None,states=None, onlyraw=False,supptab=False):
    #preprocess the dataframes
    if imputedf is None or onlyraw:
        if supptab:
            cbpdf, supptabfull = preprocess_rawCBPcnty(rawdf,supptab=supptab)
        else:
            cbpdf = preprocess_rawCBPcnty(rawdf)
    else:
        if supptab:
            rawdf, supptabfull = preprocess_rawCBPcnty(rawdf, supptab=supptab)
        else:
            rawdf = preprocess_rawCBPcnty(rawdf)
        #imputedf = preprocess_imputedCBP(imputedf)
        #combine on geoindkey (keep all rows from imputted value

        cbpdf = imputedf.merge(rawdf,on=["geoindkey","state","cnty","geography","industry"],how="outer",indicator=True,suffixes=["","_raw"],validate="one_to_one")

        #check the states
        #cbpdf = fill_from_geoindkey(cbpdf)
        lonly_states=cbpdf.loc[cbpdf['_merge']=='left_only','state'].unique()
        both_or_ronly_states=cbpdf.loc[cbpdf['_merge']!='left_only','state'].unique()
        states_shared=np.intersect1d(lonly_states,both_or_ronly_states)
        if states_shared.size>0:
            problemdf=cbpdf.loc[(cbpdf['_merge']=="left_only")&(cbpdf['state'].isin(both_or_ronly_states.tolist())),:]
            lenprob=len(problemdf)
            problemdf=problemdf.loc[problemdf['cnty']!="999",:]
            if len(problemdf)>0:
                print("There are left_only cells from the states of interest: "+str(lenprob)+" but only "+str(len(problemdf))+" are not from cnty==999")
                print(problemdf.head())
            cbpdf=cbpdf[cbpdf['state'].isin(both_or_ronly_states)]
        print("States in CBP raw: "+", ".join(both_or_ronly_states))
        state_abbr = {
            "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca", "08": "co", "09": "ct", "10": "de",
            "11": "dc", "12": "fl", "13": "ga", "15": "hi", "16": "id", "17": "il", "18": "in", "19": "ia", "20": "ks",
            "21": "ky", "22": "la", "23": "me", "24": "md", "25": "ma", "26": "mi", "27": "mn", "28": "ms", "29": "mo",
            "30": "mt", "31": "ne", "32": "nv", "33": "nh", "34": "nj", "35": "nm", "36": "ny", "37": "nc", "38": "nd",
            "39": "oh", "40": "ok", "41": "or", "42": "pa", "44": "ri", "45": "sc", "46": "sd", "47": "tn", "48": "tx",
            "49": "ut", "50": "vt", "51": "va", "53": "wa", "54": "wv", "55": "wi", "56": "wy"
        }
        states = state_abbr.keys()
        if "STATES" in generalConfig and generalConfig['STATES'] is not None:
            if "ALL" not in generalConfig["STATES"]:
                states = generalConfig["STATES"]
                if states[0] not in state_abbr.keys():
                    states = [str(int(key)) for key, value in state_abbr.items() if value.upper() in states]
                cbpdf = cbpdf.loc[cbpdf['state'].isin(states), :].copy()

        emp_impute_naI=cbpdf["emp"].isna()
        emp_raw_naI=cbpdf['emp_raw'].isna()
        specialcase=cbpdf.loc[(emp_impute_naI)&(~emp_raw_naI),'emp_raw']
        if specialcase.empty:
            pass
        else:
            cbpdf.loc[(emp_impute_naI) & (~emp_raw_naI), "emp"] = specialcase
            not99=cbpdf.loc[(emp_impute_naI) & (~emp_raw_naI) &(~cbpdf['geoindkey'].str.endswith("99----")), ]
            print("There are " + str(
                len(specialcase)) + " rows in rawCBP but not imputeCBP which have March employment values, and "+str(len(not99))+" are not 99----")

        ## Investigate weird stuff???
        impute_misscount = cbpdf["emp_nf"].isna().sum()#sum(cbpdf['_merge'] == "left_only")
        raw_misscount=sum(cbpdf['_merge']=="right_only")
        missing_emp=cbpdf['emp'].isna().sum()
        # #Note some columns appear in raw but not imputed.
        # # In 2016, These are all counties with less than 6 establishments. Most are suppressed.
        # # The not suppressed ones have 0 and 0 emp and qp1
        if missing_emp+raw_misscount+impute_misscount>0:
            print("In impute but not rawCBP: "+str(impute_misscount)+
                  "; In rawCBP but not impute: "+str(raw_misscount)+
                  "; Missing imputed emp in combined CBP:"+str(missing_emp))

        if missing_emp>0:
            raise Exception(f"something wrong missing emp values...{missing_emp}")
        #drop observations in imputed but not Census (these are from unselected states)
            print("Adjusting missing imputed employment...")
            cbpdf['_merge']=cbpdf['_merge'].cat.add_categories(['ronly_fill0','ronly_impute'])
            cbpdf=cbp_mismatch(cbpdf)
            new_raw_misscount=sum(cbpdf['_merge'] == "right_only")
            # #Note some columns appear in raw but not imputed.
            print("After adjustments, in rawCBP but not impute: "+str(raw_misscount))

    #unify column names
    cbpdf['_merge']=cbpdf['_merge'].str.replace('left_only','impute')
    cbpdf['_merge']=cbpdf['_merge'].str.replace('right_only','raw')
    cbpdf['_merge']=cbpdf['_merge'].str.replace('both','raw_impute')
    cbpdf.rename(columns={"est":"estnum","_merge":"_merge_rawimpute","emp":"emp3_cbp","qp1":"wages_cbp","emp_nf":"emp3_cbp_flag","qp1_nf":"wages_cbp_flag"},inplace=True)
    #cbpdf['geo_level']="C" #county level for all of these
    if cbpdf['emp3_cbp'].isna().sum()>0:
        cbpdf.loc[cbpdf['emp3_cbp'].isna(),"emp3_cbp"]=cbpdf.loc[cbpdf['emp3_cbp'].isna(),"emp_raw"]
    cbpdf.drop(columns=['lb', 'ub', 'emp_raw'], errors='ignore',inplace=True)
    if supptab:
        return(cbpdf,supptabfull)
    else:
        return(cbpdf)

## preprocess QWI data for combination with CBP data
# INPUT: pd.dataframe of QWI data should contain columns:
#       Emp,EmpEnd,EmpS,EarnHirAS,EarnBeg, industry, geography, year, quarter
# OUTPUT: pd.dataframe with unique key and reduced to key columns
def preprocess_qwi(qwi,generalConfig):
    #drop unneccessary columns (come with all downloads of QWI data from US Census site)
    qwi = qwi.drop(columns=['periodicity','seasonadj','agegrp','race','ethnicity','education',
              'sex','ownercode','firmage','firmsize','version'],errors='ignore')

    #change industry format to match CBP
    # If sum over all "------", otherwise naisc with trailing '/' to be 6 characters
    qwi[['geography','industry']] = qwi[['geography','industry']].astype(str)
    qwi.loc[qwi.ind_level=="A",'industry'] = "------"
    qwi.loc[qwi.ind_level=="S",'industry'] = qwi.loc[qwi.ind_level=="S",'industry'].str.ljust(6,'-')
    qwi['industry'] = qwi['industry'].str.ljust(6,"/")
    qwi.loc[qwi.ind_level=="A","ind_level"]="-1"
    qwi.loc[qwi.ind_level == "S", "ind_level"] = "2"
    qwi['agglvl_code']=72+qwi.ind_level.astype(int)
    qwi.loc[qwi.ind_level=="-1",'ind_level']="0"

    # make unique indentifier
    qwi['geoindkey'] = qwi['geography']+"_"+qwi['industry']
    qwi['state']=qwi['geography'].astype(str).str.slice(start=0,stop=-3)
    state_abbr = {
        "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca", "08": "co", "09": "ct", "10": "de",
        "11": "dc", "12": "fl", "13": "ga", "15": "hi", "16": "id", "17": "il", "18": "in", "19": "ia", "20": "ks",
        "21": "ky", "22": "la", "23": "me", "24": "md", "25": "ma", "26": "mi", "27": "mn", "28": "ms", "29": "mo",
        "30": "mt", "31": "ne", "32": "nv", "33": "nh", "34": "nj", "35": "nm", "36": "ny", "37": "nc", "38": "nd",
        "39": "oh", "40": "ok", "41": "or", "42": "pa", "44": "ri", "45": "sc", "46": "sd", "47": "tn", "48": "tx",
        "49": "ut", "50": "vt", "51": "va", "53": "wa", "54": "wv", "55": "wi", "56": "wy"
    }
    states = state_abbr.keys()
    if "STATES" in generalConfig and generalConfig['STATES'] is not None:
            if "ALL" not in generalConfig["STATES"]:
                states = generalConfig["STATES"]
                if states[0] not in state_abbr.keys():
                    states=[str(int(key)) for key,value in state_abbr.items() if value.upper() in states]
                qwi=qwi.loc[qwi['state'].isin(states),:].copy()
    #if forcombine:
    #    qwi.drop(columns=['industry','geography'],axis=1,inplace=True)
    return(qwi)

## Loop through qwi files with 'co' in file name
##  (my file names have been manually changes to reflect "co" for county level, and "st" for state level
# INPUT: str which is the path to the folder including QWI data files
# OUTPUT: pd.dataframe with all qwi county files (labeled with "co" in file name) preprocessed and combined.
def read_qwi_co(folderpath,generalConfig):
    lsdf = [] #initialize list

    fileexists=False
    #for files in the specified folder path
    for file in os.listdir(folderpath):
       # if the file name includes "co" in it, read data, preprocess it, add it to the list
        if "co" in file:
            fileexists=True
            df = pd.read_csv(folderpath+str(file))
            df = preprocess_qwi(df,generalConfig)
            dfnew = df.loc[df['geo_level']=="C"] #make sure this is just county data
            lsdf.append(dfnew)
    if not fileexists:
        print("No county files with 'co' in the name in QWI folder.")
    qwidf = pd.concat(lsdf,axis=0,ignore_index=True) #combine all of these
    qwidf.rename({"quarter":"qtr", "county":"cnty","Emp":"emp1_qwi","EmpEnd":"emp3_qwi","sEmp":"semp1_qwi","sEmpEnd":"semp3_qwi"},
                 axis=1, inplace=True,errors="ignore")
    qwidf.drop_duplicates(inplace=True)
    return(qwidf)

## Combine QWi and CBP Data
# INPUTS:
#       rawfile: str for file path of raw cbp county data
#       imputedfile: str for file path of imputed cbp data
#       qwifolder: str for folder path of folder that includes qwi files with county-level names including "co"
#       printdiagnostics: logical, if True print shapes of CBP and QWI data and count row in imputed CBP not in QWI
# OUTPUTS: pd.dataframe of combined CBP and QWI
def combine_qwi_cbp_qcew(rawfile, imputedfile,qwifolder,generalConfig,preprocessConfig, outfilename="combineFull.csv",diagnosticsfile=None,outfilepath="PythonPreprocessOut",year=2016,notsuppressedqwi=1):
    state_abbr = {
        "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca", "08": "co", "09": "ct", "10": "de",
        "11": "dc", "12": "fl", "13": "ga", "15": "hi", "16": "id", "17": "il", "18": "in", "19": "ia", "20": "ks",
        "21": "ky", "22": "la", "23": "me", "24": "md", "25": "ma", "26": "mi", "27": "mn", "28": "ms", "29": "mo",
        "30": "mt", "31": "ne", "32": "nv", "33": "nh", "34": "nj", "35": "nm", "36": "ny", "37": "nc", "38": "nd",
        "39": "oh", "40": "ok", "41": "or", "42": "pa", "44": "ri", "45": "sc", "46": "sd", "47": "tn", "48": "tx",
        "49": "ut", "50": "vt", "51": "va", "53": "wa", "54": "wv", "55": "wi", "56": "wy"
    }
    allowedstates=state_abbr.keys()
    if "STATES" in generalConfig:
        if generalConfig["STATES"] is None or "ALL" in generalConfig['STATES']:
            pass
        else:
            fipscodes_df=pd.read_csv(generalConfig['FIPS_STATE_FILE'])
            states=generalConfig["STATES"]
            fipscode = fipscodes_df.iloc[:, 0].astype(str).str.upper()  # standardize format col1
            stateselect = (fipscode.isin(states))  # initialize the logical to select states
            if len(fipscodes_df.columns) > 1:  # if there are more than 1 column
                for colname in fipscodes_df.columns:  # iterate columns to create logical
                    fipscode = fipscodes_df[colname].astype(str).str.upper()
                    stateselect = (stateselect) | (fipscode.isin(states))
            subfips=fipscodes_df[stateselect]
            allowedstates = subfips.iloc[:,0].values  # subset
    onlyraw=False
    if diagnosticsfile is not None:
        printdiagnostics=True
    qwi = read_qwi_co(qwifolder,generalConfig)  # read all qwi county files
    qwi=qwi.loc[qwi['state'].isin(allowedstates),:]

    #print(qwi.columns)
    flagCols = ["semp1_qwi", "semp3_qwi", "sEmpS", "sEarnBeg"]
    flagcodes = None
    for fcol in flagCols:
        setcodes = set(qwi[fcol].unique())
        if flagcodes is None:
            flagcodes = setcodes
        else:
            flagcodes = flagcodes.union(setcodes)
    if ~isinstance(notsuppressedqwi, list):
        notsuppressedqwi = [notsuppressedqwi]
    suppressedqwi = list(flagcodes - set(notsuppressedqwi))
    if not isinstance(suppressedqwi,list):
        suppressedqwi=[suppressedqwi]
    iter = 0
    for fcol in flagCols:
        iter += 1
        notsup = qwi[fcol].isin(notsuppressedqwi)
        qwi.loc[~notsup, fcol[1:]] = np.nan
        if printdiagnostics:
            fcolname = fcol.replace("_qwi", "")
            fcolname = fcolname[1:]
            tempxtab = pd.crosstab(qwi['agglvl_code'], qwi[fcol]).join(
                pd.crosstab(qwi['agglvl_code'], qwi[fcol], normalize="index"),
                on="agglvl_code", how='outer', lsuffix='_count', rsuffix='_prop', sort=True)
            if len(suppressedqwi) == 1:
                countsup = tempxtab[str(suppressedqwi[0])+"_count"]
                pctsup = tempxtab[str(suppressedqwi[0])+"_prop"]
            else:
                countsup = tempxtab[[str(x)+"_count" for x in suppressedqwi]].sum(axis=1)
                pctsup = countsup.div(tempxtab[[str(x)+"_count" for x in suppressedqwi]].sum(axis=1))
            if iter == 1:
                xtab = pd.DataFrame({'n': tempxtab[[str(x)+"_count" for x in suppressedqwi]].sum(axis=1),
                                     "" + fcolname: countsup,
                                     "pct_" + fcolname: pctsup.multiply(100).round(0)},
                                    index=tempxtab.index)
                xtabQWI = xtab
            else:
                xtab = pd.DataFrame(
                    {"" + fcolname: countsup,
                     "pct_" + fcolname: pctsup.multiply(100).round(0)},
                    index=tempxtab.index)
                xtabQWI = xtabQWI.join(xtab, on="agglvl_code", how="outer", lsuffix="", rsuffix="_" + fcolname)

    if not os.path.isfile(rawfile):
        print(f"Cannot locate file named {rawfile} ")
    else:
        raw = pd.read_table(rawfile, sep=",")  # read raw CBP file for counties in 50 states

    if not os.path.isfile(imputedfile):
        if float(year)<2017:
            print(f"Cannot locate file named {imputedfile} ")
        else:
            imputeCBP=None
            onlyraw=True
    else:
        onlyraw=False
        imputeCBP = preprocess_imputedCBP_file(imputedfile)  # read imputed data file (only includes Mid March Employment)
    if not os.path.exists(qwifolder):
        print(f"Cannot locate directory named {qwifolder} ")

    # combine imputted and raw
    if printdiagnostics:
        cbp, supptabCBP = combine_CBP_raw_imputed(raw,imputeCBP,states=qwi['state'].unique(),onlyraw=onlyraw,supptab=printdiagnostics)
    else:
        cbp = combine_CBP_raw_imputed(raw,imputeCBP,onlyraw=onlyraw)

    #cbp=cbp.loc[cbp['state'].isin(allowedstates),:]
    if generalConfig['QTR']==4:
        cbp['year_qtr_cbp']=float(generalConfig['YEAR'])+1.25
    else:
        cbp['year_qtr_cbp'] = float(generalConfig['YEAR']) + 0.25



    #combine QWI and CBP
    cbp["state"] = cbp["state"].astype(int).astype(str)
    cbp["cnty"] = cbp["cnty"].astype(int).astype(str).str.zfill(3)
    #cbp["ind_level"] = cbp["ind_level"].astype(int).astype(str)
    qwi["state"] = qwi["state"].astype(int).astype(str)
    qwi["cnty"] = qwi["cnty"].astype(int).astype(str).str.zfill(3)
    qwi["ind_level"] = qwi["ind_level"].astype(int).astype(str)
    combinedf = cbp.merge(qwi,how="outer",on=np.intersect1d(np.array(qwi.columns.values),np.array(cbp.columns.values)).tolist(),indicator=True,suffixes=["_cbp","_qwi"],validate="one_to_one")

    combinedf.reset_index(drop=True,inplace=True)
    cbp.reset_index(drop=True,inplace=True)

    cbpidx=cbp['agglvl_code'].isin(qwi['agglvl_code'].unique())#pd.Series([False]*len(cbp))
    combineidx=combinedf['agglvl_code'].isin(qwi['agglvl_code'])

    qwilevels=qwi['agglvl_code'].astype(str).unique()
    restrictions_str="Restricted to common aggregate levels ("+', '.join(qwilevels)+")"

    misscount2 = sum(combinedf.loc[combineidx,'_merge'] == "left_only")
    misscount1 = sum(combinedf['_merge'] == "right_only")
    misscount4 = sum(combinedf['_merge_rawimpute'] == "raw_only")
    #possible diagnostics
    if not os.path.exists(outfilepath):
        os.mkdir(outfilepath)
    if printdiagnostics:
        cbp=cbp.reset_index(drop=True)
        misscount2_prop=0
        misscount1_prop=0
        if misscount2>0 and cbp[cbpidx].shape[0]>0:
            misscount2_prop=round((misscount2/cbp[cbpidx].shape[0])*100)
        if misscount1>0 and qwi.shape[0]>0:
            misscount1_prop=round((misscount1/qwi.shape[0])*100)
        with open(outfilepath+diagnosticsfile,'w') as f:
            os.makedirs(os.path.dirname(outfilepath), exist_ok=True)
            print("Number of rows,columns in combined CBP: ",cbp.shape,file=f)
            print("Number of rows,columns in QWI: ", qwi.shape, file=f)
            print("Number of rows in CBP but not QWI "+restrictions_str+": "+str(misscount2)+"("+str(misscount2_prop)+"%)", file=f)
            print("Number of rows in QWI but not combined CBP: "+str(misscount1)+"("+str(misscount1_prop)+"%)", file=f)
            print("Number of rows in raw CBP but not imputed CBP: ", misscount4, file=f)
            print("--"*20,file=f)
            print("----- CBP Suppression Table -----",file=f)
            print("--"*20,file=f)
        supptabCBP.to_csv(outfilepath+diagnosticsfile,sep=",",mode='a',index=False)
        with open(outfilepath + diagnosticsfile, 'a') as f:
            print("--" * 20, file=f)
            print("----- QWI Suppression Table -----", file=f)
            print("--" * 20, file=f)
        xtabQWI.to_csv(outfilepath+diagnosticsfile,sep=",",mode="a")

    combinedf = combinedf.drop(columns=['_merge','_merge_rawimpute'])
    #combinedf=fill_from_geoindkey(combinedf)
    combinedf['year']=generalConfig["YEAR"]

    #fill when lb and ub
    if "lb" in combinedf.columns and combinedf['emp3_cbp'].isna().sum()>0:
        combinedf.loc[combinedf['emp3_cbp'].isna(),"emp3_cbp"]=combinedf.loc[combinedf['emp3_cbp'].isna(),:].apply(random_midpoint,axis=1)

    #rename columns
    combinedf.rename(columns={"EmpS": "lwbd_emp_qwi",
                              "sEmpS": "lwbd_emp_qwi_flag",
                              "semp1_qwi": "emp1_qwi_flag",
                              "semp3_qwi":"emp3_qwi_flag",
                              "EarnBeg":"avg_month_emp_wage",
                              "sEarnBeg":"avg_month_emp_wage_flag"}, inplace=True)
    if "QCEWDIR" not in preprocessConfig or preprocessConfig["QCEWDIR"] is None:
        combinedf = fill_from_geoindkey(combinedf)
        combinedf=combinedf[combinedf['estnum'].notna()]
        combinedf['cnty'] = combinedf['cnty'].astype("str")
        combinedf['ind_level'] = combinedf['ind_level'].astype("str")
        if outfilepath in outfilename:
            combinedf.to_csv(outfilename)
        else:
            combinedf.to_csv(outfilepath + outfilename)
        fullcombine=combinedf.copy()

    else:
        combinedf.rename(columns={"estnum": "estnum_cbp"}, inplace=True)
        combinedf['cnty'] = combinedf['cnty'].astype("str")
        combinedf['ind_level'] = combinedf['ind_level'].astype("str")
        qcew = pd.read_csv(preprocessConfig['DATA_IN_FOLDER']+preprocessConfig["QCEWDIR"] + "qcew_part.csv")
        fullcombine = preprocess_qcew(qcew, combinedf, generalConfig, preprocessConfig)
        if outfilepath in outfilename:
            fullcombine.to_csv(outfilename)
        else:
            fullcombine.to_csv(outfilepath + outfilename)
    return(fullcombine)

def qcew_format_geoindkey(data_row): #for QCEW
    naics_code=str(data_row['industry_code'])
    if data_row['agglvl_code']==71 or data_row['agglvl_code']==51:
        naics_code="------"
    elif data_row['agglvl_code']==54 or data_row['agglvl_code']==74:
        naics_code=str(naics_code).ljust(6,"-")
    else:
        naics_code=str(naics_code).ljust(6,'/')
    data_row['geoindkey']=str(data_row['area_fips'])+"_"+str(naics_code)
    return data_row

def preprocess_qcew(data,combine, generalConfig, preprocessConfig,remove_xtra_agglvl=True):
    if remove_xtra_agglvl:
        keepagglvls=combine['agglvl_code'].unique()
        print("Only keeping QCEW aggregate level codes in CBP/QWI combined data :"+', '.join([str(int(x)) for x in keepagglvls]))
        data=data[data['agglvl_code'].astype(float).isin(keepagglvls)]
    data=data.apply(qcew_format_geoindkey,axis=1)
    #prepare combine data for merging
    #combine.drop(columns=['year','qtr'],inplace=True)
    #prepare qcew for merging
    print("Combining QCEW Data with Other Sources...")
    data.drop(columns=['own_code'],inplace=True)
    data=data.loc[data['qtrly_estabs']>0,:]
    data.rename(columns={"month1_emplvl": "emp1_qcew","month2_emplvl": "emp2_qcew","month3_emplvl": "emp3_qcew",
                              "total_qtrly_wages": "wages_qcew",
                              "qtrly_estabs": "estnum_qcew"}, inplace=True)
    data=fill_from_geoindkey(data,numeric_ind_level=True)
    #combine['cnty']=combine['cnty'].astype("str")
    #combine['state'] = combine['state'].astype("str")
    #combine['geography'] = combine['geography'].astype("str")
    geoindkey_inter=set(np.intersect1d(np.array(data['geoindkey'].str.slice(start=0,stop=5)), np.array(combine['geoindkey'].str.slice(start=0,stop=5))).tolist())
    combine.drop(columns={'ind_level','industry','state','cnty','geography','agglvl_code'},inplace=True)

    colscomb = np.intersect1d(np.array(data.columns.values), np.array(combine.columns.values)).tolist()
    colscomb=[x for x in colscomb if x in ["year","qtr","geoindkey"]]
    combine.loc[combine['qtr'].isna(),'qtr']=generalConfig['QTR']

    #print(combine.dtypes)
    #print(data.dtypes)
    melddf = data.merge(combine, how="outer",
                          on=colscomb,
                          indicator=True, suffixes=["_qcew", "_other"], validate="one_to_one")
    melddf.rename(columns={"estnum": "estnum_cbp"}, inplace=True)
    melddf=fill_from_geoindkey(melddf,numeric_ind_level=True)

    melddf["_merge"] = melddf["_merge"].cat.rename_categories(
        {'right_only': 'cbp_qwi_only', 'left_only': 'qcew_only', "both": "both"})
    #melddf=data.merge(combine,how="outer",on=["geoindkey","geoindkey"],indicator=True,suffixes=["_combine","_qcew"],validate="one_to_one")
    if preprocessConfig['DIAGNOSTIC_FILE'] is not None:
        print("Adding diagnostic information to "+preprocessConfig["DIAGNOSTIC_FILE"])
        temp=melddf.copy()
        temp['cat']=temp['_merge'].astype(str)
        temp.loc[(temp['cat']=="qcew_only")&(temp['industry'].isin(excluded_cbp)),'cat']="qcew_only_excluded_cbp"
        xtabs=pd.crosstab(temp["agglvl_code"], temp["cat"])
        with open(preprocessConfig['OUTPATH'] + preprocessConfig["DIAGNOSTIC_FILE"], 'a') as f:
            print("--" * 20, file=f)
            print("----- Merging QCEW Data with Combined CBP and QWI Tables -----", file=f)
            print("--" * 20, file=f)
            print("Note the following NAICS codes are not included in CBP data: "+", ".join(excluded_cbp))
        xtabs.to_csv(preprocessConfig['OUTPATH'] + preprocessConfig["DIAGNOSTIC_FILE"],sep=",",mode="a")
    melddf.drop(columns=["area_fips","industry_code","geo_level","ind_level",'_merge'],inplace=True)
    return(melddf)

def quarter_adjustment(data,generalConfig,response,quarterConfig=None,formula=None):
        '''
        What is the point?
            get_m1emp_model() creates an OLS model that predicts employment values ('Emp') based on various
            predictors. This model is used when direct employment data is missing.
        Steps:
            1. Filters input data to include only rows where
                - 'sEmpEnd' is not suppressed
                - 'sEmp' is not supressed
                - 'ind_level' is not "A"
            2. Initial model fitting
                - Use formula specified in config.yaml (default: 'Emp ~ EmpEnd + estnum + C(sector) + C(state)')
                  to construct the design matrix to fit an OLS model. (model_pre).
            3. Influential point/Outlier detection
                - Compute Cook's distance for each observation and filter out observations where Cook's
                  Distance exceeds the threshold set in config.yaml. (default: 1)
                  Compute Studentized Residuals for each observation and filter out observations where they
                  exceed the threshold set in config.yaml
            4. Refit model after removing influential points
        Configurable Parameters:
            The regression formula and Cook's disitance thresholds are both configurable via config.yaml
            under employmentConfig
        Returns:
            1. model  -  (statsmodel.OLS)
                - Used with custom_predict in get_m1emp() to predict month 1 employment counts
            2. Prints a message if any influential points are removed.
                - Helpful Diagnostic
        '''
        # Step 1
        # get difference of quarters
        tempdata=data.copy()
        tempdata['year_qtr']=tempdata['year']+tempdata['qtr'].astype(float).multiply(0.25)
        tempdata['year_qtr_diff'] = tempdata['year_qtr_cbp'].astype(float) - tempdata['year_qtr'].astype(float)

        if tempdata['year_qtr'].nunique()>1:
            formula_stem=response+"~year_qtr_diff+qtr*naics2+"
        else:
            formula_stem=response+"~"

        # Retrieve OLS formula from config.yaml if quarterConfig exists
        if quarterConfig is not None and formula is None:
            if response=="wages_qcew":
                formula=quarterConfig['WAGE_OLS_FORMULA']
            else:
                formula=quarterConfig['EMP_OLS_FORMULA']
        elif formula is None:
            formula=formula_stem+"wages_cbp*wages_cbp_flag+np.log(estnum_cbp)+np.log(estnum)+emp3_cbp+emp3_cbp_flag+agglvl_code+agglvl_code*naics2"

        for vname in ["year","wages_cbp","estnum_cbp","estnum","wages_qcew","emp1_qcew","emp2_qcew","emp3_qcew","emp3_cbp"]:
            tempdata[vname]=tempdata[vname].astype(float)
        for vname in ['qtr','qtr_cbp','wages_cbp_flag',"emp3_cbp_flag","agglvl_code","naics2","naics3","naics4","naics5"]:
            tempdata[vname]=tempdata[vname].astype("category")
        subdata = tempdata[
            (~tempdata['wages_cbp'].isna()) & (~tempdata['wages_qcew'].isna()) & (~tempdata['emp3_cbp'].isna())].copy()

# Create design matrices (gets the variables ready for fitting in statsmodels.OLS) using the formula
        # and perform initial model fitting
        y_pre, X_pre = Formula(formula).get_model_matrix(subdata)
        model = sm.OLS(y_pre, X_pre).fit()
        # Calculate Cook's distance for each observation
        #influence = OLSInfluence(model_pre)
        #cooks_d = influence.cooks_distance[0]
        #student_resid = influence.resid_studentized_internal
        if quarterConfig is not None and quarterConfig['DIAGNOSTIC_PLOTS'] is not None:
            save_diagnostic_plots(model, formula, quarterConfig['DIAGNOSTIC_PLOTS'])
        print("Model to adjust CBP "+response+" to quarter "+str(generalConfig['QTR']))
        print(model.summary())
        data.loc[(~data['wages_cbp'].isna()) & (~data['wages_qcew'].isna()) & (~data['emp3_cbp'].isna()),response+"_cbp"]=model.fittedvalues()
        return data






# # # test code
# # #
with open('config_pre2017.yaml', 'r') as configFile:
     config = yaml.safe_load(configFile)
preprocessConfig = config['preprocessConfig']
generalConfig = config['generalConfig']

foldername = preprocessConfig['DATA_IN_FOLDER']
#
#generalConfig["QCEWDIR"]=None
temp=combine_qwi_cbp_qcew(rawfile=foldername + preprocessConfig['CBPDATA'],
                   imputedfile=foldername + preprocessConfig['IMPUTECBP'],
                   qwifolder=foldername + preprocessConfig['QWIDIR'],
                     generalConfig=generalConfig,
                          preprocessConfig=preprocessConfig,
                   outfilename=generalConfig['COMBINED_DATA'],
                   diagnosticsfile=preprocessConfig["DIAGNOSTIC_FILE"],
                   outfilepath = preprocessConfig['OUTPATH'],
                year=generalConfig['YEAR'])

#print(temp.loc[:,].sort_values(by='geoindkey').head(20))
#print(temp.loc[:,].sort_values(by='geoindkey').tail(10))
print(temp.columns)
print(temp.head())
# # #
