import sys
import os

import pandas as pd

sys.path.append(os.path.abspath('./'))
from EmploymentFunctions import *
from WageFunctions import *
from NAICS6functions import *
from MicrodataPostprocessing import *
from hierarchy_geoindkey import *
from adjustmentFunctions import  *

use_QWI_for_wages=True

def generate_NAICS6_byCounty(generalConfig, employmentConfig, wageConfig,supplementaryConfig=None,df=None,naicsdf=None):
    ##################  Dataframes set up  #######################
    # Load main dataset. Location is set in the config.yaml
    # Tell the user where the dataset is located

    #reading in data if necessary
    np.random.seed(generalConfig["SEED"])
    if df is None:
        print('---------- Loading Dataset ----------\n')
        print(f"Dataset location: {generalConfig['COMBINED_DATASET']}")
        df = pd.read_csv(generalConfig['COMBINED_DATASET'], dtype=str)  # , nrows=100000)

        df = df.iloc[:, 1:]  # Remove index
    fulldf=df.copy()


    # In 2016 this NAICS 4 has no corresponding NAICS 6
    ## Later we will automate this??
    #if "29189_525990" not in df["geoindkey"]:
    #    df = pd.concat([df, pd.DataFrame([{
    #        "state": "29", "cnty": "198", "emp": "1", "geoindkey": "29189_525990",
    #        "wages_cbp_flag": "Impute", "wages_cbp": "0", "estnum": "2", "geo_level": "C",
    #        "geography": "29189", "ind_level": "6", "industry": "525990",
    #        "avg_month_emp_wages": None, "EarnHirAS": None, "Emp": None, "emp3_qwi": None, "lwbd_emp_qwi": None,
    #        "avg_month_emp_wages_flag": None, "sEarnHirAS": None, "sEmp": None, "emp3_qwi_flag": None, "lwbd_emp_qwi_flag": None
    #    }])], ignore_index=True)
    #    df.loc[df['geoindkey'].str.contains("29189_5259//", regex=True), 'wages_cbp_flag'] = "Impute"


    # Extract 6-digit NAICS
    df6 = df[df['geoindkey'].str.contains("_[0-9]{6}", regex=True)].copy()
    df6['geo4naics'] = df6['geoindkey'].str.slice(stop=-2)
    df6['geo5naics'] = df6['geoindkey'].str.slice(stop=-1)
    df6['geo3naics'] = df6['geoindkey'].str.slice(stop=-3)

    df4 = df[df['agglvl_code'] == 76].copy()
    df4['geo4naics'] = df4['geoindkey'].str.slice(stop=-2)
    df4['geo3naics'] = df4['geoindkey'].str.slice(stop=-3)


    #df4,df6,df=adjust_negative_diff(df4=df4,df=df)
    print(f"Num na in emp1 at countyXnaics6 level {df6['emp1'].isna().sum()}")
    #get countyXnaics6 within countyXnaics4 summary information
    for vname in ["emp1", "wages"]:
        count6dig = get_codes_summary(df, groupbydigits=4, levelgrouped=6, variable=vname)
        df4 = df4.merge(count6dig, on=['geo4naics'], how='left', indicator=True, suffixes=["", "_droplater"])
        weirdones = df4.loc[df4["_merge"] == "left_only"] #only appear in county by NAICS-4 codes
        geo6naicsweird = df6.loc[df6["geo4naics"].isin(weirdones['geo4naics']), :]
        if len(geo6naicsweird) == 0: ##If no county by NAICS6 codes, hard code the relevant values
            df4.loc[df4[vname + "_sum6by4"].isna(), vname + "_sum6by4"] = 0
            df4.loc[df4[vname + "_missing6by4"].isna(), vname + "_missing6by4"] = 0
            df4.loc[df4[vname + "_propmissing6by4"].isna(), vname + "_propmissing6by4"] = 1
            df4.loc[df4['count6by4codes'].isna(), 'count6by4codes'] = 0
        else: #otherwise print diagnostic information
            print(f'count na in {vname}_sum6by4 {sum(df4.loc[:, vname + "_sum6by4"].isna())}')
            print(f'in {vname}_missing6by4 {sum(df4.loc[:, vname + "_missing6by4"].isna())}')
            raise Exception(f"Something wrong\n {geo6naicsweird.head()}")
        df4.drop(columns=["_merge"], inplace=True)
        df4.drop(columns=[dropcol for dropcol in df4.columns if "_droplater" in dropcol], errors="ignore",inplace=True)

        ## Get difference between county by NAICS4 and sum of known county by NAICS6
        df4[vname + "diff"] = df4[vname].astype(float) - df4[vname + '_sum6by4'].astype(float)



    if supplementaryConfig is not None:
        print('---------- Supplementary Data Configuration ----------')
        # Display all current employmentConfig settings
        for key, value in supplementaryConfig.items():
            print(f"{key}: {value}")

        del key, value
        if 'COUNTY_DATA' in supplementaryConfig:
            if supplementaryConfig['COUNTY_DATA_SEP']=="tab" or supplementaryConfig['COUNTY_DATA_SEP']=="\t":
                cntydata=pd.read_csv(supplementaryConfig['DATA_IN_FOLDER']+supplementaryConfig["COUNTY_DATA"],sep='\t')
            else:
                cntydata = pd.read_csv(supplementaryConfig['DATA_IN_FOLDER'] + supplementaryConfig["COUNTY_DATA"])
            cntydata[supplementaryConfig['COUNTY_FIPS_COLNAME']]=pd.to_numeric(cntydata[supplementaryConfig["COUNTY_FIPS_COLNAME"]],errors="coerce").round(0).astype(int).astype(str)
            df4['geography']=df4['geoindkey'].str.extract(r'^([^_]+)')#split['_'].str[0]
            df4=df4.merge(cntydata,right_on=supplementaryConfig['COUNTY_FIPS_COLNAME'],left_on="geography",how="left")
        if 'STATE_DATA' in supplementaryConfig:
            if supplementaryConfig['STATE_DATA_SEP']=="tab" or supplementaryConfig['STATE_DATA_SEP']=="\t":
                stdata=pd.read_csv(supplementaryConfig['DATA_IN_FOLDER']+supplementaryConfig["STATE_DATA"],sep='\t')
            else:
                stdata = pd.read_csv(supplementaryConfig['DATA_IN_FOLDER'] + supplementaryConfig["STATE_DATA"])
            stdata[supplementaryConfig['STATE_FIPS_COLNAME']]=pd.to_numeric(cntydata[supplementaryConfig["STATE_FIPS_COLNAME"]],errors="coerce").round(0).astype(int).astype(str)
            df4['state']=df4['geoindkey'].str.extract(r'^([^_]+)').str.slice(stop=-3)#split['_'].str[0]
            df4=df4.merge(cntydata,right_on=supplementaryConfig['STATE_FIPS_COLNAME'],left_on="state",how="left")
        if 'INDUSTRY_DATA' in supplementaryConfig:
            print(f'Current algorithm does not support industry supplementary data.\n The input {supplementaryConfig["INDUSTRY_DATA"]} will be ignored.')


    ############## Employment Counts ################
    print('---------- Employment Configuration ----------')
    # Display all current employmentConfig settings
    for key, value in employmentConfig.items():
        print(f"{key}: {value}")

    del key, value
    #print("inside naics6 function 2")
    #print(pd.crosstab(df4["emp1_source"],df4["emp2_source"],dropna=False))

    # Step 1: Create employment prediction model
    print('---------- Imputing Employment Data ----------')
    negdiff=(df4["emp1diff"] < 0)
    df4.loc[(negdiff) & (df4["emp1_missing6by4"] > 0), "emp1"] = np.nan
    df4.loc[(negdiff) & (df4["emp1_missing6by4"] > 0), "emp1_source"] = np.nan
    df4.loc[(negdiff) & (df4["emp1_missing6by4"] > 0), "emp1diff"] = np.nan
    df4.loc[(negdiff) & (df4["emp1_missing6by4"] == 0), "emp1diff"] = 0
    df4.loc[(negdiff) & (df4["emp1_missing6by4"] == 0), "emp1"] = df4.loc[(negdiff) & (df4["emp1_missing6by4"] == 0), "emp1_sum6by4"]
    df4.loc[(negdiff) & (df4["emp1_missing6by4"] == 0), "emp1_source"] = "sum6by4"

    m1empfit = get_m1emp_model(df=df4,employmentConfig=employmentConfig)
    #print(pd.crosstab(df4["emp1_source"], df4["emp2_source"], dropna=False))

    # Step 2: Generate monthly employment counts
    empMat, df4 = get_employmentCounts4(
        df4,
        m1emp_model=m1empfit,
        m2emp_noisecoef=employmentConfig['M2EMP_NOISECOEF'],
        rseed=employmentConfig['RSEED'],
        include_m1emp_indicator=True
    )

    #count6digemp1 = get_codes_summary(df, groupbydigits=4, levelgrouped=6, variable="emp1")
    #checkdf4 = df4.merge(count6digemp1, on=['geo4naics'], how='left', indicator=False, suffixes=["", "_droplater"])
    #checkdf4.drop(columns=[dropcol for dropcol in checkdf4.columns if "_droplater" in dropcol], errors="ignore",inplace=True)
    #
    ### Get difference between county by NAICS4 and sum of known county by NAICS6
    #checkdf4["emp1diff"] = checkdf4["emp1"].astype(float) - checkdf4['emp1_sum6by4'].astype(float)
    #print("check df4 describe (pre adjust)")
    #print(checkdf4[['emp1diff','emp1','emp1_sum6by4','emp1_missing6by4','emp1_propmissing6by4','count6by4codes']].describe())

    # Step 3: Adjust to match county totals
    if "qcew" in df4['emp1_source'].unique():
        adjust_onlyqcew=True
    else:
        adjust_onlyqcew = False
    adjustdf = adjust_geo4naics_varvalues(fitdf=df4, dfmaxmin=None, stabvals=df4['lwbd_emp_qwi'], variable="emp1",fulldf=fulldf,onlyqcew=adjust_onlyqcew,minonly=True)
    adjustdf = adjust_geo4naics_varvalues(fitdf=adjustdf, dfmaxmin=None, stabvals=df4['lwbd_emp_qwi'], variable="emp2",fulldf=fulldf,onlyqcew=adjust_onlyqcew,minonly=True)
    adjustdf = adjust_geo4naics_varvalues(fitdf=adjustdf, dfmaxmin=None, stabvals=df4['lwbd_emp_qwi'], variable="emp3",fulldf=fulldf,onlyqcew=adjust_onlyqcew,minonly=True)
    adjustdf['m1empFromModel']=empMat['m1empFromModel']

    adjustdf = adjustdf.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.name in ['emp1','emp2','emp3','wages'] else col)

    #adjustm1emp = adjust_countytotal_qwi(valdf=adjustdf, sumdf=df[df["industry"] == "------"])
    empMatA = adjustdf[['geoindkey','emp1','emp2','emp3','emp1_source','emp2_source','emp3_source','m1empFromModel','minemp1','minemp1_source']].copy()
    #adjustdf=df4
    #empMatA = adjustdf[
    #    ['geoindkey', 'emp1', 'emp2', 'emp3', 'emp1_source', 'emp2_source', 'emp3_source', 'm1empFromModel']].copy()

    del empMat, adjustdf#, adjustm1emp

    prewagemodel=df4.merge(empMatA,on=["geoindkey"],how="left",suffixes=["_wdf",""])
    prewagemodel.loc[(prewagemodel['emp2'].notna())&(prewagemodel['emp2_source'].isna()),'emp2_source']="noise_impute"
    prewagemodel.loc[(prewagemodel['emp2'].notna())&(prewagemodel['emp2_source'].isna()),'emp2_source']="noise_impute"
    prewagemodel.to_csv("DataDiag/PythonPreprocessOut/prewagemodel.csv")
    prewagemodel.drop(columns=[dropcol for dropcol in prewagemodel.columns if "_wdf" in dropcol], errors="ignore", inplace=True)
    prewagemodel.drop(columns=["minemp1","minemp1_source"], errors="ignore", inplace=True)

    ## Sanity check on na's
    #for cname in ["emp3","emp3_source","emp3_wdf","emp2","emp2_source","emp1","emp1_source"]:
    #    if "source" in cname:
    #        print(prewagemodel[cname].value_counts(dropna=False))
    #    else:
    #        print(f'na count:{sum(prewagemodel[cname].isna())}, and not na count {sum(prewagemodel[cname].notna())}')
    count6dig = get_codes_summary(df, groupbydigits=4, levelgrouped=6, variable="wages")

    if wageConfig is not None:
        ################## WAGES #######################
        print('---------- Wage Configuration ----------')
        # Display all current wageConfig settings
        for key, value in wageConfig.items():
            print(f"{key}: {value}")
        # Step 1. Create wage prediction model
        print('---------- Imputing Wage Data ----------')
        #df4,df6=adjust_negative_diff_vname(prewagemodel,df6,df,vname="wages")
        negdiff = (prewagemodel["wagesdiff"] < 0)
        source_notqcew = (prewagemodel['wages_source'] != "qcew")

        ## When wagesdiff is negative...
        # CASE 1: if there are missing county by NAICS-6 cells, and the county by NAICS-4 wages source is not "qcew",
        # then override wages, wages_source, and wagesdiff to be NA

        prewagemodel.loc[(negdiff) & (prewagemodel["wages_missing6by4"] > 0) &(source_notqcew), "wages"] = np.nan
        prewagemodel.loc[(negdiff) & (prewagemodel["wages_missing6by4"] > 0)&(source_notqcew), "wages_source"] = np.nan
        prewagemodel.loc[(negdiff) & (prewagemodel["wages_missing6by4"] > 0)&(source_notqcew), "wagesdiff"] = np.nan
        # CASE 2: if there are NO missing county by NAICS-6 cells, and the county by NAICS-4 wages source is not "qcew",
        # then override wages=wages_sum6by4, wages_source='sum6by4', and wagesdiff=0
        prewagemodel.loc[(negdiff) & (prewagemodel["wages_missing6by4"] == 0)&(source_notqcew), "wagesdiff"] = 0
        prewagemodel.loc[(negdiff) & (prewagemodel["wages_missing6by4"] == 0)&(source_notqcew), "wages"] = prewagemodel.loc[
            (negdiff) & (prewagemodel["wages_missing6by4"] == 0)&(source_notqcew), "wages_sum6by4"]
        prewagemodel.loc[(negdiff) & (prewagemodel["wages_missing6by4"] == 0)&(source_notqcew), "wages_source"] = "sum6by4"
        # CASE 3: if there wages_source is qcew, must adjust county by NAICS-6 cbp values

        #count6digwages = get_codes_summary(df, groupbydigits=4, levelgrouped=6, variable="wages")

        #checkdf4 = prewagemodel.merge(count6digwages, on=['geo4naics'], how='left', indicator=False, suffixes=["_old",""])
        #checkdf4.drop(columns=[dropcol for dropcol in checkdf4.columns if "_droplater" in dropcol], errors="ignore",
        #              inplace=True)
        ### Get difference between county by NAICS4 and sum of known county by NAICS6
        #checkdf4["wagesdiff"] = checkdf4["wages"].astype(float) - checkdf4['wages_sum6by4'].astype(float)
        #print("check df4 describe (per wage fitting)")
        #print(checkdf4[['wages','wagesdiff',"wages_sum6by4","wages_missing6by4"]].describe())
        wagefit_sub = get_wages_model(df=prewagemodel, emp_mat_adj=empMatA,wageConfig=wageConfig)
        # Step 2. Get min/max bounds
    else:
        # Step 1. Create wage prediction model
        print('---------- Imputing Wage Data (if needed) ----------')
        wagefit_sub=None
    wages_maxmin=get_varmaxmindf(df4dig=df4, fulldf=df, variable="wages", onlyqcew=False)
    #print(f"when shape of data when min>max {wages_maxmin.loc[wages_maxmin['minwages']>wages_maxmin['maxwages'],:].shape}")
    #print(
    #    f"{wages_maxmin.loc[wages_maxmin['minwages'] > wages_maxmin['maxwages'], :].head()}")

    #print(f"maxwages below 0\n {wages_maxmin.loc[wages_maxmin['maxwages']<0,'maxwages_source'].value_counts()}")
    wages_maxmin.loc[wages_maxmin["maxwages"]<0,"maxwages"]=wages_maxmin['maxwages'].max()
    #wages_maxmin = get_maxmindf(df4dig=df4, fulldf=df, emp_mat_adj=empMatA)
    # Prepare employment matrix
    empMatwage = empMatA[["geoindkey","emp1","emp2","emp3"]]
    #pd.DataFrame({
    #    "geoindkey"
    #    "emp1": empMatA.loc[:, "emp1"],
    #    "emp2": empMatA.iloc[:, "emp2"],
    #    "emp3": empMatA.iloc[:, ]
    #})

    # Step 3: Impute wage values
    wagesout = get_wages4(df4=df4, wagemodel=wagefit_sub, useEarnQWI=use_QWI_for_wages, maxmindf=wages_maxmin,count6digdf=count6dig)
    print("here 3.5")
    # Prepare final 4 digit output
    df4imp = wagesout.copy().assign(
        #EmpScale=lambda x: np.where(x['m3emp'] == 0, 1, x['emp'] / x['m3emp'].astype(float)),
        geo4naics=lambda x: x['geoindkey'].str.slice(stop=-2),
        m1empFromModel=empMatA.loc[:, 'm1empFromModel'].astype(float)  # 5th column (0-based index 4)
    )


    df6 = df[df['geoindkey'].str.contains("_[0-9]{6}", regex=True)].copy()
    df6['geo4naics'] = df6['geoindkey'].str[:-2]
    df6['geo5naics'] = df6['geoindkey'].str[:-1]
    df6.loc[df6['wages_source']!="qcew","wages"]=np.nan
    count6dig = get_codes_summary(df, groupbydigits=4, levelgrouped=6, variable="estnum",onlyQCEW=False)

    #check agreement
    checkdf=prewagemodel.loc[prewagemodel["agglvl_code"]==76,:].merge(count6dig,on="geo4naics",how="left",indicator=False,suffixes=["_df4","_codesummary"])
    print(checkdf.columns)
    checkdf['estnumdiff']=checkdf['estnum']-checkdf['estnum_sum6by4']
    baddf=checkdf.loc[checkdf['estnumdiff']!=0,['geoindkey','estnum','estnum_sum6by4','count6by4codes_codesummary']]
    print(f"checking estnum aggreement between countyXnaics4 and naics6. {baddf.shape[0]} problem countyXnaics4 codes out of {checkdf.shape[0]}. \n{baddf.head()}")
    ################## Get NAICS6 By County Aggregates #######################
    print('---------- Getting NAICS6 by County Aggregates ----------')
    print('This may take a while, please be patient...')
    # Distribute values from 4-digit to 6-digit NAICS
    #print(f'df4imp head before get_6naics_all in generate_NAICS.py:\n{df4imp.head()}')
    naics6df = get_6naics_all(df6, df4imp, codes4summary=count6dig)
    #print(naics6df.columns)
    # Final formatting and output
    naics6df = naics6df.iloc[:, :5].join(naics6df[['emp1', 'emp3', 'wages']])
    naics6df.head(50)
    naics6df.to_csv(str(generalConfig["NAICS6_FILE"]),index=False)
    return(naics6df)


