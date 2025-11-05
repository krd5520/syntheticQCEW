import sys
import os



sys.path.append(os.path.abspath('./'))
from EmploymentFunctions import *
from WageFunctions import *
from NAICS6functions import *
from MicrodataPostprocessing import *
from hierarchy_geoindkey import *



def generate_NAICS6_byCounty(generalConfig, employmentConfig, wageConfig,df=None):
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
    # dfsave = df.copy()
    tofloatcols=['agglvl_code','estnum_qcew', 'emp1_qcew', 'emp2_qcew', 'emp3_qcew',
                 'wages_qcew','emp3_cbp', 'wages_cbp','estnum_cbp', 'year_qtr_cbp',
                 'emp1_qwi', 'emp3_qwi','lwbd_emp_qwi', 'avg_month_emp_wages', 'estnum',
                 'emp3','emp2',  'emp1', 'wages', 'year_qtr']
    for x in tofloatcols:
        if x in df.columns:
            df[x]=df[x].astype(float)

    df.drop(columns=["Unnamed: 0","estnum_qcew"],inplace=True,errors="ignore")


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

    #df['year'] = generalConfig['YEAR']
    #df['quarter'] = generalConfig['QTR']
    # Extract 6-digit NAICS
    df6 = df[df['geoindkey'].str.contains("_[0-9]{6}", regex=True)].copy()
    df6['geo4naics'] = df6['geoindkey'].str[:-2]
    df6['geo5naics'] = df6['geoindkey'].str[:-1]
    #df6 = df6[['geoindkey', 'geo4naics', 'geo5naics', 'state', 'cnty', 'estnum', 'wages_cbp', 'wages_cbp_flag', 'emp']]
    # Get summary counts for distribution
    count6dig_wages = get_codes_summary(df, groupbydigits=4, levelgrouped=6, variable="wages")
    count6dig_emp1 = get_codes_summary(df, groupbydigits=4, levelgrouped=6, variable="emp1")
    count6dig_emp2 = get_codes_summary(df, groupbydigits=4, levelgrouped=6, variable="emp2", include_source=False)
    count6dig_emp3 = get_codes_summary(df, groupbydigits=4, levelgrouped=6, variable="emp3")

    # Filter and prepare 4-digit NAICS data
    df4 = df[df['agglvl_code'] == 76].copy()
    df4['geo4naics'] = df4['geoindkey'].str.slice(stop=-2)
    df4['geo3naics'] = df4['geoindkey'].str.slice(stop=-3)
    #merge with summary data
    df4 = df4.merge(count6dig_wages, on=['geo4naics'], how='left',indicator=True)
    df4.rename(columns={"_merge":"_merge_wagesI"},inplace=True)
    df4 = df4.merge(count6dig_emp1, on=['geo4naics', "grouplevels", "count6by4codes"], how='left',indicator=True,suffixes=["_wagesI","_emp1I"])
    df4.rename(columns={"_merge": "_merge_emp1I"}, inplace=True)
    df4 = df4.merge(count6dig_emp2, on=['geo4naics', "grouplevels", "count6by4codes"], how='left',indicator=True,suffixes=["","_emp2I"])
    df4.rename(columns={"_merge": "_merge_emp2I"}, inplace=True)
    df4 = df4.merge(count6dig_emp3, on=['geo4naics', "grouplevels", "count6by4codes"], how='left',indicator=True,suffixes=["","_emp3I"])
    df4.rename(columns={"_merge": "_merge_emp3I"}, inplace=True)

    weirdones=df4.loc[df4["_merge_wagesI"]=="left_only"]
    geo6naicsweird=df6.loc[df6["geo4naics"].isin(weirdones['geo4naics']),:]
    if len(geo6naicsweird)==0:
        print("fixing it")
        for cname in ['wages','emp1','emp2','emp3']:
            df4.loc[df4[cname+"_sum6by4"].isna(), cname+"_sum6by4"] = 0
            df4.loc[df4[cname + "_missing6by4"].isna(), cname + "_missing6by4"] = 0
        df4.loc[df4['count6by4codes'].isna(), 'count6by4codes'] = 0
    else:
        for cname in ['emp1', 'emp2', 'emp3', 'wages']:
            print(f'count na in {cname}_sum6by4 {sum(df4.loc[:, cname + "_sum6by4"].isna())}')
            print(f'in {cname}_missing6by4 {sum(df4.loc[:, cname + "_missing6by4"].isna())}')
        raise Exception(f"Something wrong\n {geo6naicsweird.head()}")

    df4.drop(columns={"disclosure_code", "estnum_qcew", "year_qtr_cbp", "emp1_qwi_flag", "emp3_qwi_flag",
                      "grouplevels","_merge_wagesI","_merge_emp1I","_merge_emp2I","_merge_emp3I"}, inplace=True, errors="ignore")

    #df4.drop(columns=["_merge_wagesI","_merge_emp1I","_merge_emp2I","_merge_emp3I"],inplace=True,errors="ignore")
    #print(df6.loc[df6["geo4naics"].isin(weird4naics),:].head())

    df4['wagesdiff'] = df4['wages'].astype(float) - df4['wages_sum6by4'].astype(float)
    df4['emp1diff'] = df4['emp1'].astype(float) - df4['emp1_sum6by4'].astype(float)
    df4['emp2diff'] = df4['emp2'].astype(float) - df4['emp2_sum6by4'].astype(float)
    df4['emp3diff'] = df4['emp3'].astype(float) - df4['emp3_sum6by4'].astype(float)


    ############## Employment Counts ################
    print('---------- Employment Configuration ----------')
    # Display all current employmentConfig settings
    for key, value in employmentConfig.items():
        print(f"{key}: {value}")
    # Step 1: Create employment prediction model
    print('---------- Imputing Employment Data ----------')
    m1empfit = get_m1emp_model(df=df4,employmentConfig=employmentConfig)
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
    empMatA['m1emp']=adjustm1emp
    #=-empMatA.iloc[:, 1] = adjustm1emp  # Update with adjusted values
    #print(empMatA.head())
    empMatA.rename(columns={'m1emp':'emp1','m2emp':'emp2','m3emp':'emp3'},inplace=True)

    perwagemodel=df4.merge(empMatA,on=["geoindkey"],how="left",suffixes=["_wdf",""])
    perwagemodel.loc[(perwagemodel['emp2'].notna())&(perwagemodel['emp2_source'].isna()),'emp2_source']="noise_impute"
    perwagemodel.to_csv("DataDiag/PythonPreprocessOut/perwagemodel.csv")
    ## Sanity check on na's
    #for cname in ["emp3","emp3_source","emp3_wdf","emp2","emp2_source","emp1","emp1_source"]:
    #    if "source" in cname:
    #        print(perwagemodel[cname].value_counts(dropna=False))
    #    else:
    #        print(f'na count:{sum(perwagemodel[cname].isna())}, and not na count {sum(perwagemodel[cname].notna())}')

    if wageConfig is not None:
        ################## WAGES #######################
        print('---------- Wage Configuration ----------')
        # Display all current wageConfig settings
        for key, value in wageConfig.items():
            print(f"{key}: {value}")
        # Step 1. Create wage prediction model
        print('---------- Imputing Wage Data ----------')
        wagefit_sub = get_wages_model(df=perwagemodel, emp_mat_adj=empMatA,wageConfig=wageConfig)
        # Step 2. Get min/max bounds
    else:
        # Step 1. Create wage prediction model
        print('---------- Imputing Wage Data (if needed) ----------')
        wagefit_sub=None
    wages_maxmin = get_maxmindf(df4dig=df4, fulldf=df, emp_mat_adj=empMatA)
    # Prepare employment matrix
    empMatwage = pd.DataFrame({
        "emp1": empMatA.iloc[:, 1],
        "emp2": empMatA.iloc[:, 2],
        "emp3": empMatA.iloc[:, 3]
    })
    # Step 3: Impute wage values
    wagesout = get_wages4(df4=df4, empmat=empMatwage, wagemodel=wagefit_sub, useEarnQWI=True, maxmindf=wages_maxmin)
    # Prepare final 4 digit output
    df4imp = wagesout.copy().assign(
        #EmpScale=lambda x: np.where(x['m3emp'] == 0, 1, x['emp'] / x['m3emp'].astype(float)),
        geo4naics=lambda x: x['geoindkey'].str[:-2],
        m1empFromModel=empMatA.iloc[:, 4].astype(float)  # 5th column (0-based index 4)
    )

    ################## Get NAICS6 By County Aggregates #######################
    print('---------- Getting NAICS6 by County Aggregates ----------')
    print('This may take a while, please be patient...')
    # Distribute values from 4-digit to 6-digit NAICS
    naics6df = get_6naics_all(df=df6, df4n=df4imp, codes4summary=count6dig)
    # Final formatting and output
    naics6df = naics6df.iloc[:, :5].join(naics6df[['m1emp', 'm3emp', 'wages']])
    naics6df.head(50)
    naics6df.to_csv(str(generalConfig["NAICS6_PATH"]),index=False)
    return(naics6df)


