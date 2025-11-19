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
    fulldf=df.copy()
    # dfsave = df.copy()
    #tofloatcols=['agglvl_code','estnum_qcew', 'emp1_qcew', 'emp2_qcew', 'emp3_qcew',
    #             'wages_qcew','emp3_cbp', 'wages_cbp','estnum_cbp', 'year_qtr_cbp',
    #             'emp1_qwi', 'emp3_qwi','lwbd_emp_qwi', 'avg_month_emp_wages', 'estnum',
    #             'emp3','emp2',  'emp1', 'wages', 'year_qtr']
    #for x in tofloatcols:
    #    if x in df.columns:
    #        df[x]=df[x].astype(float)
    #del x, tofloatcols

    #df.drop(columns=["Unnamed: 0","estnum_qcew"],inplace=True,errors="ignore")


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
    #df6['geo5naics'] = df6['geoindkey'].str[:-1]

    df4 = df[df['agglvl_code'] == 76].copy()
    df4['geo4naics'] = df4['geoindkey'].str.slice(stop=-2)
    df4['geo3naics'] = df4['geoindkey'].str.slice(stop=-3)
    #print("inside naics6 function")
    #print(pd.crosstab(df4["emp1_source"], df4["emp2_source"], dropna=False))

    for vname in ["emp1","wages"]:
        count6dig = get_codes_summary(df, groupbydigits=4, levelgrouped=6, variable=vname)
        df4 = df4.merge(count6dig, on=['geo4naics'], how='left', indicator=True,suffixes=["","_droplater"])
        weirdones = df4.loc[df4["_merge"] == "left_only"]
        geo6naicsweird = df6.loc[df6["geo4naics"].isin(weirdones['geo4naics']), :]
        if len(geo6naicsweird) == 0:
            df4.loc[df4[vname + "_sum6by4"].isna(), vname + "_sum6by4"] = 0
            df4.loc[df4[vname + "_missing6by4"].isna(), vname + "_missing6by4"] = 0
            df4.loc[df4[vname + "_propmissing6by4"].isna(), vname + "_propmissing6by4"] = 1
            df4.loc[df4['count6by4codes'].isna(), 'count6by4codes'] = 0
        else:
            print(f'count na in {vname}_sum6by4 {sum(df4.loc[:, vname + "_sum6by4"].isna())}')
            print(f'in {vname}_missing6by4 {sum(df4.loc[:, vname + "_missing6by4"].isna())}')
            raise Exception(f"Something wrong\n {geo6naicsweird.head()}")
        df4.drop(columns=["_merge"],inplace=True)
        df4.drop(columns=[dropcol for dropcol in df.columns if "_droplater" in dropcol],errors="ignore")
        df4[vname+"diff"]=df4[vname].astype(float) - df4[vname+'_sum6by4'].astype(float)
        #print(df4[[vname+"diff",vname+"_missing6by4",vname+"_propmissing6by4",vname+"_sum6by4",vname]].describe())
        #print(df4.loc[(df4[vname+"diff"]<0)&(df4[vname+"_missing6by4"]>0),[vname,vname+"_source",vname+"diff",vname+"_missing6by4",vname+"_propmissing6by4",vname+"_sum6by4"]].head())
        negdiff=df4.loc[
            (df4[vname + "diff"] < 0) & (df4[vname + "_missing6by4"] > 0), [vname, vname + "_source", vname + "diff",
                                                                            vname + "_missing6by4",
                                                                            vname + "_propmissing6by4",
                                                                            vname + "_sum6by4"]]

        print(negdiff.head())
        print(negdiff.shape)
        print(negdiff.describe())
    del weirdones,geo6naicsweird, df6
    #print(df4[[""]])

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

    # Step 3: Adjust to match county totals
    adjustdf = adjust_geo4naics_varvalues(fitdf=df4, dfmaxmin=None, stabvals=df4['lwbd_emp_qwi'], variable="emp1",fulldf=fulldf)
    adjustdf = adjust_geo4naics_varvalues(fitdf=adjustdf, dfmaxmin=None, stabvals=df4['lwbd_emp_qwi'], variable="emp2",fulldf=fulldf)
    adjustdf = adjust_geo4naics_varvalues(fitdf=adjustdf, dfmaxmin=None, stabvals=df4['lwbd_emp_qwi'], variable="emp3",fulldf=fulldf)
    adjustdf['m1empFromModel']=empMat['m1empFromModel']

    adjustdf = adjustdf.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.name in ['emp1','emp2','emp3','wages'] else col)

    #adjustm1emp = adjust_countytotal_qwi(valdf=adjustdf, sumdf=df[df["industry"] == "------"])
    empMatA = adjustdf[['geoindkey','emp1','emp2','emp3','emp1_source','emp2_source','emp3_source','m1empFromModel','minemp1','minemp1_source']].copy()

    #empMatA['emp1']=adjustm1emp
    del empMat, adjustdf#, adjustm1emp
    #=-empMatA.iloc[:, 1] = adjustm1emp  # Update with adjusted values
    #print(empMatA.head())
    #empMatA.rename(columns={'m1emp':'emp1','m2emp':'emp2','m3emp':'emp3'},inplace=True)

    perwagemodel=df4.merge(empMatA,on=["geoindkey"],how="left",suffixes=["_wdf",""])
    perwagemodel.loc[(perwagemodel['emp2'].notna())&(perwagemodel['emp2_source'].isna()),'emp2_source']="noise_impute"
    perwagemodel.loc[(perwagemodel['emp2'].notna())&(perwagemodel['emp2_source'].isna()),'emp2_source']="noise_impute"
    perwagemodel.to_csv("DataDiag/PythonPreprocessOut/perwagemodel.csv")
    ## Sanity check on na's
    #for cname in ["emp3","emp3_source","emp3_wdf","emp2","emp2_source","emp1","emp1_source"]:
    #    if "source" in cname:
    #        print(perwagemodel[cname].value_counts(dropna=False))
    #    else:
    #        print(f'na count:{sum(perwagemodel[cname].isna())}, and not na count {sum(perwagemodel[cname].notna())}')
    #stophere=False #used for checkpoint when testing the employment code
    #if stophere:
    #    raise Exception(f"stop here for check")
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
        geo4naics=lambda x: x['geoindkey'].str.slice(stop=-2),
        m1empFromModel=empMatA.loc[:, 'm1empFromModel'].astype(float)  # 5th column (0-based index 4)
    )

    df6 = df[df['geoindkey'].str.contains("_[0-9]{6}", regex=True)].copy()
    df6['geo4naics'] = df6['geoindkey'].str[:-2]
    df6['geo5naics'] = df6['geoindkey'].str[:-1]

    ################## Get NAICS6 By County Aggregates #######################
    print('---------- Getting NAICS6 by County Aggregates ----------')
    print('This may take a while, please be patient...')
    # Distribute values from 4-digit to 6-digit NAICS
    print(f'df4imp head before get_6naics_all in generate_NAICS.py:\n{df4imp.head()}')
    naics6df = get_6naics_all(df6, df4imp, codes4summary=count6dig)
    print(naics6df.columns)
    # Final formatting and output
    naics6df = naics6df.iloc[:, :5].join(naics6df[['emp1', 'emp3', 'wages']])
    naics6df.head(50)
    naics6df.to_csv(str(generalConfig["NAICS6_FILE"]),index=False)
    return(naics6df)


