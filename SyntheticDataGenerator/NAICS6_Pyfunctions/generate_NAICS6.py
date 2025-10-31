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

    np.random.seed(generalConfig["SEED"])
    if df is None:
        print('---------- Loading Dataset ----------\n')
        print(f"Dataset location: {generalConfig['COMBINED_DATASET']}")
        df = pd.read_csv(generalConfig['COMBINED_DATASET'], dtype=str)  # , nrows=100000)

        df = df.iloc[:, 1:]  # Remove index
    # dfsave = df.copy()

    ##check is qcew is being used?
    qcewcols=[col for col in df.columns if "_qcew" in col]
    if len(qcewcols)>0:
        print("Using QCEW when available...")
        useqcew=True
        for vname in ["estnum","emp3","emp2","emp1","wages"]:
            df[vname]=df[vname+"_qcew"]
            df[vname+"_source"]=""
            df.loc[~df[vname].isna(),vname+"_source"]="qcew"
        print("When QCEW wages are not availiable, use CBP.")
        df = quarter_source_adjustment(df, generalConfig, "wages", quarterConfig=None,
                                          formula="wages~wages_cbp",
                                          adjust_source=True, source="CBP", rseed=1)
        print("When QCEW emp1 and emp3 are not availiabl, use QWI and then CBP.")
        df = quarter_source_adjustment(df, generalConfig, "emp3", quarterConfig=None,
                                           formula="emp3~emp3_qwi",
                                           adjust_source=True, source="QWI", rseed=1)
        df = quarter_source_adjustment(df, generalConfig, "emp3", quarterConfig=None,
                                       formula="emp3~emp3_cbp",
                                       adjust_source=True, source="cbp", rseed=1)
        df = quarter_source_adjustment(df, generalConfig, "emp1", quarterConfig=None,
                                       formula="emp1~emp1_qwi",
                                       adjust_source=True, source="QWI", rseed=1)
            df.loc[:,vname+'_source']=np.nan
            df.loc[df[vname].notna(),vname+'_source'] ="QCEW"
            if vname=="wages":
                ## fill nas
                df.loc[df['wages'].isna(),'wages']=df.loc[df['wages'].isna(),'wages_cbp']
                df.loc[(df.loc[:,vname+"_source"]!="QCEW")&(df[vname].notna()),vname+"_source"]="CBP"
            else:
                df.loc[df[vname].isna(), vname] = df.loc[df[vname].isna(), vname+'_qwi']
                df.loc[(df.loc[:, vname + "_source"] != "QCEW") & (df[vname].notna()), vname + "_source"] = "QWI"
    else:
        useqcew = False
        df['estnum'] = df['estnum_cbp']
        df['emp3'] = df['emp3_qwi']
        df['emp1'] = df['emp1_qwi']
        df['wages'] = df['wages_cbp']

    # In 2016 this NAICS 4 has no corresponding NAICS 6
    ## Later we will automate this??
    #if "29189_525990" not in df["geoindkey"]:
    #    df = pd.concat([df, pd.DataFrame([{
    #        "state": "29", "cnty": "198", "emp": "1", "geoindkey": "29189_525990",
    #        "wages_cbp_flag": "Impute", "wages_cbp": "0", "estnum": "2", "geo_level": "C",
    #        "geography": "29189", "ind_level": "6", "industry": "525990",
    #        "avg_month_emp_wage": None, "EarnHirAS": None, "Emp": None, "emp3_qwi": None, "lwbd_emp_qwi": None,
    #        "avg_month_emp_wage_flag": None, "sEarnHirAS": None, "sEmp": None, "emp3_qwi_flag": None, "lwbd_emp_qwi_flag": None
    #    }])], ignore_index=True)
    #    df.loc[df['geoindkey'].str.contains("29189_5259//", regex=True), 'wages_cbp_flag'] = "Impute"

    df['year'] = generalConfig['YEAR']
    df['quarter'] = generalConfig['QTR']
    # Extract 6-digit NAICS
    df6 = df[df['geoindkey'].str.contains("_[0-9]{6}", regex=True)].copy()
    df6['geo4naics'] = df6['geoindkey'].str[:-2]
    df6['geo5naics'] = df6['geoindkey'].str[:-1]
    #df6 = df6[['geoindkey', 'geo4naics', 'geo5naics', 'state', 'cnty', 'estnum', 'wages_cbp', 'wages_cbp_flag', 'emp']]
    # Get summary counts for distribution
    count6dig = get_codes_summary(df, groupbydigits=4, levelgrouped=6)
    # Filter and prepare 4-digit NAICS data
    df4 = df[df['industry'] != "------"].copy()
    df4 = df[df['industry'].notna()].copy()
    df4 = df4[df4['industry'].str.match(r"^[0-9]{4}[^0-9]{2}", na=False)]
    # Create derived columns
    df4['sector'] = df4['naics2']
    #df4['state'] = df4['geography'].str[:-3]
    df4['geo4naics'] = df4['geoindkey'].str[:-2]
    df4['geo3naics'] = df4['geoindkey'].str[:-3]
    # Merge with summary data
    df4 = df4.merge(count6dig, on='geo4naics', how='left')
    df4['wagediff'] = df4['wages_cbp'].astype(float) - df4['wageCBP_sum6by4'].astype(float)
    columns_to_convert = ['emp3', 'wages', 'estnum', 'year', 'quarter', "emp1_qwi_flag"]
    df4[columns_to_convert] = df4[columns_to_convert].astype(float)

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
    empMatA.iloc[:, 1] = adjustm1emp  # Update with adjusted values

    if wageConfig is not None:
        ################## WAGES #######################
        print('---------- Wage Configuration ----------')
        # Display all current wageConfig settings
        for key, value in wageConfig.items():
            print(f"{key}: {value}")
        # Step 1. Create wage prediction model
        print('---------- Imputing Wage Data ----------')
        wagefit_sub = get_wage_model(df=df4, emp_mat_adj=empMatA,wageConfig=wageConfig)
        # Step 2. Get min/max bounds
    else:
        # Step 1. Create wage prediction model
        print('---------- Imputing Wage Data (if needed) ----------')
        wagefit_sub=None
    wage_maxmin = get_maxmindf(df4dig=df4, fulldf=df, emp_mat_adj=empMatA)
    # Prepare employment matrix
    empMatwage = pd.DataFrame({
        "m1emp": empMatA.iloc[:, 1],
        "m2emp": empMatA.iloc[:, 2],
        "m3emp": empMatA.iloc[:, 3]
    })
    # Step 3: Impute wage values
    wagesout = get_wages4(df4=df4, empmat=empMatwage, wagemodel=wagefit_sub, useEarnQWI=True, maxmindf=wage_maxmin)
    # Prepare final 4 digit output
    df4imp = wagesout.copy().assign(
        EmpScale=lambda x: np.where(x['m3emp'] == 0, 1, x['emp'] / x['m3emp'].astype(float)),
        geo4naics=lambda x: x['geoindkey'].str[:-2],
        m1empFromModel=empMatA.iloc[:, 4].astype(float)  # 5th column (0-based index 4)
    )

    ################## Get NAICS6 By County Aggregates #######################
    print('---------- Getting NAICS6 by County Aggregates ----------')
    print('This may take a while, please be patient...')
    # Distribute values from 4-digit to 6-digit NAICS
    naics6df = get_6naics_all(df=df6, df4n=df4imp, codes4summary=count6dig)
    # Final formatting and output
    naics6df = naics6df.iloc[:, :5].join(naics6df[['m1emp', 'm3emp', 'wage']])
    naics6df.head(50)
    naics6df.to_csv(str(generalConfig["NAICS6_PATH"]),index=False)
    return(naics6df)


