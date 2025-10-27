import pandas as pd
import os
import yaml
import sys
from formulaic import Formula

sys.path.append(os.path.abspath("./NAICS6_Pyfunctions/"))
from GeneralFunctions import *
from hierarchy_geoindkey import *
pd.set_option("display.max_columns", None)


excluded_cbp=["92----","111///","112///","482///","491///)","814///","525110", "525120","525190","525920","541120"]

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
        print("Only keeping QCEW aggregate level codes in CBP/QWI combined data :"+', '.join([str(x) for x in keepagglvls]))
        data=data[data['agglvl_code'].isin(keepagglvls)]
    data=data.apply(qcew_format_geoindkey,axis=1)
    #prepare combine data for merging
    #combine.drop(columns=['year','qtr'],inplace=True)
    #prepare qcew for merging
    data.drop(columns=['own_code'],inplace=True)
    data=data.loc[data['qtrly_estabs']>0,:]
    data.rename(columns={"month1_emplvl": "emp1_qcew","month2_emplvl": "emp2_qcew","month3_emplvl": "emp3_qcew",
                              "total_qtrly_wages": "wages_qcew",
                              "qtrly_estabs": "estnum_qcew"}, inplace=True)
    #data=fill_from_geoindkey(data,numeric_ind_level=True)
    #combine['cnty']=combine['cnty'].astype("str")
    #combine['state'] = combine['state'].astype("str")
    #combine['geography'] = combine['geography'].astype("str")
    colscomb=np.intersect1d(np.array(data.columns.values), np.array(combine.columns.values)).tolist()
    melddf = data.merge(combine, how="outer",
                          on=colscomb,
                          indicator=True, suffixes=["_qcew", "_other"], validate="one_to_one")
    melddf.rename(columns={"estnum": "estnum_cbp"}, inplace=True)
    melddf=fill_from_geoindkey(melddf,numeric_ind_level=True)
    melddf.drop(columns=["area_fips","industry_code"])

    melddf["_merge"] = melddf["_merge"].cat.rename_categories(
        {'right_only': 'cbp_qwi_only', 'left_only': 'qcew_only', "both": "both"})
    #melddf=data.merge(combine,how="outer",on=["geoindkey","geoindkey"],indicator=True,suffixes=["_combine","_qcew"],validate="one_to_one")
    if preprocessConfig['DIAGNOSTIC_FILE'] is not None:
        temp=melddf
        temp['cat']=temp['_merge'].astype(str)
        temp.loc[(temp['cat']=="qcew_only")&(temp['industry'].isin(excluded_cbp)),'cat']="qcew_only_excluded_cbp"
        xtabs=pd.crosstab(temp["agglvl_code"], temp["cat"])
        with open(preprocessConfig['OUTPATH'] + preprocessConfig["DIAGNOSTIC_FILE"], 'a') as f:
            print("--" * 20, file=f)
            print("----- Merging QCEW Data with Combined CBP and QWI Tables -----", file=f)
            print("--" * 20, file=f)
            print("Note the following NAICS codes are not included in CBP data: "+", ".join(excluded_cbp))
        xtabs.to_csv(preprocessConfig['OUTPATH'] + preprocessConfig["DIAGNOSTIC_FILE"],sep=",",mode="a")
    return(melddf)

def quarter_adjustment(data,generalConfig,quarterConfig):
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
        # Retrieve OLS formula from config.yaml
        formula="emp3_qcew"
        subqwifull = df[
            (~df["sEmpEnd"].isna()) &
            (df["sEmpEnd"].astype(float) != 5) &
            (df["sEmp"].astype(float) != 5) &
            (df["ind_level"] != "A")
            ].copy()
        subqwifull["Emp"] = subqwifull["Emp"].astype(float)
        subqwifull["EmpEnd"] = subqwifull["EmpEnd"].astype(float)
        subqwifull["estnum"] = subqwifull["estnum"].astype(float)

        # Create design matrices (gets the variables ready for fitting in statsmodels.OLS) using the formula
        # and perform initial model fitting
        y_pre, X_pre = Formula(formula).get_model_matrix(subqwifull)
        model_pre = sm.OLS(y_pre, X_pre).fit()
        # Calculate Cook's distance for each observation
        influence = OLSInfluence(model_pre)
        cooks_d = influence.cooks_distance[0]
        student_resid = influence.resid_studentized_internal




