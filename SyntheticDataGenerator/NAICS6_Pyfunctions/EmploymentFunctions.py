import re
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from formulaic import Formula
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
sys.path.append(os.path.abspath('./'))
from GeneralFunctions import *

# with open('./config.yaml','r') as configFile:
#     config = yaml.safe_load(configFile)
#     employmentConfig = config['employmentConfig']
pd.set_option('mode.chained_assignment', None)

def get_m1emp_model(df,employmentConfig):
    '''
    What is the point?
        get_m1emp_model() creates an OLS model that predicts employment values ('Emp') based on various
        predictors. This model is used when direct employment data is missing.
    Steps:
        1. Filters input data to include only rows where
            - 'emp3_qwi_flag' is not suppressed
            - 'sEmp' is not supressed
            - 'ind_level' is not "A" 
        2. Initial model fitting
            - Use formula specified in config.yaml (default: 'Emp ~ emp3_qwi + estnum + C(sector) + C(state)')
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

    df=df.copy()
    if "DIAGNOSTIC_PLOTS" in employmentConfig:
        diagplot = employmentConfig['DIAGNOSTIC_PLOTS']
    else:
        diagplot = None

    if "emp1diff" in employmentConfig['OLS_FORMULA']:
        if employmentConfig['OLS_FORMULA'].startswith("np.log") or employmentConfig['OLS_FORMULA'].startswith("np.sqrt"):
            model = get_model(df.loc[(df['emp1_missing6by4'] > 0)&(df['emp1diff']>0), :], employmentConfig['OLS_FORMULA'],
                              employmentConfig['COOKS_THRESH'], employmentConfig['OUTLIER_THRESH'],
                              diagnostic_plots=diagplot, output_removed=False, include_multicolinearity=True,
                              return_summary_and_diagnostics=False)
        else:
            model = get_model(df.loc[df['emp1_missing6by4']>0,:], employmentConfig['OLS_FORMULA'], employmentConfig['COOKS_THRESH'], employmentConfig['OUTLIER_THRESH'],
                          diagnostic_plots=diagplot, output_removed=False, include_multicolinearity=True,
                          return_summary_and_diagnostics=False)
    else:
        model=get_model(df, employmentConfig['OLS_FORMULA'], employmentConfig['COOKS_THRESH'], employmentConfig['OUTLIER_THRESH'], diagnostic_plots=diagplot, output_removed=False,include_multicolinearity=True,
                  return_summary_and_diagnostics=False)
    print(model.summary())


    # if "emp1diff" in employmentConfig['OLS_FORMULA']:
    #     modeldf=df.loc[(df['emp1_missing6by4']>0),:]
    #     #modeldf=modeldf.loc[(modeldf['emp1diff']>0)&(modeldf['emp3']>=0),:]
    # else:
    #     modeldf=df
    # model = get_model(modeldf,employmentConfig['OLS_FORMULA'],employmentConfig['COOKS_THRESH'],employmentConfig['OUTLIER_THRESH'],diagnostic_plots=diagplots,output_removed=False,include_multicolinearity=True,return_summary_and_diagnostics=False)#.OLS(y, X).fit()
    # print(model.summary())
    # # end. return fitted model.
    return model

def check_lwbd_emp_qwi(empvals, stablevals):
    '''
    What is the point?
        check_lwbd_emp_qwi() is used as a helper function in get_m1emp() and get_m2emp().
        The purpose of this function is to choose whether or not to use predicted employment values or
        the stable ones given as 'lwbd_emp_qwi' in the dataset
    Inputs:
        1. empvals  -  any array-like
            - Contains employment values to be checked
        2. stablevals  -  any array like
            - Stable employment values for reference ('lwbd_emp_qwi') from the dataset
        3. stableFlag  - any array-like
            - Suppression flags for stable values (1: not suppressed, NaN / \neq 1: suppressed )
    Steps:
        1. Convert everything to np arrays of floats
        2. 'empfitokay'
            - True when stable value is suppressed (stableFlag is NaN or \neq 1) OR
              when stable value exists but empvals \geq stablevals 
        3. Replace empvals with stablevals where conditions aren't met
    Returns:
        1. empvals  -  np.ndarray of floats
            - Corrected employment values
    '''
    if isinstance(stablevals,pd.Series):
        stabnan=stablevals.isna()
    else:
        stabnan=np.isnan(stablevals)
    empvals = np.array(empvals, dtype=float)
    stablevals = np.array(stablevals, dtype=float)
    #stableFlag = np.array(stableFlag, dtype=float)
    empfitokay = np.isnan(stablevals) | ((~stabnan) & (empvals >= stablevals))
    empvals[~empfitokay] = stablevals[~empfitokay]
    return empvals

def get_m1emp(df, m1empmodel, rseed=None, include_indicator=False):
    '''
    What is the point?
        get_m1emp() is used as a helper function in get_employmentCounts4().
        It fills missing / suppressed m1emp values with model predictions.
    Inputs: 
        1. df  -  pd.DataFrame
        2. m1empmodel  -  statsmodels.OLS
            - Pre-trained regression model for employment prediction (from get_m1emp_model())
        3. rseed  -  int
            - Random seed (configurable in config.yaml)
        4. include_indicator  -  bool
            - Whether to include flag that tells if m1emp is imputed or not.
    Steps:
        1. Initialization:
            - Sets random seed
            - Starts with Emp from dataset
        2. Flags rows where employment is suppressed.
        3. Prediction:
            - Use custom_predict() to get predictions and standard errors
            - Add normally distributed noise scaled by SEs
        4. Correction:
            - Ensures no negative employment
            - Cross-checks with stable values using check_lwbd_emp_qwi()
        5. Output prep:
            - Rounds results to whole numbers
            - Optionally appends imputation indicator.
    Returns:
        output  -  np.ndarray of floats (if include_indicator=False)
            - Array of employment values
        OR
                -  Stacked np.ndarray of floats and bools (if include_indicator=True)
            - Array of [employment values, imputation flags]
    '''
    if rseed is not None:
        np.random.seed(rseed)
    m1emp = df["emp1"]

    # Identify rows to be imputed
    missm1indicator = df["emp1"].isna()
    if 'estnum_emp1_missing6by4' in df.columns:
        nomissingestnum=(df['estnum_emp1_missing6by4']==0)
        df.loc[(missm1indicator)&(nomissingestnum),'emp1']=df.loc[(missm1indicator)&(nomissingestnum),'emp1_sum6by4']
        missm1indicator = df["emp1"].isna()

    missingsub = df[missm1indicator]
    # Get model predictions and standard errors for missing values
    predm1emp, sem1emp= custom_predict(missingsub, m1empmodel)

    ##check index is now filled
    predidx=predm1emp[predm1emp.notna()].index.values
    missingidx=missingsub.index.values


    if set(predidx)!=set(missingidx):
        print(f'Something wrong: some of the {len(missingidx)} missing emp1 values have not been filled by the {len(predidx)} prediction values. Printing head of unfilled missing values...')
        print(df.loc[list(set(missingidx)-set(predidx)),["state","naics2","estnum","emp3","emp3_source","emp1_sum6by4","emp1_missing6by4","emp1"]].head())

    # Generate predicted values with random noise based on standard errors
    m1empfit = np.random.normal(
        loc=predm1emp, # Center at predicted values
        scale=sem1emp, # Scale by prediction uncertainty
        size=sum(missm1indicator)
    )
    response = m1empmodel.model.endog_names
    if "np.sqrt" in response:
        m1empfit=m1empfit**2
    elif "np.log" in response:
        m1empfit=np.exp(m1empfit)

    if "emp1diff" in response:
        m1empfit=m1empfit+missingsub['emp1_sum6by4']
    # Ensure no negative employment and validate against stable values

    m1empfit[m1empfit < 0] = 0
    m1emp[missm1indicator]=m1empfit
    m1emp[missm1indicator] = check_lwbd_emp_qwi(m1empfit, missingsub["lwbd_emp_qwi"])
    #m1emp[missm1indicator]=adjust_varvalues(m1emp[missm1indicator], dfmaxmin, stabvals=None, variable="emp1")
    # Round to whole numbers
    output = np.round(m1emp.astype(float), 0)
    # Optionally include imputation indicatior
    if include_indicator:
        return output, missm1indicator
    return output

def get_m2emp(m1emp, m3emp, stabval, noisecoef, rseed=None):
    '''
    What is the point?
        get_m1emp() is used as a helper function in get_employmentCounts4().
        It estimates month 2 employment by interpolating between the predicted m3emp and 
        the predicted m1emp values.
    Inputs:
        1. m1emp  -  array-like
            - Month 1 employment values
        2. m3emp  -  array-like
            - Month 3 employment values
        3. stabval  -  array-like
            - Stable employment values for reference
        4. stabF  -  array-like
            - Suppression flags for stable values
        5. noisecoef  -  float
            - Coefficient controlling noise magnitude (configurable in config.yaml)
        6. rseed  - int
            - Random seed (configurable in config.yaml)
    Steps:
        1. Initialization:
            - Sets random seed
            - creates array for m2emp
        2. Identify rows with non-zero employment in either month 1 or 3
        3. Set m2emp
            - Calculate midpoint of m1 and m3emp
            - Add noise
        4. Correct
            - Ensure non negative values
            - Cross-checks with stable values using check_lwbd_emp_qwi()
            - Round output to whole number
    Returns:
        m2emp  -  np.ndarray of floats (if include_indicator=False)
            - Array of employment values
        
    '''
    if rseed is not None: 
        np.random.seed(rseed)
    m2emp = np.zeros(len(m1emp))
    m3emp=m3emp.astype(float)
    # Identify rows with non-zero employment in either month 1 or 3
    nonzeroindic = (m1emp > 0) | (m3emp > 0)
    m1emp_nz = m1emp[nonzeroindic]
    m3emp_nz = m3emp[nonzeroindic]
    # Calculate SD for noise
    # Proportional to employment change relative to mean employment
    noisesd = np.sqrt((noisecoef * 2 * np.abs(m1emp_nz - m3emp_nz)) / (m1emp_nz + m3emp_nz))
    # Generate random noise and add to midpoint of m1 and m3emp
    changeFromMid = np.random.normal(0, noisesd)
    m2emp_nz = m1emp_nz + ((m3emp_nz - m1emp_nz) / 2) + changeFromMid
    m2emp[nonzeroindic] = m2emp_nz
    #Handle negative values and consult check_lwbd_emp_qwi()
    m2emp[m2emp < 0] = np.where((m1emp[m2emp < 0] == 0) | (m3emp[m2emp < 0] == 0), 0, 1)
    m2emp = check_lwbd_emp_qwi(m2emp, stabval)
    #return
    return np.round(m2emp, 0)

def get_employmentCounts4(df4,m1emp_model, m2emp_noisecoef, rseed=None, include_m1emp_indicator=True):
    '''
    What is the point?
        Putting everything together adjust_countytotal_qwi() generates the complete 
        quarterly employment matrix (months 1-3)
    Inputs:
        1. df4  -  pd.DataFrame
        2. m1emp_model  -  statsmodels.OLS
            - Pre-trained regression model for employment prediction (from get_m1emp_model())
        3. m2emp_noisecoef  -  float
            - Coefficient controlling noise magnitude (configurable in config.yaml)
        4. rseed  -  int
            - Random seed (configurable in config.yaml)
        4. include_m1emp_indicator  -  bool
            - Whether to include flag that tells if m1emp is imputed or not.
    Returns:
        empMat  -  pd.DataFrame
            - geoindkey
            - m1emp
            - m2emp
            - m3emp
            - m1empFromModel (if include_indicator=True)
    '''
    if rseed is not None:
        np.random.seed(rseed)
    #qwiemp3_qwiAvailable = ~df4["emp3_qwi"].isna()
    #df4.loc[~qwiemp3_qwiAvailable, "emp3_qwi"] = df4.loc[~qwiemp3_qwiAvailable, "emp"]
    if include_m1emp_indicator:
        m1emp, m1empFlag = get_m1emp(df=df4, m1empmodel=m1emp_model, include_indicator=True)

        df4.loc[m1empFlag,"emp1_source"]="model"
    else:
        m1emp = get_m1emp(df=df4, m1empmodel=m1emp_model, include_indicator=False)
    m3emp=df4['emp3']

    m2emp = get_m2emp(m1emp, m3emp, df4['lwbd_emp_qwi'].values, noisecoef=m2emp_noisecoef)
    empMat = pd.DataFrame({
    'geoindkey': df4['geoindkey'],
    'm1emp': m1emp,
    'm2emp': m2emp,
    'm3emp': m3emp,
    'm1empFromModel': m1empFlag
    })

    check_emp2=False #used for internal validatation while testing code.
    if check_emp2:
        check_m2emp(df4,m2emp)

    df4.loc[df4["emp1_source"]=="model","emp1"]=m1emp
    df4.loc[df4["emp2_source"].isna(), "emp2_source"] = "emp2_noise"
    df4.loc[df4["emp2"].isna(),"emp2"]=m2emp[df4["emp2"].isna()]
    return empMat, df4

def check_m2emp(df,m2emp,justqcew=False,plotsave="DataDiag/DiagnosticPlots/check_m2emp_plot.png",bins=45):
    #if justqcew:
    #    df4=df[(df["emp1_source"]=="qcew")&(df["emp2_source"]=="qcew")&(df["emp3_source"]=="qcew"),:].copy()
    #else:
    #    df4 = df[(df["emp1_source"] != "model") & (df["emp2_source"] == "qcew") ,
    #          :].copy()
    df4=df.loc[df["emp1_source"]=="qcew"]
    emp2_diff = df4["emp2"] - m2emp[df4.index.values]
    print("Checking emp2.... emp2_diff")
    print(emp2_diff.describe())
    print("abs(emp2_diff)")
    print(emp2_diff.abs().describe())
    # find cases with zeros
    for empvar in ["emp1","emp2","emp3"]:
        df4[empvar+"_zero"]=(df4[empvar]==0)
    print("zero combinations")
    print(df4.groupby(['emp1_zero',"emp2_zero","emp3_zero"]).size().reset_index(name="Count"))
    m1emp_nz = df4.loc[(df4["emp1"] > 0) & (df4["emp3"] > 0), "emp1"]
    m3emp_nz = df4.loc[(df4["emp3"] > 0) & (df4["emp1"] > 0), "emp3"]
    noisescalar = 2/(m1emp_nz+m3emp_nz)#(2 * np.abs(m1emp_nz - m3emp_nz)) / (m1emp_nz + m3emp_nz)
    midpoint=((m3emp_nz-m1emp_nz)/2)+m1emp_nz
    print("midpoint summary")
    print(midpoint.describe())
    #noisescalar[noisescalar==0]=1
    print("noisescalar summary")
    print(noisescalar.describe())
    check_norm=((m2emp[midpoint.index.values]-midpoint)*noisescalar)
    print("check norm summary")
    print(check_norm.describe())
    plot_hist_with_normal(check_norm,filename=plotsave,bins=bins)
    return (df,m2emp)



    #noisesd = np.sqrt((noisecoef * 2 * np.abs(m1emp_nz - m3emp_nz)) / (m1emp_nz + m3emp_nz))
def adjust_countytotal_qwi(valdf, sumdf):
    '''
    What is the point?
        adjust_countytotal_qwi() ensures that the sum of imputed industry-level employment values
        (m1emp in valdf) matches the real county-level totals (Emp in sumdf)
    Inputs:
        1. valdf  -  pd.DataFrame
            Output of get_employmentCounts4()
        2. sumdf  -  pd.DataFrame
            County Level full df
    Returns:
        Adjusted m1emp values as np.ndarray of floats rounded to whole numbers.
    '''
    sumdf = sumdf.copy()
    # Extract county codes from sumdf(eg. '11111' from '11111_XXXXXX')
    sumdf["stcnty"] = sumdf["geoindkey"].apply(lambda x: re.sub(r"_.*", "", x))
    sumdf = sumdf[["stcnty", "emp1", "emp1_source"]]
    # Filter to counties with non-suppressed totals
    HasSumIndic = sumdf["emp1_source"].notna()#.astype(float) == 1.0
    groupdf = valdf.copy()
    # Extract county codes from valdf
    groupdf["stcnty"] = groupdf["geoindkey"].apply(lambda x: re.sub(r"_.*", "", x))
    # Covert imputation flag from 0,1 to 'QWI','Model'
    groupdf["m1empFromModel"] = groupdf["m1empFromModel"].apply(
        lambda x: "Model" if x >= 1 else "Data"
    )
    # Keep only counties with known totals
    filtered_df = groupdf[groupdf['stcnty'].isin(sumdf.loc[HasSumIndic, 'stcnty'])]
    filtered_df = filtered_df[['stcnty', 'emp1', 'm1empFromModel']]
    # Calculate sum of m1emp and counts for "Model"/"QWI" categories
    result_df = filtered_df.groupby(['stcnty', 'm1empFromModel']).agg(
        summ1emp=('emp1', 'sum'),
        CellCount=('emp1', 'size')
    ).reset_index()
    # Pivot to wide format (columns like summ1emp_Model, CellCount_QWI)
    groupeddf = result_df.pivot_table(
        index='stcnty', 
        columns='m1empFromModel', 
        values=['summ1emp', 'CellCount'], 
        aggfunc='first'
    ).reset_index()
    # Flatten multi-level column names
    groupeddf.columns = [
        f"{col[0]}_{col[1]}" if col[0]!='stcnty' else col[0] 
        for col in groupeddf.columns
    ] 
    groupeddf = groupeddf[['stcnty', 'summ1emp_Model', 'summ1emp_Data', 'CellCount_Model', 'CellCount_Data']]
    groupdf = groupeddf.copy()
    # QWI_Emp: Total employment from non-imputed rows
    groupdf['Data_Emp'] = groupdf['summ1emp_Data']
    # Model: Total employment from imputed records
    groupdf['Model'] = groupdf['summ1emp_Model']
    groupdf = groupdf.drop(columns=['summ1emp_Data', 'summ1emp_Model', 'CellCount_Data'])
    # Merge with official county totals
    mergedf = pd.merge(
        groupdf, 
        sumdf.loc[HasSumIndic],#.drop(columns=['emp1_qwi_flag']),
        on='stcnty', 
        how='outer'
    )
    # ModelTotal: Target total for imputed records (official total - QWI_Emp)
    mergedf['ModelTotal'] = mergedf['emp1'].astype(float) - mergedf['Data_Emp']
    # MissingModel: Residual to distribute among imputed records
    mergedf['MissingModel'] = mergedf['ModelTotal'] - mergedf['Model']
        # Merge discrepancies back into original industry-level data
    valdf['stcnty'] = valdf['geoindkey'].str.replace(r'_.*', '', regex=True)
    valdf['stcnty'] = valdf['stcnty'].astype(float)
    mergedf['stcnty'] = mergedf['stcnty'].astype(float)
    valdf = pd.merge(
        valdf, mergedf,
        on='stcnty', 
        how='outer', 
        suffixes=('', '_agg')
    )
    # Filter to only rows with imputation flags
    valdf = valdf[valdf['m1empFromModel'].notna()]
    # Calculate proposed adjustments:
    # - Proportional adjustment if Model > 0
    # - Equal distribution if Model = 0
    valdf['ProposedM1emp'] = np.where(
        (valdf['Model'] == 0) | valdf['Model'].isna(),
        valdf['emp1'] + (valdf['MissingModel'] / valdf['CellCount_Model']),
        valdf['emp1'] + (valdf['MissingModel'] * valdf['emp1'] / valdf['Model'])
    )
    # Apply adjustments only to imputed records
    valdf.loc[(valdf['m1empFromModel'] == 1) & valdf['Data_Emp'].notna(), 'emp1'] = valdf['ProposedM1emp']
    # Ensure non-negative, round and return
    valdf['emp1'] = valdf['emp1'].clip(lower=0)
    return valdf['emp1'].round(0)

