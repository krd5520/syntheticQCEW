import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from formulaic import Formula
import patsy
import yaml
import sys
import os
sys.path.append(os.path.abspath('./'))
from GeneralFunctions import *
with open('./config.yaml','r') as configFile:
    config = yaml.safe_load(configFile)
    employmentConfig = config['employmentConfig']
pd.set_option('mode.chained_assignment', None)

def get_m1emp_model(df): 
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
    subqwifull = df[
        (~df["sEmpEnd"].isna()) & 
        (df["sEmpEnd"].astype(float) != 5) &
        (df["sEmp"].astype(float) != 5) &
        (df["ind_level"] != "A")
    ].copy()
    subqwifull["Emp"] = subqwifull["Emp"].astype(float)
    subqwifull["EmpEnd"] = subqwifull["EmpEnd"].astype(float)
    subqwifull["estnum"] = subqwifull["estnum"].astype(float)
    # Retrieve OLS formula from config.yaml
    formula = employmentConfig['OLS_FORMULA']
    # Create design matrices (gets the variables ready for fitting in statsmodels.OLS) using the formula
    # and perform initial model fitting
    y_pre, X_pre = Formula(formula).get_model_matrix(subqwifull)
    model_pre = sm.OLS(y_pre, X_pre).fit()
    # Calculate Cook's distance for each observation
    influence = OLSInfluence(model_pre)
    cooks_d=influence.cooks_distance[0]
    student_resid = influence.resid_studentized_internal
    # Identify and remove indices of influential points. (Cook's Distance > threshold)
    # Threshold is configurable in config.yaml -> employmentConfig -> 'COOKS_THRESH'
    influential_indices = [i for i, d in enumerate(cooks_d) if d > employmentConfig['COOKS_THRESH']]
    outliers = [i for i, r in enumerate(student_resid) if np.abs(r) > employmentConfig['OUTLIER_THRESH']]
    if influential_indices:
        print("Filtered out the following indices due to influence (Cook's Distance):", influential_indices)
    if outliers:
        print("# of outliers filtered (Studentized Residuals):", len(outliers), '|', np.round((len(outliers)/len(subqwifull)),3) * 100, '%')
    rows_to_drop = influential_indices + outliers
    subqwifull = subqwifull.drop(subqwifull.index[rows_to_drop])
    # Rebuild design matrices without influential points and perform final model fitting.
    y, X = Formula(formula).get_model_matrix(subqwifull)
    model = sm.OLS(y, X).fit()
    # end. return fitted model.
    return model

def check_EmpS(empvals, stablevals, stableFlag):
    '''
    What is the point?
        check_EmpS() is used as a helper function in get_m1emp() and get_m2emp().
        The purpose of this function is to choose whether or not to use predicted employment values or
        the stable ones given as 'EmpS' in the dataset
    Inputs:
        1. empvals  -  any array-like
            - Contains employment values to be checked
        2. stablevals  -  any array like
            - Stable employment values for reference ('EmpS') from the dataset
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
    empvals = np.array(empvals, dtype=float)
    stablevals = np.array(stablevals, dtype=float)
    stableFlag = np.array(stableFlag, dtype=float)
    empfitokay = np.isnan(stableFlag) | (stableFlag != 1) | ((stableFlag == 1) & (empvals >= stablevals))
    empvals[~empfitokay] = stablevals[~empfitokay]
    return empvals

def get_m1emp(df, m1empmodel, rseed=employmentConfig['RSEED'], include_indicator=False):
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
            - Cross-checks with stable values using check_EmpS()
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
    m1emp = df["Emp"]
    # Identify rows to be imputed
    missm1indicator = df["sEmp"] != 1.0
    missingsub = df[missm1indicator]
    # Get model predictions and standard errors for missing values
    predm1emp, sem1emp= custom_predict(missingsub, m1empmodel)
    # Generate predicted values with random noise based on standard errors
    m1empfit = np.random.normal(
        loc=predm1emp, # Center at predicted values
        scale=sem1emp, # Scale by prediction uncertainty
        size=sum(missm1indicator)
    )
    # Ensure no negative employment and validate against stable values
    m1empfit[m1empfit < 0] = 0
    m1emp[missm1indicator] = check_EmpS(m1empfit, missingsub["EmpS"], missingsub["sEmpS"])
    # Round to whole numbers
    output = np.round(m1emp.astype(float), 0)
    # Optionally include imputation indicatior 
    if include_indicator:
        output = np.column_stack((output, missm1indicator))
    return output

def get_m2emp(m1emp, m3emp, stabval, stabF, noisecoef, rseed=employmentConfig['RSEED']):
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
            - Cross-checks with stable values using check_EmpS()
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
    #Handle negative values and consult check_EmpS()
    m2emp[m2emp < 0] = np.where((m1emp[m2emp < 0] == 0) | (m3emp[m2emp < 0] == 0), 0, 1)
    m2emp = check_EmpS(m2emp, stabval, stabF)
    #return
    return np.round(m2emp, 0)

def get_employmentCounts4(df4,m1emp_model, m2emp_noisecoef, rseed=employmentConfig['RSEED'], include_m1emp_indicator=False):
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
    qwiEmpEndAvailable = ~df4["EmpEnd"].isna()
    df4.loc[~qwiEmpEndAvailable, "EmpEnd"] = df4.loc[~qwiEmpEndAvailable, "emp"]
    if include_m1emp_indicator:
        m1empAndFlag = get_m1emp(df=df4, m1empmodel=m1emp_model, include_indicator=True)
        m1emp = m1empAndFlag[:, 0]  # First column (m1emp values)
        m1empFlag = m1empAndFlag[:, 1]  # Second column (m1empFlag)
    m3emp=df4['EmpEnd']
    m2emp = get_m2emp(m1emp, m3emp, df4['EmpS'].values, df4['sEmpS'].values, noisecoef=m2emp_noisecoef)
    empMat = pd.DataFrame({
    'geoindkey': df4['geoindkey'],
    'm1emp': m1emp,
    'm2emp': m2emp,
    'm3emp': m3emp,
    'm1empFromModel': m1empFlag
    })
    return empMat

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
    sumdf = sumdf[["stcnty", "Emp", "sEmp"]] 
    # Filter to counties with non-suppressed totals
    HasSumIndic = sumdf["sEmp"].astype(float) == 1.0
    groupdf = valdf.copy()
    # Extract county codes from valdf
    groupdf["stcnty"] = groupdf["geoindkey"].apply(lambda x: re.sub(r"_.*", "", x))
    # Covert imputation flag from 0,1 to 'QWI','Model'
    groupdf["m1empFromModel"] = groupdf["m1empFromModel"].apply(
        lambda x: "Model" if x >= 1 else "QWI"
    )
    # Keep only counties with known totals
    filtered_df = groupdf[groupdf['stcnty'].isin(sumdf.loc[HasSumIndic, 'stcnty'])]
    filtered_df = filtered_df[['stcnty', 'm1emp', 'm1empFromModel']]
    # Calculate sum of m1emp and counts for "Model"/"QWI" categories
    result_df = filtered_df.groupby(['stcnty', 'm1empFromModel']).agg(
        summ1emp=('m1emp', 'sum'),
        CellCount=('m1emp', 'size')
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
    groupeddf = groupeddf[['stcnty', 'summ1emp_Model', 'summ1emp_QWI', 'CellCount_Model', 'CellCount_QWI']]
    groupdf = groupeddf.copy()
    # QWI_Emp: Total employment from non-imputed rows
    groupdf['QWI_Emp'] = groupdf['summ1emp_QWI']
    # Model: Total employment from imputed records
    groupdf['Model'] = groupdf['summ1emp_Model']
    groupdf = groupdf.drop(columns=['summ1emp_QWI', 'summ1emp_Model', 'CellCount_QWI'])
    # Merge with official county totals
    mergedf = pd.merge(
        groupdf, 
        sumdf.loc[HasSumIndic].drop(columns=['sEmp']),  
        on='stcnty', 
        how='outer'
    )
    # ModelTotal: Target total for imputed records (official total - QWI_Emp)
    mergedf['ModelTotal'] = mergedf['Emp'].astype(float) - mergedf['QWI_Emp']
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
        valdf['m1emp'] + (valdf['MissingModel'] / valdf['CellCount_Model']),
        valdf['m1emp'] + (valdf['MissingModel'] * valdf['m1emp'] / valdf['Model'])
    )
    # Apply adjustments only to imputed records
    valdf.loc[(valdf['m1empFromModel'] == 1) & valdf['QWI_Emp'].notna(), 'm1emp'] = valdf['ProposedM1emp']
    # Ensure non-negative, round and return
    valdf['m1emp'] = valdf['m1emp'].clip(lower=0)
    return valdf['m1emp'].round(0)
