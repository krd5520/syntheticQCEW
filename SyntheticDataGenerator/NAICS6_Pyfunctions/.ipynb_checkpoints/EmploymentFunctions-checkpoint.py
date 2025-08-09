import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import statsmodels.api as sm
pd.set_option('mode.chained_assignment', None)

def get_m1emp_model(df): 
    subqwifull = df[
        (~df["sEmpEnd"].isna()) & 
        (df["sEmpEnd"].astype(float) != 5) &
        (df["sEmp"].astype(float) != 5) &
        (df["ind_level"] != "A")
    ].copy()

    # Create the "EmpProp" column as the ratio
    subqwifull["EmpProp"] = subqwifull["Emp"].astype(float) / subqwifull["EmpEnd"].astype(float)

    # Drop the outlier row with index 12004
    subqwifull = subqwifull.drop(index=12004, errors="ignore")

    # Define the dependent and independent variables
    y = subqwifull["Emp"].astype(float)
    X = subqwifull[["EmpEnd", "estnum", "sector", "state"]]

    # Convert categorical variables (e.g., "sector" and "state") to dummy variables
    X = pd.get_dummies(X, columns=["sector", "state"], drop_first=True)
    X = X.astype(float)

    # Add a constant (intercept) to the independent variables
    X = sm.add_constant(X)

    # Fit the model using statsmodels OLS
    model = sm.OLS(y, X).fit()

    # Return the fitted model
    return model

def check_EmpS(empvals, stablevals, stableFlag):
    """
    If stable employment (EmpS) is not suppressed + fit value is > EmpS -> use fit value.
    Otherwise, use EmpS.
    """
    # Convert inputs to NumPy arrays if they aren't already (for element-wise operations)
    empvals = np.array(empvals, dtype=float)
    stablevals = np.array(stablevals, dtype=float)
    stableFlag = np.array(stableFlag, dtype=float)

    # Condition where we keep empvals
    empfitokay = np.isnan(stableFlag) | (stableFlag != 1) | ((stableFlag == 1) & (empvals >= stablevals))

    # Apply the condition
    empvals[~empfitokay] = stablevals[~empfitokay]

    return empvals
def custom_predict(df, ols_model):
    # Prepare the features for prediction
    X = df[["EmpEnd", "estnum", "sector", "state"]].copy()
    X = X.astype(float)

    # Create dummy variables for categorical features
    X = pd.get_dummies(X, columns=["sector", "state"], drop_first=True)
    
    # Add constant to X (intercept term)
    X = sm.add_constant(X)

    # Use the OLS model to make predictions
    prediction = ols_model.predict(X)
    
    # Compute standard errors of predictions
    cov_matrix = ols_model.cov_params()  # Covariance matrix of estimated coefficients
    X_np = X.to_numpy()  # Convert to numpy array for matrix operations
    pred_var = np.einsum("ij,jk,ik->i", X_np, cov_matrix, X_np)  # Variance of predictions
    se_fit = np.sqrt(pred_var.astype(float))  # Standard error of fitted values

    return prediction, se_fit
def get_m1emp(df, m1empmodel, rseed=None, include_indicator=False):
    # If a random seed is supplied, set the random seed
    if rseed is not None:
        np.random.seed(rseed)
    
    # M1 emp is the Beginning of Q1 measure from QWI
    m1emp = df["Emp"]
    #df["sEmp"] = pd.to_numeric(df["sEmp"], errors='coerce')  # Convert to numeric, invalid parsing will be set as NaN

    missm1indicator = df["sEmp"] != 1.0
    missingsub = df[missm1indicator]
    predm1emp, sem1emp= custom_predict(missingsub, m1empmodel)
    m1empfit = np.random.normal(loc=predm1emp, scale=sem1emp, size=sum(missm1indicator))
    m1empfit[m1empfit < 0] = 0
    m1emp[missm1indicator] = check_EmpS(m1empfit, missingsub["EmpS"], missingsub["sEmpS"])
    output = np.round(m1emp.astype(float), 0)
    if include_indicator:
        output = np.column_stack((output, missm1indicator))


    return output
def get_m2emp(m1emp, m3emp, stabval, stabF, noisecoef=0.5, rseed=None):
    if rseed is not None:  # If a random seed is supplied, set the seed
        np.random.seed(rseed)

    m2emp = np.zeros(len(m1emp))
    m3emp=m3emp.astype(float)# Initialize m2emp as 0

    # Indicator that m1emp or m3emp is non-zero
    nonzeroindic = (m1emp > 0) | (m3emp > 0)

    # Filter non-zero elements
    m1emp_nz = m1emp[nonzeroindic]
    m3emp_nz = m3emp[nonzeroindic]

    # Noise variance is proportional to the distance between employment values relative to the mean
    noisesd = np.sqrt((noisecoef * 2 * np.abs(m1emp_nz - m3emp_nz)) / (m1emp_nz + m3emp_nz))

    # Generate normal noise with the calculated standard deviation
    changeFromMid = np.array([np.random.normal(0, sd) for sd in noisesd])

    # Calculate m2emp for non-zero entries
    m2emp_nz = m1emp_nz + ((m3emp_nz - m1emp_nz) / 2) + changeFromMid

    # Assign m2emp values where m1emp or m3emp is non-zero
    m2emp[nonzeroindic] = m2emp_nz

    # Force non-negative values for m2emp
    m2emp[m2emp < 0] = np.where((m1emp[m2emp < 0] == 0) | (m3emp[m2emp < 0] == 0), 0, 1)

    # Apply stability check
    m2emp = check_EmpS(m2emp, stabval, stabF)

    # Return rounded m2emp
    return np.round(m2emp, 0)
def get_employmentCounts4(df4,m1emp_model, m2emp_noisecoef=1, rseed=None, include_m1emp_indicator=False):
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
    # Take summation data and extract the Emp and sEmp for the geography code
    sumdf = sumdf.copy()
    sumdf["stcnty"] = sumdf["geoindkey"].apply(lambda x: re.sub(r"_.*", "", x))
    sumdf = sumdf[["stcnty", "Emp", "sEmp"]]  # Emp is total for county
    HasSumIndic = sumdf["sEmp"].astype(float) == 1.0
    groupdf = valdf.copy()
    groupdf["stcnty"] = groupdf["geoindkey"].apply(lambda x: re.sub(r"_.*", "", x))  # Get geography code
    groupdf["m1empFromModel"] = groupdf["m1empFromModel"].apply(lambda x: "Model" if x >= 1 else "QWI")
    filtered_df = groupdf[groupdf['stcnty'].isin(sumdf.loc[HasSumIndic, 'stcnty'])]

    # Select only the relevant columns
    filtered_df = filtered_df[['stcnty', 'm1emp', 'm1empFromModel']]
    
    # Group by 'stcnty' and 'm1empFromModel' and compute sum of 'm1emp' and count of rows
    result_df = filtered_df.groupby(['stcnty', 'm1empFromModel']).agg(
        summ1emp=('m1emp', 'sum'),
        CellCount=('m1emp', 'size')
    ).reset_index()
    groupeddf = result_df.pivot_table(
        index='stcnty', 
        columns='m1empFromModel', 
        values=['summ1emp', 'CellCount'], 
        aggfunc='first'
    ).reset_index()
    
    # Flatten the multi-level column index
    groupeddf.columns = [f"{col[0]}_{col[1]}" if col[0]!='stcnty' else col[0] for col in groupeddf.columns] #ignores stcnty to avoid column named stcnty_
    groupeddf = groupeddf[['stcnty', 'summ1emp_Model', 'summ1emp_QWI', 'CellCount_Model', 'CellCount_QWI']] #reorder columns
    
    groupdf = groupeddf.copy()
    groupdf['QWI_Emp'] = groupdf['summ1emp_QWI']
    groupdf['Model'] = groupdf['summ1emp_Model']
    
    # Drop the columns 'summ1emp_QWI', 'summ1emp_Model', and 'CellCount_QWI'
    groupdf = groupdf.drop(columns=['summ1emp_QWI', 'summ1emp_Model', 'CellCount_QWI'])
    #columns are stcnty, CellCount_Model, 
    #  QWI_Emp (employment summed for values that did not come from the model),
    #  Model (employment summed for values that are generated from the model)
    mergedf = pd.merge(
        groupdf, 
        sumdf.loc[HasSumIndic].drop(columns=['sEmp']),  # Select only relevant rows and columns from sumdf
        on='stcnty', 
        how='outer'
    )
    #columns are: stcnty, CellCount_Model, QWI_Emp (sum NAICS6 from QWI data), 
    #    Model (sum NAICS6 from model), Emp (from sumdf)
    # get difference for each county between month 1 employment (across all industries) 
    # and sum of month 1 employments (not from regression model) at naics4 level
    # Emp comes from sumdf, QWI comes from sum naics4 Emp in valdf
    mergedf['ModelTotal'] = mergedf['Emp'].astype(float) - mergedf['QWI_Emp']
    mergedf['MissingModel'] = mergedf['ModelTotal'] - mergedf['Model']
    
    valdf['stcnty'] = valdf['geoindkey'].str.replace(r'_.*', '', regex=True)
    valdf = pd.merge(valdf, mergedf, on='stcnty', how='outer', suffixes=('', '_agg'))
    valdf = valdf[valdf['m1empFromModel'].notna()]
    #if month 1 employment (Emp) summed over naics4 (from valdf) is 0 or NA
    #     then proposed m1emp is valdf m1emp (from regression model or given in QWI) 
    #        adjusted evenly to sum up to Emp in sumdf
    # otherwise
    #      proposed m1emp is valdf m1emp adjusted proportional to 
    #        the row's m1emp share of sum m1emp across county (sum from valdf m1emp)
    #        (i.e. ratio of valdf m1emp over sum m1emp across county)
    valdf['ProposedM1emp'] = np.where(
        (valdf['Model'] == 0) | valdf['Model'].isna(),
        valdf['m1emp'] + (valdf['MissingModel'] / valdf['CellCount_Model']),
        valdf['m1emp'] + (valdf['MissingModel'] * valdf['m1emp'] / valdf['Model'])
    )
    #update m1emp with proposed m1emp if... 
    #      m1emp is from regression model and 
    #      valdf has Emp for >=1 naics4 row in county group (not from regression model)
    valdf.loc[(valdf['m1empFromModel'] == 1) & valdf['QWI_Emp'].notna(), 'm1emp'] = valdf['ProposedM1emp']
    # Force 'm1emp' to be non-negative (set negative values to 0)
    valdf['m1emp'] = valdf['m1emp'].clip(lower=0)
    #return valdf
    return valdf['m1emp'].round(0)

    #return groupdf
