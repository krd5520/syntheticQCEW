import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import statsmodels.api as sm
import sys
import os
from statsmodels.stats.outliers_influence import OLSInfluence
from formulaic import Formula
import patsy
import yaml
sys.path.append(os.path.abspath('./'))
from getAggLevelSummaries import *
from GeneralFunctions import *
with open('config.yaml','r') as configFile:
    config = yaml.safe_load(configFile)
    wageConfig = config['wageConfig']

def get_wage_model(df, emp_mat_adj): 
    '''
    What is the point?
        get_wage_model() creates an OLS model that predicts wage differences based on:
        - CBP wage data
        - Employment counts
        - Sector and state effects
    Inputs:
        1. df - pd.DataFrame containing wage and employment data
        2. emp_mat_adj - Adjusted employment matrix from employment functions
    Steps:
        1. Merge employment matrix with wage data
        2. Filter to valid observations (non-missing qp1_nf and wagediff)
        3. Transformations:
           - Square root of wage difference
           - Numeric conversion of key variables
        4. Initial model fitting using formula from config.yaml
        5. Outlier detection:
           - Cook's distance
           - Studentized residuals
        6. Refit model after removing outliers
    Configurable Parameters:
        - OLS_FORMULA in config.yaml specifies model formula
        - OUTLIER_THRESH controls residual-based outlier removal
    Returns:
        model - statsmodels.OLS fitted model for wage prediction
    '''
    # Merge employment data with wage data
    emp_mat_adj = pd.DataFrame(emp_mat_adj)
    emp_mat_adj = emp_mat_adj.apply(lambda col: pd.to_numeric(col) if 'm' in col.name else col)
    wagedf4 = pd.concat([
        df.drop(columns=['geoindkey']),  
        emp_mat_adj
    ], axis=1)
    # Filter nans
    wagedf4 = wagedf4[(wagedf4['qp1_nf'].notna()) & (wagedf4['qp1_nf'] != 'D') & (wagedf4['wagediff'].notna())]
    # sqrt_wagediff is, uh, well, the sqrt of wagediff
    # Also type conversions
    wagedf4["sqrt_wagediff"]=wagedf4["wagediff"].astype(float) ** 0.5
    wagedf4["wageCBP_missing6by4"]=wagedf4["wageCBP_missing6by4"].astype(float)
    wagedf4["m3emp"]=wagedf4["m3emp"].astype(float)
    # Define formula and fit initiial model
    formula = wageConfig['OLS_FORMULA']
    y_pre, X_pre = Formula(formula).get_model_matrix(wagedf4)
    model_pre = sm.OLS(y_pre, X_pre).fit()
    # Calculate Cook's Distance and Studentized Residuals
    influence = OLSInfluence(model_pre)
    cooks_d=influence.cooks_distance[0]
    student_resid = influence.resid_studentized_internal
    # Identify Outliers based on thresholds
    outliers = [i for i, r in enumerate(student_resid) if r > wageConfig['OUTLIER_THRESH']]
    influential_indices = [i for i, d in enumerate(cooks_d) if d > 1]
    if influential_indices:
        print("# of rows filtered due to influence (Cook's Distance):", len(influential_indices), '|', np.round((len(influential_indices)/len(wagedf4)),3) * 100, '%')
    if outliers:
        print("# of outliers filtered (Studentized Residuals):", len(outliers), '|', np.round((len(outliers)/len(wagedf4)),3) * 100, '%')
    # Remove outliers
    rows_to_drop = influential_indices + outliers
    wagedf4 = wagedf4.drop(wagedf4.index[rows_to_drop])
    # Refit
    y,X = Formula(formula).get_model_matrix(wagedf4)
    model = sm.OLS(y,X).fit()
    return model

def get_wagemax(codes4naics, fulldf):
    '''
    What is the point?
        get_wagemax() calculates upper bounds for wages by:
        1. First trying 3-digit NAICS level data
        2. Falling back to 2-digit sector level if needed
        3. Using county-wide totals as last resort
    Inputs:
        1. codes4naics - Array of 4-digit NAICS codes
        2. fulldf - Complete dataset with wage information
    Returns:
        DataFrame with geoindkey and calculated maxwage values
    '''
    # Initialize the output dataframe
    outdf = pd.DataFrame({
        "geoindkey": codes4naics,
        "maxwage": np.nan,
        "geo3naics": codes4naics.str[:-3],
        "geo2naics": codes4naics.str[:-4],
        "geography": codes4naics.str[:-7]
    })
    fulldf['qp1'] = fulldf['qp1'].astype(int)
    # Try 3-digit NAICS level first
    tomergedf3 = fulldf[
        fulldf["geoindkey"].str.contains(r"_[0-9]{3}[^0-9]{3}", regex=True)
    ].copy()
    tomergedf3["geo3naics"] = tomergedf3["geoindkey"].str[:-3]
    tomergedf3["wage_naics3"] = np.where(tomergedf3["qp1_nf"] == "D", np.nan, tomergedf3["qp1"])
    tomergedf3 = tomergedf3[["geo3naics", "estnum", "wage_naics3"]]
    # Merge 3-digit data
    outdf = outdf.merge(tomergedf3, on='geo3naics', how='left', suffixes=('', '_naics3'))
    # For missing values, try 2-digit sector level
    notmaxcodes = outdf[outdf['wage_naics3'].isna()]['geo2naics'].tolist()
    fulldf['geo2naics'] = fulldf['geoindkey'].str[:-4]
    tomergedf2 = fulldf[fulldf['geo2naics'].isin(notmaxcodes) & 
                        fulldf['geoindkey'].str.contains(r"_[0-9]{2}[^0-9]{4}")].copy()
    tomergedf2['wage_naics2'] = np.where(tomergedf2['qp1_nf'] == 'D', np.nan, tomergedf2['qp1'])
    tomergedf2 = tomergedf2[['geo2naics', 'estnum', 'wage_naics2']]
    # Calculate differences between sector and summed 3-digit wages
    tomergedf3['wage_naics3'] = tomergedf3['wage_naics3'].astype(float)
    tomergedf3['geo2naics'] = tomergedf3['geo3naics'].str[:-1]  # Extract sector codes
    tomergedf3 = tomergedf3.groupby('geo2naics', as_index=False).agg(sumwage3=('wage_naics3', 'sum'))
    tomergedf2 = tomergedf2.merge(tomergedf3, on='geo2naics', how='left')
    tomergedf2['missing_wage_naics2'] = tomergedf2['wage_naics2'].astype(float) - tomergedf2['sumwage3']
    tomergedf2 = tomergedf2[['geo2naics', 'missing_wage_naics2', 'estnum', 'wage_naics2']]
    # Merge sector-level data
    outdf = outdf.merge(tomergedf2, on='geo2naics', how='left', suffixes=('', '_naics2'))
    outdf['maxwage'] = outdf.apply(lambda row: row['wage_naics3'] if pd.notna(row['wage_naics3']) else row['missing_wage_naics2'], axis=1)
    outdf = outdf.drop(columns=['wage_naics2'])
    fulldf['qp1']=fulldf['qp1'].astype(int)
    # For remaining missing values, use county-wide totals
    max_allind_allcounty = fulldf[fulldf['ind_level'] == '4']['qp1'].max(skipna=True)
    notmaxcodes = outdf[outdf['maxwage'].isna()]['geography'].tolist()
    tomergedfall = fulldf.copy()
    tomergedfall['geography'] = tomergedfall['geoindkey'].str[:-7]
    tomergedfall = tomergedfall[tomergedfall['geography'].isin(notmaxcodes)]
    tomergedfall = tomergedfall[tomergedfall['geoindkey'].str.contains('_------')]
    tomergedfall['wageall'] = tomergedfall.apply(
        lambda row: max_allind_allcounty if row['qp1_nf'] == 'D' else row['qp1'], axis=1)
    tomergedfall = tomergedfall[['geography', 'estnum', 'wageall']]
    # Calculate county-level differences
    tomergedf2['geography'] = tomergedf2['geo2naics'].str[:-3]
    tomergedf2 = tomergedf2.groupby('geography', as_index=False)['wage_naics2'].sum(min_count=1)
    tomergedf2.rename(columns={'wage_naics2': 'sumwage2'}, inplace=True)
    tomergedfall = tomergedfall.merge(tomergedf2, on="geography", how="left")
    tomergedfall['missingwageall'] = tomergedfall['wageall'].astype(float) - tomergedfall['sumwage2'].astype(float)
    tomergedfall = tomergedfall[['geography', 'missingwageall', 'estnum', 'wageall']]
    # Final merge and return
    outdf = outdf.merge(tomergedfall, on="geography", how="left", suffixes=("", "_allindustry"))
    outdf['maxwage'] = outdf.apply(lambda row: row['missingwageall'] if pd.isna(row['maxwage']) else row['maxwage'], axis=1)
    outdf = outdf[['geoindkey', 'maxwage']]
    return outdf

def get_wagemin(codes4naics, fulldf):
    '''
    What is the point?
        get_wagemin() calculates lower bounds for wages using 6-digit NAICS summaries
    Inputs:
        1. codes4naics - Array of 4-digit NAICS codes
        2. fulldf - Complete dataset with wage information
    Returns:
        DataFrame with geoindkey and calculated minwage values
    '''
    # Get 6-digit NAICS summaries
    tomerge6dig = get_codes_summary(dfin=fulldf, groupbydigits=4, levelgrouped=6)
    # Create minwage column (0 if no data available)
    tomerge6dig['minwage'] = np.where(tomerge6dig['wageCBP_sum6by4'].isna(), 0, tomerge6dig['wageCBP_sum6by4'])
    tomerge6dig['geoindkey'] = tomerge6dig['geo4naics'].astype(str) + "//"
    tomerge6dig = tomerge6dig[['geoindkey', 'minwage']]
    return tomerge6dig

def get_maxmindf(df4dig, fulldf, emp_mat_adj):
    '''
    What is the point?
        get_maxmindf() combines wage data with min/max bounds
    Inputs:
        1. df4dig - 4-digit NAICS level data
        2. fulldf - Complete dataset
        3. emp_mat_adj - Adjusted employment matrix
    Returns:
        DataFrame with original data plus minwage and maxwage columns
    '''
    # Merge employment data with wage data
    emp_mat_adj = pd.DataFrame(emp_mat_adj)
    emp_mat_adj = emp_mat_adj.apply(lambda col: pd.to_numeric(col) if 'm' in col.name else col)
    wagedf4 = pd.concat([
        df4dig.drop(columns=['geoindkey']),  # Drop geoindkey column
        emp_mat_adj
    ], axis=1)
    # Get min and max wage bounds
    minwagedf = get_wagemin(codes4naics = wagedf4['geoindkey'],fulldf=fulldf)
    maxwagedf = get_wagemax(codes4naics = wagedf4['geoindkey'],fulldf=fulldf)
    # Merge all data
    wagedf4_maxmin = wagedf4.merge(maxwagedf, on="geoindkey", how="left") \
                            .merge(minwagedf, on="geoindkey", how="left")
    wagedf4_maxmin['minwage'] = wagedf4_maxmin['minwage'].fillna(0)
    return wagedf4_maxmin

def wagesFromModel(df, modelwage):
    '''
    What is the point?
        wagesFromModel() generates wage predictions from the fitted model
        with random noise based on prediction uncertainty
    Inputs:
        1. df - DataFrame with predictor variables
        2. modelwage - Fitted statsmodels OLS model
    Returns:
        Array of predicted wage values
    '''
    # Get predictions and standard errors
    pred, se_fit = custom_predict(df, modelwage)
    # Generate random normal values and square them
    wagefit = np.random.normal(loc=pred, scale=se_fit) ** 2
    return wagefit

def wagesFromQWI(df, empmat):
    '''
    What is the point?
        wagesFromQWI() calculates wages using QWI earnings data
    Inputs:
        1. df - DataFrame with QWI earnings data
        2. empmat - Employment matrix
    Returns:
        Array of calculated wage values
    '''
    # Multiply employment by earnings
    empmat = empmat.astype(float)
    earnbeg = df['EarnBeg'].astype(float)
    wages = np.sum(empmat, axis=1) * earnbeg.values
    return wages


def adjust_wagevalues(fitwagedf, dfmaxmin):
    '''
    What is the point?
        adjust_wagevalues() constrains wage estimates to stay within min/max bounds
    Inputs:
        1. fitwagedf - DataFrame with wage estimates
        2. dfmaxmin - DataFrame with min/max bounds
    Returns:
        DataFrame with adjusted wage values
    '''
    # Merge with min/max bounds
    maxmindf = dfmaxmin[['geo4naics', 'minwage', 'maxwage']].copy()
    fitwagedf = fitwagedf.merge(maxmindf, on='geo4naics', how='left')
    # Apply constraints
    fitwagedf['wage'] = fitwagedf['wage'].clip(
        lower=fitwagedf['minwage'].astype(float),
        upper=fitwagedf['maxwage'].astype(float)
    )
    fitwagedf = fitwagedf.drop(columns=['minwage', 'maxwage'])
    return fitwagedf

def get_wages4(df4, empmat, wagemodel, useEarnQWI=False, count6digdf=None, maxmindf=None):
    '''
    What is the point?
        get_wages4() is the main function for wage estimation that:
        1. Uses CBP data when available
        2. Falls back to QWI earnings data when configured
        3. Uses model predictions as last resort
    Inputs:
        1. df4 - 4-digit NAICS level data
        2. empmat - Employment matrix
        3. wagemodel - Fitted wage prediction model
        4. useEarnQWI - Whether to use QWI earnings data
        5. count6digdf - 6-digit NAICS data (optional)
        6. maxmindf - Min/max bounds (optional)
    Returns:
        DataFrame with estimated wages and method indicators
    '''
    # First try using CBP data
    qp1Available = df4['qp1_nf'] != 'D'
    df4['wage'] = df4['qp1']
    useModel = ~qp1Available  
    # Optionally use QWI earnings data 
    if useEarnQWI:
        print("Number which uses QWI:", end=" ")
        EarnBegAvailable = df4['sEarnBeg'].astype(float) == 1.0
        useQWI = (EarnBegAvailable) & (~qp1Available)
        print(sum(useQWI))
        df4.loc[useQWI, 'wage'] = wagesFromQWI(df4[useQWI], empmat[useQWI])
        useModel = ~EarnBegAvailable & ~qp1Available
    # Convert empmat to DataFrame 
    empmat_df = pd.DataFrame(empmat, columns=['m1emp', 'm2emp', 'm3emp'])
    # Use model for remaining cases
    df4 = pd.concat([df4, empmat_df], axis=1)
    print("Number of cells which use Model:", sum(useModel))
    predicted_wages = wagesFromModel(df4[useModel], wagemodel)
    df4.loc[useModel, 'wage'] = predicted_wages + df4.loc[useModel, 'wageCBP_sum6by4'].values
    # Add method indicators
    df4['usewageModel'] = useModel.astype(int)
    if useEarnQWI:
        df4.loc[useQWI, 'usewageModel'] = 2
    # Apply bounds, round, and return
    df4 = adjust_wagevalues(df4,maxmindf)
    df4['wage'] = df4['wage'].round()
    return df4
