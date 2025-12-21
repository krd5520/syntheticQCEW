import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from tqdm import tqdm
from multiprocessing import Pool
import statsmodels.api as sm
import time
import sys
import os

sys.path.append(os.path.abspath('./'))
from GeneralFunctions import *
from hierarchy_geoindkey import *



def get_m3emp6_all(df,df4n):
    '''
    What is the point?
        Scales month 3 employment (m3emp) from 4-digit to 6-digit NAICS level using
        EmpScale factors that account for establishment size differences.
    Why is this needed?
        - Maintains proportional relationships between detailed industries
        - Handles edge cases (infinite scale factors, missing values)
        - Prepares base values for subsequent month 1 imputation
    Inputs:
        1. df - DataFrame of 6-digit NAICS data
        2. df4n - DataFrame of 4-digit NAICS data containing:
           - geoindkey: Geographic-industry keys
           - EmpScale: Scaling factors for employment distribution
    Steps:
        1. Extract 4-digit NAICS codes from geoindkey
        2. Merge with scaling factors
        3. Calculate scaled employment values
        4. Handle special cases:
           - Infinite scaling factors → set to 1
           - Missing scaling factors → use raw employment counts
    '''
    sub4df = df4n.copy()
    sub4df['geo4naics'] = df4n['geoindkey'].str[:-2]
    sub4df = sub4df[['geo4naics', 'EmpScale']]
    dfout = df.merge(sub4df, on='geo4naics', how='inner') \
          .assign(m3emp=lambda x: round(x['emp'].astype(float) / x['EmpScale']))
    dfout = (
        dfout
        .assign(
            m3emp=lambda x: np.where(
                x['EmpScale'] == float('inf'),
                1,
                np.where(
                    x['EmpScale'].isna(),
                    x['emp'],
                    x['emp3']
                )
            )
        )
        .drop(columns=['EmpScale'])
    )
    return dfout

def dirichletparams_m1emp(sub6):
    '''
    Prepares parameters for Dirichlet distribution used in employment imputation.
    Handles zero-sum cases by returning uniform distribution parameters.
    '''
    row2 = sub6['emp3'].values.copy()
    if sum(row2) == 0:
        row2 = np.repeat(1, len(row2))
    return np.maximum(row2.astype(float), 1e-10)

def dirichletparams_wages(sub6):
    ''' 
    Prepares parameters for Dirichlet distribution used in wage imputation.
    Similar to dirichletparams_m1emp but specifically for wage distribution.
    '''
    row1 = sub6['emp3'].values.copy()
    if sum(row1) == 0:  
        row1 = np.repeat(1, len(row1))
    return np.maximum(row1.astype(float), 1e-10)

def get_m1emp6_per4(df6n,df4n,rseed=None):
    '''
    Distributes month 1 employment from 4-digit to 6-digit NAICS level using
    random proportional allocation based on Dirichlet distribution.
    '''
    if rseed is not None:
        np.random.seed(rseed)
        # Calculate remaining wage after accounting for known values
    remain_emp1 = float(df4n['emp1'].sum() - df6n['emp1'].astype(float).sum())

    #remain_emp1 = float(df4n['emp1'].sum() - df['emp1'].astype(float).sum())
    # Handle negative remainders (data consistency check)
    if not np.isnan(remain_emp1):
        if float(remain_emp1) < 0:
            print("WARNING: remainders are negative!")
            codes = ','.join(df6n['geoindkey'].astype(str).tolist())
            print(f"Remainders: emp1 {float(remain_emp1)} Codes: {codes}")
            print(f'Head of df4n\n {df4n[["geoindkey","minemp1","minemp1_source","emp1","emp1_source"]].head()}')
            print(f'Head of df6n\n {df6n[["geoindkey","minemp1","minemp1_source","emp1","emp1_source"]].head()}')
    # subdf6['wages'] = subdf6['q'] # Start with known values
    unknown_indic = (df6n['emp1'].isna())
    # Distribute remaining wage to suppressed entries
    if len(df6n[unknown_indic]) == 1:
        df6n.loc[unknown_indic, 'emp1'] = remain_emp1
        df6n.loc[unknown_indic, 'emp1_source'] = "remainder"
    else:
        subdf6unknown = df6n[unknown_indic]
        rprop = np.random.dirichlet(dirichletparams_m1emp(sub6=subdf6unknown), size=1)
        mask = (df6n['emp1'].isna())  # & (~subdf6['qp1_nf'].isna())
        df6n.loc[mask, 'emp1'] = np.round(remain_emp1 * rprop.flatten()[:sum(mask)])
        df6n.loc[mask, 'emp1_source'] = "dirichlet_divider"
    return df6n
    
def get_wage6_per4(subdf6,subdf4,rseed=None):
    '''
    Distributes wage values from 4-digit to 6-digit NAICS level, handling:
    - Known wage values (qp1)
    - Suppressed values (qp1_nf = 'D')
    - Negative remainders (with warnings)
    '''
    if rseed is not None:
        np.random.seed(rseed)
    # Calculate remaining wage after accounting for known values
    remain_wage = float(subdf4['wages'].sum() - subdf6['wages'].astype(float).sum())
    # Handle negative remainders (data consistency check)
    if not np.isnan(remain_wage):
        if float(remain_wage) < 0:
            print("WARNING: remainders are negative!")
            codes = ','.join(subdf6['geoindkey'].astype(str).tolist())
            print(f"Remainders: wage {float(remain_wage)} Codes: {codes}")
            print(f"subdf4 wages len {subdf4['wages'].shape[0]} sum {subdf4['wages'].sum()} (max, min) {subdf4['maxwages'].values}, {subdf4['minwages'].values}, wages source {subdf4['wages_source'].values}, subdf6 wages sum {subdf6['wages'].astype(float).sum()} \n subdf6 {subdf6[['geoindkey','wages','wages_source']]}")
    #subdf6['wages'] = subdf6['q'] # Start with known values
    unknown_indic = (subdf6['wages'].isna())
    # Distribute remaining wage to suppressed entries
    if len(subdf6[unknown_indic]) == 1:
        subdf6.loc[unknown_indic, 'wages'] = remain_wage
        subdf6.loc[unknown_indic, 'wages_source'] = "remainder"
    else:
        subdf6unknown = subdf6[unknown_indic]
        rprop = np.random.dirichlet(dirichletparams_wages(sub6=subdf6unknown), size=1)
        mask = (subdf6['wages'].isna())# & (~subdf6['qp1_nf'].isna())
        subdf6.loc[mask, 'wages'] = np.round(remain_wage * rprop.flatten()[:sum(mask)])
        subdf6.loc[mask, 'wages_source'] = "dirichlet_divider"
    return subdf6

def get_6naics_per4(naics4dig,df6,df4imp,rseed=None):
    '''
    Core function that processes a single 4-digit NAICS code to:
    1. Distribute employment values
    2. Distribute wage values
    3. Clean output columns
    '''
    if rseed is not None:
        np.random.seed(rseed)
    subdf6 = df6[df6['geo4naics'] == naics4dig].copy()
    subdf4 = df4imp[df4imp['geo4naics'] == naics4dig].copy()
    # Step 1: Employment imputation
    subdf6emp = get_m1emp6_per4(df6n=subdf6,df4n=subdf4,rseed=rseed)
    # Step 2: Wage imputation (with fallback for empty groups)
    if len(subdf6emp) == 0:
        subdf6wage = subdf6.copy()
        subdf6wage['emp1'] = np.nan
        subdf6wage['emp3'] = np.nan
        subdf6wage['wages'] = np.nan
    else:
        subdf6wage = get_wage6_per4(subdf6=subdf6emp,subdf4=subdf4)
    # Cleanup before returning
    #subdf6wage = subdf6wage.drop(columns=['qp1', 'qp1_nf', 'emp', 'geo5naics'],errors="ignore")
    return subdf6wage

def process_chunk(x, df6_toget, df4n):
    '''Wrapper function for parallel processing'''
    return get_6naics_per4(x, df6=df6_toget, df4imp=df4n)

def get_6naics_all(df, df4n, codes4summary, rseed=None,keepqcew=True):
    '''
    Main function that coordinates full 6-digit NAICS imputation:
    1. Separates simple cases (1:1 mappings)
    2. Processes complex cases in parallel
    3. Combines all results
    '''
    if rseed is not None:
        np.random.seed(rseed)
    timestart1 = time.time()
    # Handle simple cases (1 6-digit code per 4-digit)
    codesNOTtoget = codes4summary['geo4naics'][codes4summary['count6by4codes'] == 1]
    df4forjoin = df4n[['geo4naics', 'emp1','emp2', 'emp3', 'wages','estnum']].copy()

    df6_onecodeper4 = (
        df[df['geo4naics'].isin(codesNOTtoget)]
        .merge(df4forjoin, on='geo4naics', how='inner',suffixes=["_naics6",""])
    )
    ## Check agreement
    check_diff_emp3=df6_onecodeper4['emp3']-df6_onecodeper4['emp3_naics6']
    check_diff_wages=df6_onecodeper4['wages']-df6_onecodeper4['wages_naics6']
    check_diff_wages[df6_onecodeper4['wages_naics6'].isna()]=0
    check_diff=check_diff_emp3+check_diff_wages
    if check_diff.round(0).sum()!=0:
        print(f"checking agreement on countyXnaics4 and countyXnaics6 when there is only 1 naics6 code in the naics4\n{df6_onecodeper4.loc[check_diff.round(0)!=0,['geoindkey', 'geo4naics', 'state', 'cnty', 'estnum','estnum_naics6', 'emp1','emp1_naics6','emp2','emp2_naics6', 'emp3','emp3_naics6', 'wages','wages_naics6']].head()}")
    df6_onecodeper4=df6_onecodeper4[['geoindkey', 'geo4naics', 'state', 'cnty', 'estnum', 'emp1','emp2', 'emp3', 'wages']]

    ##Check all countyXnaics4 codes have countyXnaics6 subcodes
    geo4naics_codes=df4n['geoindkey'].str.slice(stop=-2)
    df6_geo4naics_codes=df.loc[df['agglvl_code']==78,'geo4naics']
    n_geo4naics=geo4naics_codes.nunique()
    n_df6_geo4naics=df6_geo4naics_codes.nunique()
    if n_geo4naics!=n_df6_geo4naics:
        print(f"Number of unique geo4naics codes do not align. In countyXnaics4 {n_geo4naics} in countyXnaics6 {n_df6_geo4naics}.")
        print(df6_geo4naics_codes[~df6_geo4naics_codes.isin(geo4naics_codes)].head())
    raise Exception("stop here")
    # Prepare complex cases for parallel processing
    df6_toget = df[~df['geo4naics'].isin(codesNOTtoget)]
    #get_m3emp6_all(
    #    df=df[~df['geo4naics'].isin(codesNOTtoget)],
    #    df4n=df4n
    #)
    test4dig = df6_toget['geo4naics'].unique()
    print(f"Execution time: {time.time() - timestart1:.4f} seconds")
    # Parallel processing setup
    args = [(x, df6_toget, df4n) for x in test4dig]
    with Pool(processes=3) as pool:
        args = [(x, df6_toget, df4n) for x in test4dig]
        results = pool.starmap(process_chunk, args)
    # Combine results
    df6_toget_imputed = pd.concat(results, ignore_index=True)
    combined_df = pd.concat([df6_toget_imputed, df6_onecodeper4], ignore_index=True)
    return combined_df


