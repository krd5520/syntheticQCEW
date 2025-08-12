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
from getAggLevelSummaries import *



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
                    x['m3emp']
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
    row2 = sub6['m3emp'].values.copy()
    if sum(row2) == 0:
        row2 = np.repeat(1, len(row2))
    return np.maximum(row2.astype(float), 1e-10)

def dirichletparams_wage(sub6):
    ''' 
    Prepares parameters for Dirichlet distribution used in wage imputation.
    Similar to dirichletparams_m1emp but specifically for wage distribution.
    '''
    row1 = sub6['m3emp'].values.copy()
    if sum(row1) == 0:  
        row1 = np.repeat(1, len(row1))
    return np.maximum(row1.astype(float), 1e-10)

def get_m1emp6_per4(df,df4n,rseed=None):
    '''
    Distributes month 1 employment from 4-digit to 6-digit NAICS level using
    random proportional allocation based on Dirichlet distribution.
    '''
    if rseed is not None:
        np.random.seed(rseed)
    # Get Dirichlet parameters based on m3emp distribution
    m1empparams=dirichletparams_m1emp(df)
    # Generate random proportions
    rprops = np.random.dirichlet(m1empparams, size=1)
    if len(df) > 1:  # Multiple 6-digit codes
        # Split the m1emp value proportionally using rprops
        df['m1emp'] = np.round(rprops.flatten() * df4n['m1emp'].values[0]).astype(float)
    elif len(df) == 1:  # Single 6-digit code
        df['m1emp'] = df4n['m1emp'].values[0]
    return df
    
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
    remain_wage = float(subdf4['wage'].sum() - subdf6['qp1'].astype(float).sum())
    # Handle negative remainders (data consistency check)
    if not np.isnan(remain_wage):
        if float(remain_wage) < 0:
            print("WARNING: remainders are negative!")
            codes = ','.join(subdf6['geoindkey'].astype(str).tolist())
            print(f"Remainders: wage {float(remain_wage)} Codes: {codes}")
    subdf6['wage'] = subdf6['qp1'] # Start with known values
    unknown_indic = (subdf6['qp1_nf'] == 'D')
    # Distribute remaining wage to suppressed entries
    if len(subdf6[unknown_indic]) == 1:
        subdf6.loc[unknown_indic, 'wage'] = remain_wage
    else:
        subdf6unknown = subdf6[unknown_indic]
        rprop = np.random.dirichlet(dirichletparams_wage(sub6=subdf6unknown), size=1)
        mask = (subdf6['qp1_nf'] == 'D') & (~subdf6['qp1_nf'].isna())
        subdf6.loc[mask, 'wage'] = np.round(remain_wage * rprop.flatten()[:sum(mask)])
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
    subdf6emp = get_m1emp6_per4(df=subdf6,df4n=subdf4,rseed=rseed)
    # Step 2: Wage imputation (with fallback for empty groups)
    if len(subdf6emp) == 0:
        subdf6wage = subdf6.copy()
        subdf6wage['m1emp'] = np.nan
        subdf6wage['m3emp'] = np.nan
        subdf6wage['wage'] = np.nan
    else:
        subdf6wage = get_wage6_per4(subdf6=subdf6emp,subdf4=subdf4)
    # Cleanup before returning
    subdf6wage = subdf6wage.drop(columns=['qp1', 'qp1_nf', 'emp', 'geo5naics'])
    return subdf6wage

def process_chunk(x, df6_toget, df4n):
    '''Wrapper function for parallel processing'''
    return get_6naics_per4(x, df6=df6_toget, df4imp=df4n)

def get_6naics_all(df, df4n, codes4summary, rseed=None):
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
    codesNOTtoget = codes4summary['geo4naics'][codes4summary['Count6Codes'] == 1]
    df4forjoin = df4n[['geo4naics', 'm1emp', 'm3emp', 'wage']].copy()
    df6_onecodeper4 = (
        df[df['geo4naics'].isin(codesNOTtoget)]
        .merge(df4forjoin, on='geo4naics', how='inner')
        [['geoindkey', 'geo4naics', 'state', 'cnty', 'estnum', 'm1emp', 'm3emp', 'wage']]
    )
    # Prepare complex cases for parallel processing
    df6_toget = get_m3emp6_all(
        df=df[~df['geo4naics'].isin(codesNOTtoget)],
        df4n=df4n
    )
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


