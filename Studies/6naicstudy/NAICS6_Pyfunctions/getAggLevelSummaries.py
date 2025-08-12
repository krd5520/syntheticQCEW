import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import statsmodels.api as sm


def get_codes_summary(dfin, groupbydigits=3, levelgrouped=4):
    '''
    What is the point?
        get_codes_summary() aggregates wage data (qp1) from detailed NAICS codes up to higher
        levels of aggregation, while tracking data availability and missingness patterns.
        
        For example: Can aggregate from 6-digit NAICS up to 3-digit sector level while
        preserving counts of available/missing wage values.
    Inputs:
        1. dfin - pd.DataFrame containing:
           - geoindkey: Composite geographic-industry keys (e.g., '01001_1111//')
           - qp1: Wage values from CBP data
           - qp1_nf: Wage suppression flags ('D' = suppressed)
        2. groupbydigits - Target aggregation level (default=3):
           - 2: Sector level (e.g., '31-33' Manufacturing)
           - 3: Subsector level
           - 4: Industry group level
        3. levelgrouped - Source data level (default=4):
           - Typically 4 or 6 digit NAICS codes being aggregated
    Returns:
        pd.DataFrame with columns:
        - geo[groupbydigits]naics: Composite geographic-aggregated industry keys
        - Count[levelgrouped]Codes: Number of original codes in each group  
        - wageCBP_sum[levelgrouped]by[groupbydigits]: Sum of available wages
        - wageCBP_missing[levelgrouped]by[groupbydigits]: Count of suppressed values
        - grouplevels: Metadata about aggregation level
    '''
    # Step 1: Define regex pattern to filter appropriate geoindkey values
    # Handles different NAICS code lengths (e.g., '01001_1111//' for 4-digit)
    pattern_grep = rf"_[0-9]{{{levelgrouped}}}[^0-9]{{{6 - levelgrouped}}}"
    if levelgrouped == 6:
        pattern_grep = r"_[0-9]{6}"
    # Step 2: Determine string positions for grouping keys
    # Creates keys like '01001_111' for groupbydigits=3
    str_end_idx = -(6 - groupbydigits)# - 1
    label_group = f"{levelgrouped}by{groupbydigits}"
    # Step 3: Filter and prepare dataframe
    df = dfin[dfin['geoindkey'].str.contains(pattern_grep, regex=True)].copy()
    df['geodignaics'] = df['geoindkey'].str[:str_end_idx]
    df = df[['geoindkey', 'geodignaics', 'state', 'cnty', 'estnum', 'qp1', 'qp1_nf']]
    # Step 4: Aggregate data by grouping key
    count6dig = df.groupby('geodignaics').agg(
        CountCodes=('geoindkey', 'count'),
        wageCBP=('qp1', lambda x: np.nansum(x.astype(float))),
        wageCBP_missing=('qp1_nf', lambda x: (x == 'D').sum())
    ).reset_index()
    count6dig['grouplevels'] = f"group{label_group}"
    # Step 5: Special handling for non-6-digit aggregations
    if levelgrouped != 6:
        count6dig.loc[count6dig['wageCBP_missing'] == count6dig['CountCodes'], 'wageCBP'] = np.nan
    count6dig = count6dig.rename(columns={
        'geodignaics': f'geo{groupbydigits}naics',
        'CountCodes': f'Count{levelgrouped}Codes',
        'wageCBP': f'wageCBP_sum{label_group}',
        'wageCBP_missing': f'wageCBP_missing{label_group}'
    })
    return count6dig
