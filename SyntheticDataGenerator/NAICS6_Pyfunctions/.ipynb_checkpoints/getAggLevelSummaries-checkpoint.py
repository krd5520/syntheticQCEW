import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import statsmodels.api as sm


def get_codes_summary(dfin, groupbydigits=3, levelgrouped=4):
    #pattern to subset data to
    pattern_grep = rf"_[0-9]{{{levelgrouped}}}[^0-9]{{{6 - levelgrouped}}}"
    if levelgrouped == 6:
        pattern_grep = r"_[0-9]{6}"
    
    #character to end substring on to get [county#]_[industry at group-by digits level]
    str_end_idx = -(6 - groupbydigits)# - 1
    #to label the type of groupingh
    label_group = f"{levelgrouped}by{groupbydigits}"
    
    # Filter and transform dataframe
    df = dfin[dfin['geoindkey'].str.contains(pattern_grep, regex=True)].copy()#subset to levelgrouped rows
    df['geodignaics'] = df['geoindkey'].str[:str_end_idx]#get groupby vals
    df = df[['geoindkey', 'geodignaics', 'state', 'cnty', 'estnum', 'qp1', 'qp1_nf']]
    # Grouping by geodignaics
    count6dig = df.groupby('geodignaics').agg(     #group by specified level
        CountCodes=('geoindkey', 'count'), #count of rows in group
        wageCBP=('qp1', lambda x: np.nansum(x.astype(float))),#sum available wage in group
        wageCBP_missing=('qp1_nf', lambda x: (x == 'D').sum())#count rows w/out wage val in group
    ).reset_index()#label group-by & what level grouped
    
    # Assign grouping label
    count6dig['grouplevels'] = f"group{label_group}"
    
    # Set wageCBP to NaN if all values in the group are missing
    if levelgrouped != 6:#if we are not grouping 6 digit NAICS
        #replace sum available wage with NA when all rows are missing values
        count6dig.loc[count6dig['wageCBP_missing'] == count6dig['CountCodes'], 'wageCBP'] = np.nan
    
    # Rename columns
    count6dig = count6dig.rename(columns={
        'geodignaics': f'geo{groupbydigits}naics',
        'CountCodes': f'Count{levelgrouped}Codes',
        'wageCBP': f'wageCBP_sum{label_group}',
        'wageCBP_missing': f'wageCBP_missing{label_group}'
    })
    #return summary df
    return count6dig
