import numpy as np
import pandas as pd
import os
import yaml
from typing import Optional, Tuple, List, Dict


"""
Geographic and Industry Hierarchy Functions

This module provides utilities for working with the geoindkey identifier system, which encodes
both geographic (state/county FIPS codes) and industry (NAICS codes) information in a single
composite key. Functions handle parsing, aggregation, and validation of these hierarchical
relationships across different geographic and industry levels.

The geoindkey format is: [geography]_[industry]
  - geography: State code (2 digits) + County code (3 digits, zero-padded)
  - industry: NAICS code (2-6 digits) padded with "-" or "/" for aggregation levels
"""



## takes CBP format of fips state and county codes as 2 columns to
## the QWI format of [state code] + [3 characters: leading zeros and county code]
# INPUT: pandas dataframe with columns 'fipstate' & 'fipscty' for state & county codes
# OUTPUT: pandas series of QWI format geography
def fips_to_geography(df):
    # combine fipstate and fipscty to match geography format of QWI files.
    # state + leading zeros to make county number 3 characters
    df[['fipstate', 'fipscty']] = df[['fipstate', 'fipscty']].astype(int).astype(str)
    df['fipscty'] = df['fipscty'].str.zfill(3)
    return(df['fipstate'] + df['fipscty'])

## gets state, cnty, geography, industry, and ind_level from geoindkey
def fill_from_geoindkey(data,
                        numeric_ind_level=True,
                        naics_xwalk=None,
                        naicsdata=None):
    """
        Parse geoindkey identifier into geographic and industry components.

        Extracts state, county, geography, industry, and aggregation level information
        from the composite geoindkey field. Optionally merges with NAICS classification
        data for sector and supersector information.

        Args:
            data (pd.DataFrame):
                Input DataFrame containing 'geoindkey' column.
                Format: '[state][county]_[naics_code]' where industry is padded with "-" or "/"

            numeric_ind_level (bool):
                If True, ind_level is numeric (0-6 digits) and agglvl_code (72-78) is calculated.
                If False, ind_level is categorical ('A', 'S', etc.).
                Default is True.

            naics_xwalk (pd.DataFrame or str, optional):
                NAICS crosswalk table or filepath containing sector/supersector mappings.
                If string, treated as filepath to crosswalk file.
                If provided, merges columns: ['naics2', 'supersector', 'domain']
                Default is None.
            naicsdata (pd.DataFrame, optional):
                row for each naics6 code and
                columns for the corresponding naics5,4,3,2 levels
                default for 2016 data is used

        Returns:
            pd.DataFrame:
                Input DataFrame with added columns:
                - geography: State + County (5 characters)
                - state: State FIPS code (2 characters)
                - cnty: County FIPS code (3 characters)
                - industry: Industry code extracted from geoindkey
                - ind_level: Industry detail level (0-6 digits, or categorical)
                - agglvl_code: Aggregation level code (72-78, if numeric_ind_level=True)
                - naics2/3/4/5/6: NAICS codes at each level (if numeric_ind_level=True)
                - supersector, domain: From NAICS crosswalk (if provided)

        Note:
            - Modifies input DataFrame in-place
            - Automatically maps 2-digit aggregations: 31↔32↔33, 44↔45, 48↔49 (2-digit groupings)
        """
    #split geoindkey into geography and industry portions
    expandgeoind=data['geoindkey'].str.split('_',expand=True)
    if len(expandgeoind.columns)>2:
        raise Exception(f'geoindkey format error. There should only be 1 underscore: {data.loc[expandgeoind.iloc[:,3] is not None,"geoindkey"].head()}')
    data['geography']=expandgeoind.iloc[:,0]
    data.loc[:,'industry'] = expandgeoind.iloc[:, 1]

    #get county and state from geographu
    data['state']=data['geography'].astype(str).str.slice(start=0,stop=-3)
    data['cnty']=data['geography'].astype(str).str.slice(start=-3)

    #get ind_level
    ninddig=data['industry'].str.count(r'\d')
    if numeric_ind_level:
        data['ind_level'] = ninddig.astype(int)
        ninddig[ninddig==0]=-1
        data['agglvl_code']=72+ninddig
        for i in [2, 3, 4, 5]: #naics2, 3, 4, and 5 digits
            data["naics"+str(i)] = ""
            data.loc[data['ind_level']>=i,"naics"+str(i)]=data.loc[data['ind_level']>=i,"industry"].str.slice(start=0,stop=i)
    else:
        ninddig=ninddig.astype(str)
        ninddig[ninddig=='0']="A"
        ninddig[ninddig == '2'] = "S"
        data['ind_level']=ninddig

    # Optionally merge NAICS classification data (sectors, supersectors)
    if naics_xwalk is not None:
        containsdash = data['naics2'].astype(str).str.contains("-").sum()
        if isinstance(naics_xwalk,str):
            if containsdash>0:
                naics_xwalk=get_xwalk_naics(naics_xwalk,expand_naics2=False)
            else:
                naics_xwalk=get_xwalk_naics(naics_xwalk,expand_naics2=True)
            naics_xwalk.rename(columns={"naics_sector":"naics2","super_sector":"supersector"},inplace=True, errors="ignore")
        data=data.merge(naics_xwalk[['naics2','supersector','domain']],on='naics2',how='left')

    if naicsdata is not None:
        dashdict = {"31": [31, 32, 33],"44": [44, 45],"48": [48, 49]}
    else:
        dashdict = {}
        for dash2 in naicsdata.loc[
            (naicsdata["dashed_naics2"]) & (naicsdata["ind_level"] == 2), "naics2"].to_list():
            dashdict[str(dash2)] = list(naicsdata.loc[(naicsdata["naics2"].astype(str) == str(dash2)) &
                                                      (naicsdata["ind_level"] == 3), "naics3"].astype(str).str.slice(stop=2).unique())
    for key, value in dashdict.items():
        data["naics2"] = data["naics2"].str.replace("|".join(["_" + str(val) for val in value]),
                                                    "_" + str(key), regex=True)

    return data

def count_notna(x):
    return x.notna().sum()

def get_codes_summary(dfin, groupbydigits=3, levelgrouped=4,variable="wages",
                      include_source=True,onlyQCEW=False,
                      perestab_stats=False,naicsdf=None,
                      include_estab_emp3_stats=True):
    """
    Aggregate detailed industry codes up to a higher aggregation level.

    Summarizes wage, employment, or establishment data from detailed NAICS codes
    (e.g., 6-digit) up to broader categories (e.g., 3-digit sector) while tracking
    data availability and source information.

    PURPOSE:
        Enables hierarchical aggregation (e.g., from 6-digit NAICS to 3-digit sector)
        while preserving information about data quality, suppression, and source.

    WORKFLOW:
        1. Parse geoindkey to extract geographic/industry components
        2. Group by target aggregation level (groupbydigits)
        3. Sum variable values and count available/missing data
        4. Track data sources (CBP, QCEW, QWI) if requested
        5. Return aggregated summary with metadata

    Args:
        dfin (pd.DataFrame):
            Input data with columns:
            - geoindkey: Geographic-industry composite key
            - Variable columns (e.g., 'wages', 'estnum', 'emp1')
            - Optional source columns (e.g., 'wages_source', 'wages_cbp')

        groupbydigits (int):
            Target NAICS aggregation level for grouping (2-5).
            2=2-digit sector, 3=3-digit subsector, 4=4-digit group
            Default is 3.

        levelgrouped (int):
            Source detail level of data being aggregated (typically 4 or 6).
            Default is 4.

        variable (str):
            Variable to aggregate ('wages', 'emp1', 'emp2', 'emp3', 'estnum').
            Default is "wages".

        include_source (bool):
            If True, includes data source information in output.
            Default is True.

        onlyQCEW (bool):
            If True, filters to only QCEW source data before aggregation.
            Default is False.

        perestab_stats (bool):
            If True, statistics about the variable per establishment are also used

        naicsdf (pd.DataFrame, optional):
            NAICS crosswalk table for hierarchical processing.
            Default is None.

        include_estab_emp3_stats (bool):
            If True, includes establishment and emp3 statistics in output.
            Default is True.

    Returns:
        pd.DataFrame:
            Aggregated summary with columns:
            - geo[groupbydigits]naics: Geographic-industry grouping key
            - [variable]_sum[levelgrouped]by[groupbydigits]: Sum of values
            - [variable]_missing[levelgrouped]by[groupbydigits]: Count of missing/suppressed values
            - [variable]_propmissing[levelgrouped]by[groupbydigits]: Proportion missing
            - count[levelgrouped]codes: Number of source codes in each group
            - [variable]_source: Predominant source (if include_source=True)
            - Additional source-specific columns if include_source=True


    """
    # Step 1: Define regex pattern to filter appropriate geoindkey values
    # Handles different NAICS code lengths (e.g., '01001_1111//' for 4-digit)
    pattern_grep = rf"_[0-9]{{{levelgrouped}}}[^0-9]{{{6 - levelgrouped}}}"

    if variable+'_source' not in dfin.columns:
        include_source=False
        onlyQCEW=False

    if levelgrouped == 6:
        pattern_grep = r"_[0-9]{6}"
        dfin=dfin.loc[dfin["agglvl_code"]==78,:]
    else:
        dfin=dfin.loc[dfin['agglvl_code']==72+levelgrouped,:]
    if onlyQCEW:
        dfin[variable+"_qcew"]=dfin[variable]
        dfin.loc[dfin[variable+"_source"]!="qcew",variable+"_qcew"]=np.nan
        variable=variable+"_qcew"
        include_source=False
    # Step 2: Determine string positions for grouping keys
    # Creates keys like '01001_111' for groupbydigits=3
    str_end_idx = -(6 - groupbydigits)# - 1
    label_group = f"{levelgrouped}by{groupbydigits}"
    # Step 3: Filter and prepare dataframe
    df = dfin[dfin['geoindkey'].str.contains(pattern_grep, regex=True)].copy()

    if "geo"+str(groupbydigits)+"naics" in df.columns:
        df["geodignaics"]=df["geo"+str(groupbydigits)+"naics"]
    elif groupbydigits==2:
        df["geo2naics"] = df["geoindkey"].str.slice(stop=-4)
        df=geo2naics_dash_handler(df,naicsdata=naicsdf)
        df['geodignaics']=df['geo2naics']
    else:
        df['geodignaics'] = df['geoindkey'].str[:str_end_idx]

    if include_source:
        df = df[['geoindkey', 'geodignaics', 'state', 'cnty', 'estnum', variable, variable+'_source',"emp3"]]
    elif variable=="estnum" or "estnum" not in df.columns:
        df = df[['geoindkey', 'geodignaics', 'state', 'cnty', variable,"emp3"]]
    else:
        df = df[['geoindkey', 'geodignaics', 'state', 'cnty', 'estnum', variable,"emp3"]]

    if variable!='estnum' and include_estab_emp3_stats:
        df['estab_notna']=df['estnum']
        df.loc[df[variable].isna(),'estab_notna']=0
        df['estab_na']=df['estnum']
        df.loc[df[variable].notna(),'estab_na']=0
        df['emp3_notna'] = df['emp3']
        df.loc[df[variable].isna(), 'emp3_notna'] = 0
        df['emp3_na'] = df['emp3']
        df.loc[df[variable].notna(), 'emp3_na'] = 0
    elif include_estab_emp3_stats:
        df['emp3_notna'] = df['emp3']
        df.loc[df[variable].isna(), 'emp3_notna'] = 0
        df['emp3_na'] = df['emp3']
        df.loc[df[variable].notna(), 'emp3_na'] = 0
    df[variable]=df[variable].astype(float)
    # Step 4: Aggregate data by grouping key
    if perestab_stats and variable!="estnum":
        df['newcolname_perest'] = df[variable] / df['estnum']
        if include_estab_emp3_stats:
            count6dig = df.groupby('geodignaics').agg(
                CountCodes=('geoindkey', 'count'),
                newcolname=(variable, lambda x: np.nansum(x)),
                newcolname_missing=(variable, lambda x: x.isna().sum()),
                newcolname_avgperest=('newcolname_perest',"mean"),
                newcolname_medperest=('newcolname_perest', "median"),
                estnum_na=('estab_na',lambda x:x.sum()),
                estnum_notna=('estab_notna', lambda x: x.sum()),
                emp3_na=('emp3_na', lambda x: x.sum()),
                emp3_notna=('emp3_notna', lambda x: x.sum())
            )
        else:
            count6dig = df.groupby('geodignaics').agg(
                CountCodes=('geoindkey', 'count'),
                newcolname=(variable, lambda x: np.nansum(x)),
                newcolname_missing=(variable, lambda x: x.isna().sum()),
                newcolname_avgperest=('newcolname_perest',"mean"),
                newcolname_medperest=('newcolname_perest', "median")
            )
    elif include_estab_emp3_stats and variable!="estnum":
        # Step 4: Aggregate data by grouping key
        count6dig = df.groupby('geodignaics').agg(
            CountCodes=('geoindkey', 'count'),
            newcolname=(variable, lambda x: np.nansum(x)),
            newcolname_missing=(variable, lambda x: x.isna().sum()),
            estnum_na=('estab_na', lambda x: x.sum()),
            estnum_notna=('estab_notna', lambda x: x.sum()),
            emp3_na=('emp3_na', lambda x: x.sum()),
            emp3_notna=('emp3_notna', lambda x: x.sum())
        )
    elif include_estab_emp3_stats:
        # Step 4: Aggregate data by grouping key
        count6dig = df.groupby('geodignaics').agg(
            CountCodes=('geoindkey', 'count'),
            newcolname=(variable, lambda x: np.nansum(x)),
            newcolname_missing=(variable, lambda x: x.isna().sum()),
            emp3_na=('emp3_na', lambda x: x.sum()),
            emp3_notna=('emp3_notna', lambda x: x.sum())
        )
    else:
        # Step 4: Aggregate data by grouping key
        count6dig = df.groupby('geodignaics').agg(
            CountCodes=('geoindkey', 'count'),
            newcolname=(variable, lambda x: np.nansum(x)),
            newcolname_missing=(variable, lambda x: x.isna().sum())
        )
    count6dig['grouplevels'] = f"group{label_group}"
    if include_source:
        sumsource=df.pivot_table(index="geodignaics",columns=variable+"_source",values=variable,aggfunc='sum',fill_value=0).add_prefix("sum_"+variable+"_")
        countsource = df.pivot_table(index="geodignaics", columns=variable + "_source", values=variable,
                                   aggfunc=count_notna,dropna=True,fill_value=0).add_prefix("count_" + variable + "_from_")
        count6dig=pd.concat([count6dig,sumsource,countsource],axis=1)


    count6dig=count6dig.reset_index()
    count6dig['propmissing']=count6dig['newcolname_missing']/count6dig['CountCodes']
    if include_estab_emp3_stats:
        for cname in ["estnum_na",'estnum_notna','emp3_na',"emp3_notna"]:
            if cname in count6dig.columns:
                count6dig[cname].fillna(0)
        count6dig['emp3_propmissing'] = count6dig['emp3_na'] / (
                    count6dig['emp3_na'] + count6dig['emp3_notna'])  # CountCodes']
        if 'estnum_na' in count6dig.columns:
            count6dig['estnum_propmissing'] = count6dig['estnum_na'] / (count6dig['estnum_na']+count6dig['estnum_notna'])#CountCodes']
    if set(df['geodignaics'].to_list()) != set(count6dig['geodignaics'].to_list()):
        print(
            f"count6dig is missing some geodignaics: {set(df['geodignaics'].to_list()) - set(count6dig['geodignaics'].to_list())}")
    # Step 5: Special handling for non-6-digit aggregations
    if levelgrouped != 6:
        count6dig.loc[count6dig['newcolname_missing'] == count6dig['CountCodes'], 'newcolname'] = np.nan
    for cname in ['newcolname_missing','CountCodes',"newcolname"]:
        count6dig[cname]=count6dig[cname].fillna(0)
    count6dig["propmissing"].fillna(1)
    if onlyQCEW:
        variable=variable.replace("_qcew","")
    if "geo"+str(groupbydigits)+"naics" in count6dig.columns:
        count6dig.drop(columns="geo"+str(groupbydigits)+"naics",inplace=True)

    count6dig = count6dig.rename(columns={
        'geodignaics': f'geo{groupbydigits}naics',
        'CountCodes': f'count{label_group}codes',
        'newcolname': f'{variable}_sum{label_group}',
        'newcolname_missing': f'{variable}_missing{label_group}',
        'propmissing': f'{variable}_propmissing{label_group}',
        'newcolname_avgperest': f'{variable}_avgperest{label_group}',
        'newcolname_medperest': f'{variable}_medperest{label_group}',
        'estnum_na':f'estnum_{variable}_missing{label_group}',
        'estnum_propmissing': f'estnum_{variable}_propmissing{label_group}',
        'emp3_na': f'emp3_{variable}_missing{label_group}',
        'emp3_propmissing': f'emp3_{variable}_propmissing{label_group}'
        },errors="ignore")

    #count6dig.rename(columns={"newcolname":newcolname,"newcolname_missing":newcolname+"_missing"},inplace=True)
    return count6dig

def get_xwalk_naics(crosswalk_file,expand_naics2=True):
    """
    Reads and processes the NAICS crosswalk file.

    Steps:
    1. Reads the CSV file.
    2. Cleans the 'super_sector' column by removing all non-numeric characters.
    3. Cleans the 'naics_sector' column by removing all characters except digits and the '-'
    4. For rows where 'naics_sector' contains a dash, expands the row into multiple rows
       for each individual sector in the range, specifically 31-33, 44-45, and 48-49
    5. Removes any rows still containing a dash in 'naics_sector'.
    """
    xwalk = pd.read_csv(crosswalk_file)

    # Clean 'super_sector' by removing non-numeric characters
    xwalk['super_sector'] = xwalk['super_sector'].astype(str).str.replace(r'[^0-9]', '', regex=True)

    # Clean 'naics_sector' while preserving dashes
    xwalk['naics_sector'] = xwalk['naics_sector'].astype(str).str.replace(r'[^0-9-]', '', regex=True)

    if expand_naics2:
        # Identify rows with a dash in 'naics_sector'
        dash_rows = xwalk[xwalk['naics_sector'].str.contains("-")]

        # Define expanded ranges:
        expand_mapping = {
            "31-33": ["31", "32", "33"],
            "44-45": ["44", "45"],
            "48-49": ["48", "49"]
        }

        # Expand each row that contains a dash
        expanded_rows = []
        for idx, row in dash_rows.iterrows():
            key = row['naics_sector']
            if key in expand_mapping:
                for val in expand_mapping[key]:
                    new_row = row.copy()
                    new_row['naics_sector'] = val
                    expanded_rows.append(new_row)

        if expanded_rows:
            df_expanded = pd.DataFrame(expanded_rows)
            xwalk = pd.concat([xwalk, df_expanded], ignore_index=True)

        # Remove rows still containing a dash in 'naics_sector'
        xwalk = xwalk[~xwalk['naics_sector'].str.contains("-")]

    return xwalk

## Some industry codes only have one 'child' industry code. Sometimes the data omits the child cells for redundancy
## We need to add it back in.
## INPUTS:
## df is dataframe with "geoindkey" column
## naicsdf is output of process_naics_file
def one_naics_code_below_filler(df,naicsdf):
    onlyonebelow=[] #initialize list of dataframes
    # for X in levels naics2,3,4,and 5
    #       find the naicsX codes with only one child naics(X+1)
    for ilvl in [2,3,4,5]:
        #get count of each naicsX and industry industry level combination
        grbydf=naicsdf.groupby(['naics'+str(ilvl),'ind_level']).size().reset_index()
        grbydf.columns=["naics"+str(ilvl),"ind_level","count"]
        grbydf=grbydf.loc[grbydf["ind_level"]!=ilvl,:] #remove those in the current ilvl (only those below remain)
        #get data (naicsX codes) which is only one (or 0) naics(X+1) codes below it
        ilvl_onebelow=grbydf.loc[(grbydf["ind_level"]==ilvl+1)&(grbydf["count"]<2),:].copy()
        if ilvl_onebelow.shape[0]>0: #if there are such codes
            #make dataframe with ilvl+1, naicsX code, naicsX code formatted like geoindkey, and naics(X+1) code
            lwdf = naicsdf.loc[
                naicsdf["ind_level"] == ilvl + 1, ["naics" + str(ilvl), "formatted_code", "ind_level", "naics"]]
            lwdf["additional_digit"]=lwdf["naics"].astype(str).str.slice(start=-1)
            codesbelow = lwdf.loc[
                lwdf["naics" + str(ilvl)].isin(ilvl_onebelow["naics" + str(ilvl)].to_list()), ["formatted_code",
                                                                                               "naics" + str(ilvl),
                                                                                               "additional_digit"]]
            codesbelow.rename(columns={"naics" + str(ilvl): "code", "formatted_code": "single_code_level_below"},
                              inplace=True)
            ilvl_onebelow["code"]=ilvl_onebelow['naics'+str(ilvl)]
            ilvl_onebelow.drop(columns=["naics"+str(ilvl),"count"],inplace=True)
            ilvl_onebelow=ilvl_onebelow.merge(codesbelow,on="code",how="left",indicator=False)
            #ilvl_onebelow has "ind_level" for ilvl+1,"code" is just numeric code at ilvl,
            # "single_code_level_below" is formatted code for ilvl+1 that is below 'code'
            # "additional_digit" is the digit appended to 'code' to get numeric code for 'single_code_level_below'
            onlyonebelow.append(ilvl_onebelow)
    onebelowdf=pd.concat(onlyonebelow) #combined the dataframes from ilvl 2,3,4,5
    onebelowdf["formatted_code"]=onebelowdf["code"].map(format_industry_from_code)

    if "industry" not in df.columns:
        df["industry"]=df['geoindkey'].str.split("_")[1]
    for ilvl in [6,5,4,3]:
        level_onebelow=onebelowdf.loc[onebelowdf['ind_level']==ilvl,:]
        already_filled=df.loc[df['industry'].isin(level_onebelow['single_code_level_below']),["geoindkey","estnum","industry","emp1","emp2","emp3","wages","emp1_source","emp2_source","emp3_source","wages_source","estnum_source"]]
        df = one_level_one_naics_code_below_filler_up(df, onebelowdf, level=ilvl)

    for ilvl in [2,3,4,5]: #for each level fill the df codes below
        df=one_level_one_naics_code_below_filler_down(df,onebelowdf,level=ilvl)
    return(df)

#given a level. this function repeats rows of df when there is only 1 code below that level
def one_level_one_naics_code_below_filler_up(df,onebelowdf,level=6):
    onelevelonebelow=onebelowdf.loc[onebelowdf["ind_level"]==level,:].copy()
    #get values already in df. change geoindkey to match the parent industry
    already_filled = df.loc[
        df['industry'].isin(onelevelonebelow['single_code_level_below']), ].copy()
    onelevelonebelow.rename(columns={"formatted_code":"up_industry","single_code_level_below":"industry"},inplace=True)
    already_filled=already_filled.merge(onelevelonebelow[['up_industry','industry']],on="industry",how='left')
    already_filled.loc[already_filled['up_industry'].notna(),'industry']=already_filled.loc[already_filled['up_industry'].notna(),'up_industry']
    already_filled.drop(columns=['up_industry','agglvl_code','ind_level'],inplace=True,errors='ignore')
    if 'geography' not in already_filled.columns:
        already_filled['geography']=already_filled['geoindkey'].astype(str).str.split("_")[0]
    already_filled['geoindkey']=already_filled['geography'].astype(str)+"_"+already_filled['industry'].astype(str)
    df_i=df.set_index('geoindkey')
    already_filled_i=already_filled.set_index('geoindkey')
    df=df_i.combine_first(already_filled_i).reset_index()

    return df

#given a level. this function repeats rows of df when there is only 1 code below that level
def one_level_one_naics_code_below_filler_down(df,onebelowdf,level=2):
    onelevelonebelow=onebelowdf.loc[onebelowdf["ind_level"]==level+1,:]
    if 'ind_level' not in df.columns:
        df['ind_level']=df['agglvl_code'].astype(float)-72
    #get data match the parent code, adjust it to be from the child code
    to_possibly_fill=df.loc[(df['ind_level']==level)&(df['industry'].isin(onelevelonebelow['formatted_code'])),:].copy()
    to_possibly_fill['ind_level'] = to_possibly_fill['ind_level'] + 1
    to_possibly_fill['agglvl_code'] = to_possibly_fill['agglvl_code'].astype(int) + 1
    cols_for_merge=['agglvl_code','ind_level','industry','geography','geoindkey']
    for add_digit in list(onelevelonebelow["additional_digit"].unique()):
        add_digit_indic = to_possibly_fill["industry"].isin(
            onelevelonebelow.loc[onelevelonebelow["additional_digit"] == add_digit, "formatted_code"].tolist())
        if level == 2:
            to_possibly_fill.loc[add_digit_indic, "geoindkey"] = to_possibly_fill.loc[add_digit_indic, 'geoindkey'].str.replace("----",
                                                                                                          str(add_digit) + "///")
            to_possibly_fill.loc[add_digit_indic, "industry"] = to_possibly_fill.loc[add_digit_indic, 'industry'].str.replace("----",
                                                                                                        str(add_digit) + "///")
        else:
            to_possibly_fill.loc[add_digit_indic, "geoindkey"] = to_possibly_fill.loc[add_digit_indic, 'geoindkey'].str.replace(
                "/" * (6 - level), str(add_digit) + ("/" * (5 - level)))
            to_possibly_fill.loc[add_digit_indic, "industry"] = to_possibly_fill.loc[add_digit_indic, 'industry'].str.replace(
                "/" * (6 - level), str(add_digit) + ("/" * (5 - level)))

    #merge possible fill with data
    df_joined=df.merge(to_possibly_fill,on=cols_for_merge,how='outer',suffixes=('','_possible_fill'))
    #fill relevant columns
    cols_to_fill=[col for col in to_possibly_fill.columns if col in df.columns and col not in cols_for_merge and "naics" not in col.lower()]
    for col in cols_to_fill:
        df_joined[col] = df_joined[col].fillna(df_joined[f'{col}_possible_fill'])
    df_joined.drop(columns=[col for col in df_joined.columns if col.endswith('_possible_fill')])

    return df



## takes in naics string for filename and location "codes_file"
# where code_col is numeric value for where the naics codes are and code_sep is the seperator for columns
# it returns a dataframe with a row for each naics code at any level and columns:
#       ind_level: industry level for code in 'naics' column
#       naics: numeric naics code
#       naics6: repeats 'naics' entry but used for loops in further processing
#       formatted_code: is the 'naics' code formatted to match the format of the goeindkey industry code portion
#       naics5, naics4, naics3, naics2: are numeric codes for the 5,4,3, and 2 prefix of 'naics'. If 'naics' is shorter than the level, then it just has the 'naics' code
#       dashed_naics2: indicator that the 'naics' code is a part of one of the dashed 2-digit naics groupings. (i.e. 31-33)
def process_naics_file(code_file,code_col=1,code_sep=','):
    codedf = pd.read_csv(code_file, sep=code_sep)
    # Extract NAICS codes from specified column

    if code_col in codedf.columns:
        codedf['naics'] = codedf.loc[:, code_col]
    else:
        codedf["naics"] = codedf.iloc[:, code_col]

    # Identify and expand dashed code ranges
    dashI = codedf["naics"].astype(str).str.contains("-")
    if dashI.sum() > 0:
        dashdf = codedf.loc[dashI, :]
        list_dfs = [codedf.loc[~dashI, :]]
        startcode_dict={}
        for dashcode in dashdf["naics"].unique():
            splitcode = dashcode.split("-")
            startcode = splitcode[0].strip()
            endcode = splitcode[1].strip()
            ncodes = float(endcode) - float(startcode) + 1
            rw = dashdf.loc[dashdf["naics"] == dashcode, :]
            dfrep = rw.loc[rw.index.repeat(ncodes)].reset_index()
            list_codes=list(range(int(float(startcode)), int(float(endcode) + 1)))
            dfrep["naics"] = list_codes
            list_dfs.append(dfrep)
            startcode_dict[startcode]=list_codes
        codedf = pd.concat(list_dfs)
    #Calculate aggregation level
    codedf["ind_level"] = codedf['naics'].astype(str).str.count(r'\d')

    # Create columns for each NAICS level
    naicsdf=codedf.loc[:,["ind_level","naics"]]
    for ilvl in [6,5,4,3,2]:
        naicsdf['naics' + str(ilvl)] = naicsdf['naics'].astype(str).str.slice(stop=ilvl)
        naicsdf["formatted_code"]=naicsdf["naics"].map(format_industry_from_code)
        # Mark codes that belong to dashed 2-digit grouping
        if ilvl==2 and dashI.sum() > 0:
            naicsdf["dashed_naics2"]=False
            for key,value in startcode_dict.items():
                naicsdf.loc[naicsdf["naics2"].astype(str).isin([str(val) for val in value]),"naics2"]=key
                naicsdf.loc[naicsdf["naics2"].astype(str).isin([str(val) for val in value]),"dashed_naics2"]=True
    return naicsdf

# takes a single code and formats it like the industry portion of the geoindkey
def format_industry_from_code(code):
    code=str(code).strip()
    if len(code)==2:
        return code.ljust(6,"-")
    else:
        return code.ljust(6,"/")

def geo2naics_dash_handler(data,naicsdata=None):
    """
    Handle 2-digit NAICS dashed groupings in geographic-industry keys.

    Converts individual NAICS codes (31, 32, 33) to their grouped representation
    (31-33, 31-33, 31-33) in the geo[N]naics columns. This ensures consistency with
    published aggregations where certain 2-digit sectors are reported as ranges.

    Args:
        data (pd.DataFrame):
            DataFrame containing geo[N]naics columns with individual NAICS codes.

        naicsdata (pd.DataFrame, optional):
            NAICS crosswalk identifying which codes belong to dashed groupings.
            If None, uses hardcoded 2012 NAICS defaults.
            Default is None.

    Returns:
        pd.DataFrame:
            Input DataFrame with geo[N]naics values updated to use dashed groupings
            where applicable.

    Note:
        Common 2-digit dashed groupings in NAICS:
        - 31, 32, 33 → 31-33 (Manufacturing)
        - 44, 45 → 44-45 (Retail Trade)
        - 48, 49 → 48-49 (Transportation)
    """
    if naicsdata is None: #use 2012 default
        dashdict={"31":[31,32,33],
                  "44":[44,45],
                  "48":[48,49]}

    else:         # Build dictionary from NAICS crosswalk
        dashdict={}
        for dash2 in naicsdata.loc[(naicsdata["dashed_naics2"])&(naicsdata["ind_level"]==2),"naics2"].to_list():
            dashdict[str(dash2)]=list(naicsdata.loc[(naicsdata["naics2"].astype(str)==str(dash2))&(naicsdata["ind_level"]==3),"naics3"].astype(str).str.slice(stop=2).unique())
    for key,value in dashdict.items():
        data["geo2naics"]=data["geo2naics"].str.replace("|".join(["_"+str(val) for val in value]),"_"+str(key),regex=True)
    return data



# with open('config_pre2017.yaml', 'r') as configFile:
#      config = yaml.safe_load(configFile)
# preprocessConfig = config['preprocessConfig']
# generalConfig = config['generalConfig']
# foldername = preprocessConfig['DATA_IN_FOLDER']
# #
# dftemp=pd.read_csv("DataDiag/PythonPreprocessOut/combine_data_1208.csv")
# dftemp=dftemp.iloc[:,1:]
# #print(dftemp.columns)
# naicsdf=process_naics_file(generalConfig['NAICS_FILE'])
# print("start filler")
# temp=one_naics_code_below_filler(dftemp,naicsdf)
# print("end filler")
# #print(temp.head())
# #print(temp.columns)
