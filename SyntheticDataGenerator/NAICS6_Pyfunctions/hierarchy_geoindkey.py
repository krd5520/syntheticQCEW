import numpy as np
import pandas as pd
import os
import yaml

## Functions used to navigate the hierarchical nature of both the geographic and industry features of the geoindkey identifier

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

## NOT CURRENTLY USED
def get_state_totals_emp(df):
    #subset to only necessary columns
    dfsub = df[['state','emp','industry','estnum']]
    #group by state and industry code
    dfsub.groupby(['state','industry']).sum()
    # make indentifying key
    dfsub['geoindkey'] = dfsub['state']+"_"+dfsub['industry']
    #unify column names
    dfsub['cnty'] = "---"
    dfsub['geography'] = dfsub['state']
    dfsub['geo_level'] = "S"
    return(dfsub)

## NOT CURRENTLY USED
def find_one_sublevel(naicsdf,level=2):
    #print(naicsdf.columns)
    hasdashI=naicsdf['code'].str.contains("-")
    expanddash=None
    for badcode in naicsdf.loc[hasdashI,"code"]:
        splitcode=badcode.split("-")
        lw=int(splitcode[0])
        up=int(splitcode[-1])
        newcodes=range(lw,up+1)
        expandcodedf=pd.concat([naicsdf[naicsdf['code']==badcode]]*len(newcodes),axis=0,ignore_index=True)
        expandcodedf['code']=[str(x) for x in newcodes]
        if expanddash is None:
            expanddash=expandcodedf
        else:
            expanddash=pd.concat([expanddash,expandcodedf],axis=0,ignore_index=True)
    naicsdf=naicsdf.loc[~hasdashI,:]
    naicsdf=pd.concat([naicsdf,expanddash],axis=0,ignore_index=True)

    naicsdf['level']=naicsdf['code'].str.len()
    onelevelbelow=None
    for level in [2,3,4,5]:
        abovedf=naicsdf[naicsdf['level']==level+1]
        abovedf['levelcode']=abovedf['code'].str[:-1]
        countcodes=abovedf['levelcode'].value_counts()
        onesublevel=pd.Series(countcodes[countcodes==1].index.values)
        if level==2:
            onesublevel=onesublevel.str.ljust(6,"-")
        elif level<5:
            onesublevel=onesublevel.str.ljust(6,"/")
        if onelevelbelow is None:
            onelevelbelow=onesublevel
        else:
            onelevelbelow=np.concat([onelevelbelow,onesublevel])
    #print(onelevelbelow)
    return onelevelbelow, naicsdf



## gets state, cnty, geography, industry, and ind_level from geoindkey
def fill_from_geoindkey(data,numeric_ind_level=True,naics_xwalk=None):
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

    if naics_xwalk is not None:
        containsdash = data['naics2'].astype(str).str.contains("-").sum()
        if isinstance(naics_xwalk,str):
            if containsdash>0:
                naics_xwalk=get_xwalk_naics(naics_xwalk,expand_naics2=False)
            else:
                naics_xwalk=get_xwalk_naics(naics_xwalk,expand_naics2=True)
            naics_xwalk.rename(columns={"naics_sector":"naics2","super_sector":"supersector"},inplace=True, errors="ignore")
        data=data.merge(naics_xwalk[['naics2','supersector','domain']],on='naics2',how='left')
    if numeric_ind_level:
        data["naics2"] = data["naics2"].astype(str)
        data.loc[data["naics2"] == "31","naics2"] = "31-33"
        data.loc[data["naics2"] == "44","naics2"] = "44-45"
        data.loc[data["naics2"] == "48","naics2"] = "48-49"
        data.loc[data["naics2"] == "32", "naics2"] = "31-33"
        data.loc[data["naics2"] == "45", "naics2"] = "44-45"
        data.loc[data["naics2"] == "49", "naics2"] = "48-49"
        data.loc[data["naics2"] == "33", "naics2"] = "31-33"
    return data

def count_notna(x):
    return x.notna().sum()

def get_codes_summary(dfin, groupbydigits=3, levelgrouped=4,variable="wages",include_source=True,onlyQCEW=True,perestab_stats=False):
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
    #if set(df.index.values)!=set(dfin.loc[dfin['agglvl_code']==78,:].index.values):
    #    print(f"missing index {set(df.index.values)-set(dfin.loc[dfin['agglvl_code']==78,:].index.values)}")

    df['geodignaics'] = df['geoindkey'].str[:str_end_idx]
    if include_source:
        df = df[['geoindkey', 'geodignaics', 'state', 'cnty', 'estnum', variable, variable+'_source']]
    elif variable=="estnum" or "estnum" not in df.columns:
        df = df[['geoindkey', 'geodignaics', 'state', 'cnty', variable]]
    else:
        df = df[['geoindkey', 'geodignaics', 'state', 'cnty', 'estnum', variable]]
    df[variable]=df[variable].astype(float)
    # Step 4: Aggregate data by grouping key
    if perestab_stats and variable!="estnum":
        df['newcolname_perest']=df[variable]/df['estnum']
        count6dig = df.groupby('geodignaics').agg(
            CountCodes=('geoindkey', 'count'),
            newcolname=(variable, lambda x: np.nansum(x)),
            newcolname_missing=(variable, lambda x: x.isna().sum()),
            newcolname_avgperest=('newcolname_perest',"mean"),
            newcolname_medperest=('newcolname_perest', "median")
        )
    else:
        # Step 4: Aggregate data by grouping key
        count6dig = df.groupby('geodignaics').agg(
            CountCodes=('geoindkey', 'count'),
            newcolname=(variable, lambda x: np.nansum(x)),
            newcolname_missing=(variable, lambda x: x.isna().sum())
        )
    count6dig['grouplevels'] = f"group{label_group}"
    #print(count6dig[count6dig["CountCodes"]!=count6dig['newcolname_missing']].head())
    if include_source:
        sumsource=df.pivot_table(index="geodignaics",columns=variable+"_source",values=variable,aggfunc='sum',fill_value=0).add_prefix("sum_"+variable+"_")
        countsource = df.pivot_table(index="geodignaics", columns=variable + "_source", values=variable,
                                   aggfunc=count_notna,dropna=True,fill_value=0).add_prefix("count_" + variable + "_from_")
        count6dig=pd.concat([count6dig,sumsource,countsource],axis=1)
    #print(count6dig.loc[(count6dig["CountCodes"]!=count6dig['newcolname_missing'])&(count6dig['newcolname_missing']>3),:].head())


    count6dig=count6dig.reset_index()
    count6dig['propmissing']=count6dig['newcolname_missing']/count6dig['CountCodes']
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
    if perestab_stats and variable!="estnum":
        count6dig = count6dig.rename(columns={
            'geodignaics': f'geo{groupbydigits}naics',
            'CountCodes': f'count{label_group}codes',
            'newcolname': f'{variable}_sum{label_group}',
            'newcolname_missing': f'{variable}_missing{label_group}',
            'propmissing': f'{variable}_propmissing{label_group}',
            'newcolname_avgperest':f'{variable}_avgperest{label_group}',
            'newcolname_medperest':f'{variable}_medperest{label_group}'
        })
    else:
        count6dig = count6dig.rename(columns={
            'geodignaics': f'geo{groupbydigits}naics',
            'CountCodes': f'count{label_group}codes',
            'newcolname': f'{variable}_sum{label_group}',
            'newcolname_missing': f'{variable}_missing{label_group}',
            'propmissing':f'{variable}_propmissing{label_group}'
        })

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