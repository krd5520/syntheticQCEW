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
def fill_from_geoindkey(data,numeric_ind_level=True):
    expandgeoind=data['geoindkey'].str.split('_',expand=True)
    if len(expandgeoind.columns)>2:
        print(data.loc[expandgeoind.iloc[:,3] is not None,'geoindkey'].head())
    data['geography']=expandgeoind.iloc[:,0]
    data.loc[:,'industry'] = expandgeoind.iloc[:, 1]
    data['state']=data['geography'].astype(str).str.slice(start=0,stop=-3)
    data['cnty']=data['geography'].astype(str).str.slice(start=-3)
    ninddig=data['industry'].str.count(r'\d')
    if numeric_ind_level:
        data['ind_level'] = ninddig.astype(int)
        ninddig[ninddig==0]=-1
        data['agglvl_code']=72+ninddig
        for i in [2, 3, 4, 5]:
            data["naics"+str(i)] = ""
            data.loc[data['ind_level']>=i,"naics"+str(i)]=data.loc[data['ind_level']>=i,"industry"].str.slice(start=0,stop=i)
            #data.rename(columns={"tempnaicslevel":'naics'+str(i)},inplace=True)
    else:
        ninddig=ninddig.astype(str)
        ninddig[ninddig=='0']="A"
        ninddig[ninddig == '2'] = "S"
        data['ind_level']=ninddig
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

def get_codes_summary(dfin, groupbydigits=3, levelgrouped=4,variable="wages",include_source=True,onlyQCEW=True):
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
        dfin=dfin.loc[dfin["agglvl_code"]==78,:]
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
    if set(df.index.values)!=set(dfin.loc[dfin['agglvl_code']==78,:].index.values):
        print(f"missing index {set(df.index.values)-set(dfin.loc[dfin['agglvl_code']==78,:].index.values)}")

    df['geodignaics'] = df['geoindkey'].str[:str_end_idx]
    if include_source:
        df = df[['geoindkey', 'geodignaics', 'state', 'cnty', 'estnum', variable, variable+'_source']]
    else:
        df = df[['geoindkey', 'geodignaics', 'state', 'cnty', 'estnum', variable]]
    # Step 4: Aggregate data by grouping key
    count6dig = df.groupby('geodignaics').agg(
        CountCodes=('geoindkey', 'count'),
        newcolname=(variable, lambda x: np.nansum(x.astype(float))),
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
    count6dig = count6dig.rename(columns={
        'geodignaics': f'geo{groupbydigits}naics',
        'CountCodes': f'count{label_group}codes',
        'newcolname': f'{variable}_sum{label_group}',
        'newcolname_missing': f'{variable}_missing{label_group}',
        'propmissing':f'{variable}_propmissing{label_group}'
    })

    #count6dig.rename(columns={"newcolname":newcolname,"newcolname_missing":newcolname+"_missing"},inplace=True)
    return count6dig


def get_varmin(codes4naics, fulldf, variable="emp1"):
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
    tomerge6dig = get_codes_summary(dfin=fulldf, groupbydigits=4, levelgrouped=6, variable=variable)
    # Create minwage column (0 if no data available)
    tomerge6dig['min' + variable] = np.where(tomerge6dig[variable + '_sum6by4'].isna(), 0,
                                                  tomerge6dig[variable + '_sum6by4'])
    tomerge6dig['geoindkey'] = tomerge6dig['geo4naics'].astype(str) + "//"
    tomerge6dig = tomerge6dig[['geoindkey', 'min' + variable]]
    return tomerge6dig


def adjust_geo4naics_varvalues(fitdf, dfmaxmin=None, stabvals=None, variable="emp1",fulldf=None):
    '''
    What is the point?
        adjust_geo4naics_varvalues() constrains wage/emp estimates to stay within min/max bounds
    Inputs:
        1. fitdf - DataFrame with estimates
        2. dfmaxmin - DataFrame with min/max bounds (if none, then fulldf must be the full data without the estimates)
        3. stabvals- if using stable employment as lower bound on employment, this is a series of those values
        4. variable- string name of variable to be adjusted
        5. adjust_indic- series of indicators to determine which of fitdf[variable] can be adjusted.
        6. fulldf- if dfmaxmin is not provided, them fulldf must be the full data without the estimates
    Returns:
        DataFrame with adjusted wage values
    '''
    # Merge with min/max bounds
    if dfmaxmin is None:
        fulldf['geo4naics'] = fulldf['geoindkey'].str.slice(stop=-2)
        df4 = fulldf[fulldf['agglvl_code'] == 76].copy()
        dfmaxmin = get_varmaxmindf(df4dig=df4, fulldf=fulldf, variable=variable)
    if 'geo4naics' not in fitdf.columns:
        fitdf['geo4naics'] = fitdf['geoindkey'].str.slice(stop=-2)
    maxmindf = dfmaxmin[['geo4naics', 'min' + variable, 'max' + variable]].copy()
    fitdf = fitdf.merge(maxmindf, on='geo4naics', how='left')
    fitdf['min'+variable+'_source'] = 'hierarchy'
    fitdf.loc[fitdf['min'+variable]==0,'min'+variable+'_source'] = 'structural'
    #if stabvals is not None and "emp" in variable:
    #    fitdf.loc[:, "min" + variable] = np.fmin(fitdf["min" + variable].to_numpy(), stabvals.to_numpy())
    #    fitdf.loc[fitdf['min'+variable]==stabvals,'min_source']="stable_emp"

    ## check given qcew values
    fitdf['value_status']="within calculated bounds"
    fitdf.loc[(fitdf[variable]<fitdf['min'+variable]),'value_status']="below calculated min"
    fitdf.loc[(fitdf[variable]>fitdf['max'+variable]),'value_status']="above calculated max"
    print(f'When adjusting {variable}: \n{pd.crosstab(fitdf["value_status"],fitdf[variable+"_source"],dropna=False)}')
    fitdf.drop(columns="value_status",inplace=True)
    printheads=False #for testing in development
    if printheads:
        print(f'Check Maxes\n{fitdf.loc[fitdf["max"+variable].notna(),[variable,variable+"_source","min"+variable,"max"+variable]].head()}')

    # Apply constraints
    fitdf[variable] = fitdf[variable].clip(
        lower=fitdf['min' + variable].astype(float),
        upper=fitdf['max' + variable].astype(float)
    )
    #fixed=fitdf.loc[below_min_notqcew.index.values,[variable,variable+"_source",'min'+variable,'max'+variable,'min_source']]
    #print(f'Fixed bounds?\n{fixed.head()}')
    #fitdf = fitdf.drop(columns=['min' + variable, 'max' + variable,'min_source'])
    return fitdf

def get_varmax(codes4naics, fulldf, variable="emp1"):
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
        "maxwages": np.nan,
        "geo3naics": codes4naics.str[:-3],
        "geo2naics": codes4naics.str[:-4],
        "geography": codes4naics.str[:-7]
    })
    fulldf[variable] = fulldf[variable].astype(float)
    # Try 3-digit NAICS level first
    tomergedf3 = fulldf[
        fulldf["geoindkey"].str.contains(r"_[0-9]{3}[^0-9]{3}", regex=True)
    ].copy()
    tomergedf3["geo3naics"] = tomergedf3["geoindkey"].str.slice(stop=-3)

    tomergedf3[variable + "_naics3"] = tomergedf3[variable]#np.where(tomergedf3[variable].notna(), np.nan, tomergedf3[variable])
    tomergedf3 = tomergedf3[["geo3naics", "estnum", variable + "_naics3"]]
    # Merge 3-digit data
    outdf = outdf.merge(tomergedf3, on='geo3naics', how='left', suffixes=('', '_naics3'))
    # For missing values, try 2-digit sector level
    notmaxcodes = outdf[outdf[variable + '_naics3'].isna()]['geo2naics'].tolist()
    fulldf['geo2naics'] = fulldf['geoindkey'].str.slice(stop=-4)
    tomergedf2 = fulldf[fulldf['geo2naics'].isin(notmaxcodes) &
                        fulldf['geoindkey'].str.contains(r"_[0-9]{2}[^0-9]{4}")].copy()
    tomergedf2[variable + '_naics2'] = tomergedf2[variable]#np.where(tomergedf2[variable].notna(), np.nan, tomergedf2[variable])
    tomergedf2 = tomergedf2[['geo2naics', 'estnum', variable + '_naics2']]
    # Calculate differences between sector and summed 3-digit wages
    tomergedf3[variable + '_naics3'] = tomergedf3[variable + '_naics3'].astype(float)
    tomergedf3['geo2naics'] = tomergedf3['geo3naics'].str[:-1]  # Extract sector codes
    tomergedf3 = tomergedf3.groupby('geo2naics', as_index=False).agg(sumvar3=(variable + '_naics3', 'sum'))
    tomergedf2 = tomergedf2.merge(tomergedf3, on='geo2naics', how='left')
    tomergedf2['missing_' + variable + '_naics2'] = tomergedf2[variable + '_naics2'].astype(float) - tomergedf2[
        'sumvar3']
    tomergedf2 = tomergedf2[['geo2naics', 'missing_' + variable + '_naics2', 'estnum', variable + '_naics2']]
    # Merge sector-level data
    outdf = outdf.merge(tomergedf2, on='geo2naics', how='left', suffixes=('', '_naics2'))

    outdf['max' + variable] = outdf.apply(
        lambda row: row[variable + '_naics3'] if pd.notna(row[variable + '_naics3']) else row[
            'missing_' + variable + '_naics2'], axis=1)
    outdf = outdf.drop(columns=[variable + '_naics2'])
    fulldf[variable] = fulldf[variable].astype(float)
    # For remaining missing values, use county-wide totals
    max_allind_allcounty = fulldf[fulldf['agglvl_code'] == 76][variable].max(skipna=True)
    notmaxcodes = outdf[outdf['max' + variable].isna()]['geography'].tolist()
    tomergedfall = fulldf.copy()
    tomergedfall['geography'] = tomergedfall['geoindkey'].str[:-7]
    tomergedfall = tomergedfall[tomergedfall['geography'].isin(notmaxcodes)]
    tomergedfall = tomergedfall[tomergedfall['geoindkey'].str.contains('_------')]
    tomergedfall[variable + 'all'] = tomergedfall.apply(
        lambda row: max_allind_allcounty if row[variable] != "" else row[variable], axis=1)
    tomergedfall = tomergedfall[['geography', 'estnum', variable + 'all']]
    # Calculate county-level differences
    tomergedf2['geography'] = tomergedf2['geo2naics'].str[:-3]
    tomergedf2 = tomergedf2.groupby('geography', as_index=False)[variable + '_naics2'].sum(min_count=1)
    tomergedf2.rename(columns={variable + '_naics2': 'sum' + variable + '2'}, inplace=True)
    tomergedfall = tomergedfall.merge(tomergedf2, on="geography", how="left")
    tomergedfall['missing' + variable + 'all'] = tomergedfall[variable + 'all'].astype(float) - tomergedfall[
        'sum' + variable + '2'].astype(float)
    tomergedfall = tomergedfall[['geography', 'missing' + variable + 'all', 'estnum', variable + 'all']]
    # Final merge and return
    outdf = outdf.merge(tomergedfall, on="geography", how="left", suffixes=("", "_allindustry"))
    outdf['max' + variable] = outdf.apply(
        lambda row: row['missing' + variable + 'all'] if pd.isna(row['max' + variable]) else row['max' + variable],
        axis=1)
    outdf = outdf[['geoindkey', 'max' + variable]]
    return outdf


def get_varmaxmindf(df4dig, fulldf, variable="emp1"):
    '''
    What is the point?
        get_varmaxmindf() combines county by naics4 data with min/max bounds
    Inputs:
        1. df4dig - 4-digit NAICS level data
        2. fulldf - Complete dataset
    Returns:
        DataFrame with original data plus minwage and maxwage columns
    '''
    # Merge employment data with wage data

    # Get min and max wage bounds
    mindf = get_varmin(codes4naics=df4dig['geoindkey'], fulldf=fulldf, variable=variable)
    maxdf = get_varmax(codes4naics=df4dig['geoindkey'], fulldf=fulldf, variable=variable)
    # Merge all data
    df4_maxmin = df4dig.merge(maxdf, on="geoindkey", how="left") \
        .merge(mindf, on="geoindkey", how="left")
    df4_maxmin['min'+variable] = df4_maxmin['min'+variable].fillna(0)
    return df4_maxmin

#def adjust_negative_diff(df,count6dig)