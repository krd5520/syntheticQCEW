import re
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from formulaic import Formula
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
sys.path.append(os.path.abspath('./'))
from GeneralFunctions import custom_predict
from hierarchy_geoindkey import *

# with open('./config.yaml','r') as configFile:
#     config = yaml.safe_load(configFile)
#     employmentConfig = config['employmentConfig']
pd.set_option('mode.chained_assignment', None)

printheads=False #for testing in development
## Hard-coded, NAICS codes which CBP does not include in its data.
excluded_cbp_naics6=["525110", "525120","525190","525920","541120"]
hardcode_cbp_flags=pd.DataFrame({'flag':["G","H","J"],'min_noise_percent':[0.0,0.02,0.05],'max_noise_percent':[0.02,0.05,np.nan]})

def excluded_cbp_adjustments(df,excluded_cbp):
    return df

def quarter_source_adjustment(data, generalConfig, response, quarterConfig=None, formula=None, adjust_source=True,
                              source="CBP", rseed=None):
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
    if rseed is not None:
        np.random.seed(rseed)
    if "year_qtr" not in data.columns:
        data['year_qtr'] = data['year'] + data['qtr'].astype(float).multiply(0.25)
    # Step 1
    # get difference of quarters
    tempdata = data.copy()

    if adjust_source:  # adjusting from one data source to another
        if formula is None:
            formula = response + "~."
        usenewsource = data.loc[:, response].isna()
        subdata = tempdata.loc[~usenewsource, :].copy()
    else:  # adjusting CBP for quarter
        tempdata['year_qtr_diff'] = tempdata['year_qtr_cbp'].astype(float) - tempdata['year_qtr'].astype(float)

        if tempdata['year_qtr'].nunique() > 1:
            formula_stem = response + "~year_qtr_diff+qtr*naics2+"
        else:
            formula_stem = response + "~"

        # Retrieve OLS formula from config.yaml if quarterConfig exists
        if quarterConfig is not None and formula is None:
            if response == "wages_qcew":
                formula = quarterConfig['WAGE_OLS_FORMULA']
            else:
                formula = quarterConfig['EMP_OLS_FORMULA']
        elif formula is None:  # use defaults
            formula = formula_stem + "wages_cbp*wages_cbp_flag+np.log(estnum_cbp)+np.log(estnum)+emp3_cbp+emp3_cbp_flag+agglvl_code+agglvl_code*naics2"

            # ensure variable type is correct
            for vname in ["year", "wages_cbp", "estnum_cbp", "estnum", "wages_qcew", "emp1_qcew", "emp2_qcew",
                          "emp3_qcew",
                          "emp3_cbp"]:
                tempdata[vname] = tempdata[vname].astype(float)
            for vname in ['qtr', 'qtr_cbp', 'wages_cbp_flag', "emp3_cbp_flag", "agglvl_code", "naics2", "naics3",
                          "naics4",
                          "naics5"]:
                tempdata[vname] = tempdata[vname].astype("category")
        # dataset to fit model on
        subdata = tempdata[
            (~tempdata['wages_cbp'].isna()) & (~tempdata['wages_qcew'].isna()) & (~tempdata['emp3_cbp'].isna())].copy()

    # Create design matrices (gets the variables ready for fitting in statsmodels.OLS) using the formula
    # and perform initial model fitting
    y_pre, X_pre = Formula(formula).get_model_matrix(subdata)
    model = sm.OLS(y_pre, X_pre).fit()

    if adjust_source:  # adjusting datasources
        print("Model to adjust " + source + "  " + response)
        print(model.summary())

        # dataset with missing response values
        # no_response=data.loc[data.loc[:,response].isna(),:].copy()
        pred, se_fit = custom_predict(tempdata[usenewsource], model, rseed=rseed)
        # responsefit = np.random.normal(
        #    loc=pred,  # Center at predicted values
        #    scale=se_fit,  # Scale by prediction uncertainty
        #    size=len(no_response)
        # )

        data.loc[usenewsource, response] = np.round(pred, decimals=0)
        if response + "_source" not in data.columns:
            data.loc[:, response + "_source"] = ""
        data.loc[usenewsource, response + "_source"] = source.lower()
        data.loc[data[response].isna(), response + "_source"] = ""




    else:
        if quarterConfig is not None and quarterConfig['DIAGNOSTIC_PLOTS'] is not None:
            save_diagnostic_plots(model, formula, quarterConfig['DIAGNOSTIC_PLOTS'])
        print("Model to adjust CBP " + response + " to quarter " + str(generalConfig['QTR']))
        print(model.summary())

        split_response = response.split("_")
        split_response.pop()
        response_stem = "_".join(split_response)
        data.loc[subdata.index.tolist(), response_stem + "_cbp"] = model.fittedvalues()

    return data


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


def adjust_geo4naics_varvalues(fitdf, dfmaxmin=None, stabvals=None, variable="emp1",fulldf=None,onlyqcew=True,minonly=True):
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
        dfmaxmin = get_varmaxmindf(df4dig=df4, fulldf=fulldf, variable=variable,onlyqcew=onlyqcew)
    if 'geo4naics' not in fitdf.columns:
        fitdf['geo4naics'] = fitdf['geoindkey'].str.slice(stop=-2)
    maxmindf = dfmaxmin[['geo4naics', 'min' + variable, 'max' + variable,
                         "max" + variable + "_source"]].copy()

    #maxmindf = dfmaxmin[['geo4naics', 'min' + variable, 'max' + variable,'max'+variable+'_qcewsource',"max"+variable+"_source"]].copy()
    fitdf = fitdf.merge(maxmindf, on='geo4naics', how='left')
    fitdf['min'+variable+'_source'] = 'hierarchy'
    fitdf.loc[fitdf['min'+variable]==0,'min'+variable+'_source'] = 'structural'
    #if stabvals is not None and "emp" in variable:
    #    fitdf.loc[:, "min" + variable] = np.fmin(fitdf["min" + variable].to_numpy(), stabvals.to_numpy())
    #    fitdf.loc[fitdf['min'+variable]==stabvals,'min_source']="stable_emp"

    if printheads:
        ## check given qcew values
        fitdf['value_status'] = "within calculated bounds"
        fitdf.loc[(fitdf[variable] < fitdf['min' + variable]), 'value_status'] = "below calculated min"
        fitdf.loc[(fitdf[variable] > fitdf['max' + variable]), 'value_status'] = "above calculated max"
        print(
            f'When adjusting {variable}: \n{pd.crosstab(fitdf["value_status"], fitdf[variable + "_source"], dropna=False)}')
        abovedf=fitdf.loc[(fitdf["value_status"] == "above calculated max"), [variable + "_source",
                                                                      "max" + variable + "_qcewsource","max"+variable+"_source"]].groupby(
            [variable + "_source","max"+variable+"_source"]).describe()
        print(f'Above Calculated Max prop from qcew:\n {abovedf}')
        abovedf=fitdf.loc[(fitdf["value_status"] == "above calculated max"), [variable + "_source","max"+variable+"_source"]]
        print(f'Above Calculated Max prop from qcew:\n {pd.crosstab(abovedf[variable+"_source"],abovedf["max"+variable+"_source"])}')
        fitdf.drop(columns="value_status", inplace=True)
        print(f'Check Maxes\n{fitdf.loc[fitdf["max"+variable].notna(),[variable,variable+"_source","min"+variable,"max"+variable]].head()}')

    # Apply constraints
    if minonly:
        fitdf[variable]=fitdf[variable].clip(lower=fitdf['min'+variable].astype(float))
    else:
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
        "max"+variable: np.nan,
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
    tomergedf3[variable + "_source_naics3"] = tomergedf3[variable+"_source"]
    tomergedf3 = tomergedf3[["geo3naics", "estnum", variable + "_naics3", variable + "_source_naics3"]]
    # Merge 3-digit data
    outdf = outdf.merge(tomergedf3, on='geo3naics', how='left', suffixes=('', '_naics3'))
    # For missing values, try 2-digit sector level
    notmaxcodes = outdf[outdf[variable + '_naics3'].isna()]['geo2naics'].tolist()
    fulldf['geo2naics'] = fulldf['geoindkey'].str.slice(stop=-4)
    tomergedf2 = fulldf[fulldf['geo2naics'].isin(notmaxcodes) &
                        fulldf['geoindkey'].str.contains(r"_[0-9]{2}[^0-9]{4}")].copy()
    tomergedf2[variable + '_naics2'] = tomergedf2[variable]#np.where(tomergedf2[variable].notna(), np.nan, tomergedf2[variable])
    tomergedf2[variable + '_source_naics2'] = tomergedf2[variable+"_source"]
    tomergedf2 = tomergedf2[['geo2naics', 'estnum', variable + '_naics2', variable + "_source_naics2"]]
    # Calculate differences between sector and summed 3-digit wages
    tomergedf3[variable + '_naics3'] = tomergedf3[variable + '_naics3'].astype(float)
    tomergedf3['geo2naics'] = tomergedf3['geo3naics'].str[:-1]  # Extract sector codes
    tomergedf3grouped = tomergedf3.groupby('geo2naics', as_index=False).agg(sumvar3=(variable + '_naics3', 'sum'))

    extrainvestigate=False
    if extrainvestigate:
        ## extra investigation
        tallysource = tomergedf3.groupby(['geo2naics', variable + '_source_naics3'], dropna=False).size().to_frame("count")
        tallysource['prop']=tallysource['count']/tallysource.groupby(level=0)['count'].transform('sum')
        tallysource=tallysource.reset_index().pivot_table(
            index='geo2naics', columns=variable + "_source_naics3", values=["count","prop"],
            dropna=False)  # fill_value=0, dropna=False)
        tallysource.columns = tallysource.columns.map(lambda index: f'{variable}_{index[0]}_source_{index[1]}_naics3')
        tallysource = tallysource.reset_index()
        #print(f'tallysource in tomergedf3 groupby geo2naics head after pivot\n{tallysource.head(10)}')
        tomergedf3=tomergedf3grouped.merge(tallysource,on='geo2naics',how="left")
    else:
        tomergedf3=tomergedf3grouped
    tomergedf2 = tomergedf2.merge(tomergedf3, on='geo2naics', how='left')
    tomergedf2['missing_' + variable + '_naics2'] = tomergedf2[variable + '_naics2'].astype(float) - tomergedf2[
        'sumvar3']
    #print(f"describe missing geo2naics \n{tomergedf3['missing_'+variable+'_naics2'].describe()}")
    if extrainvestigate:
        tomergedf2 = tomergedf2[['geo2naics', 'missing_' + variable + '_naics2', 'estnum', variable + '_naics2',variable+'_source_naics2',variable+'_prop_source_qcew_naics3']]
    else:

        tomergedf2 = tomergedf2[['geo2naics', 'missing_' + variable + '_naics2', 'estnum', variable + '_naics2',variable+'_source_naics2']]
    # Merge sector-level data
    outdf = outdf.merge(tomergedf2, on='geo2naics', how='left', suffixes=('', '_naics2'))
    #print(f'outdf head before getting max \n {outdf.head()}')
    outdf['max' + variable] = outdf.apply(
        lambda row: row[variable + '_naics3'] if pd.notna(row[variable + '_naics3']) else row[
            'missing_' + variable + '_naics2'], axis=1)
    if extrainvestigate:
        outdf['max'+variable+"_qcewsource"]=outdf[variable+'_prop_source_qcew_naics3']
        outdf.loc[(outdf['max' + variable] == outdf[variable + "_naics3"]) & (
                    outdf['max' + variable + '_source'] == "qcew"), 'max' + variable + "_qcewsource"] = 1
        outdf.loc[(outdf['max' + variable] == outdf[variable + "_naics3"]) & (
                    outdf['max' + variable + '_source'] != "qcew"), 'max' + variable + "_qcewsource"] = 0

    outdf['max'+variable+"_source"]=""
    outdf.loc[outdf['max' + variable]==outdf[variable+"_naics3"],"max"+variable+"_source"]="geo3naics"
    outdf.loc[outdf['max' + variable]==outdf["missing_"+variable+"_naics2"],"max"+variable+"_source"]="missing_geo2naics"



    #print(f'inside get_varmax, head of outdf after max{variable} added when max{variable} is not na \n {outdf.loc[outdf["max"+variable].notna(),:].head()}')
    outdf = outdf.drop(columns=[variable + '_naics2'])
    fulldf[variable] = fulldf[variable].astype(float)
    # For remaining missing values, use county-wide totals
    max_allind_allcounty = fulldf[fulldf['agglvl_code'] == 76][variable].max(skipna=True)
    #print(f'number of maxes from allind_allcounty {outdf["max"+variable].isna().sum()}')
    if outdf['max'+variable].isna().sum()>0:
        #print(f'number of maxes from allind_allcounty {outdf["max"+variable].isna().sum()}')
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
        tomergedf2group = tomergedf2.groupby('geography', as_index=False)[variable + '_naics2'].sum(min_count=1)
        tomergedf2group.rename(columns={variable + '_naics2': 'sum' + variable + '2'}, inplace=True)

        if extrainvestigate:
            ## extra investigation
            tallysource = tomergedf2.groupby(['geography', variable + '_source_naics2'], dropna=False).size().to_frame("count")
            tallysource['prop'] = tallysource['count'] / tallysource.groupby(level=0)['count'].transform('sum')
            tallysource = tallysource.reset_index().pivot_table(
                index='geography', columns=variable + "_source_naics2", values=["count", "prop"],
                dropna=False)  # fill_value=0, dropna=False)
            tallysource.columns = tallysource.columns.map(lambda index: f'{variable}_{index[0]}_source_{index[1]}_naics2')
            tallysource = tallysource.reset_index()
            # print(f'tallysource in tomergedf3 groupby geo2naics head after pivot\n{tallysource.head(10)}')
            tomergedf2 = tomergedf2group.merge(tallysource, on='geography', how="left")
        else:
            tomergedf2=tomergedf2group
        tomergedfall = tomergedfall.merge(tomergedf2, on="geography", how="left")
        tomergedfall['missing' + variable + 'all'] = tomergedfall[variable + 'all'].astype(float) - tomergedfall[
            'sum' + variable + '2'].astype(float)

        if extrainvestigate:
            tomergedfall = tomergedfall[['geography', 'missing' + variable + 'all', 'estnum', variable + 'all',variable+'_prop_source_qcew_naics2']]
        else:
            tomergedfall = tomergedfall[['geography', 'missing' + variable + 'all', 'estnum', variable + 'all']]
        # Final merge and return
        outdf = outdf.merge(tomergedfall, on="geography", how="left", suffixes=("", "_allindustry"))
        outdf['max' + variable] = outdf.apply(
            lambda row: row['missing' + variable + 'all'] if pd.isna(row['max' + variable]) else row['max' + variable],
            axis=1)
        if extrainvestigate:
            outdf.loc[outdf['max' + variable] == outdf[
                "missing_" + variable + "all"], 'max' + variable + "_qcewsource"] = outdf.loc[
                outdf['max' + variable] == outdf[
                    "missing_" + variable + "all"], variable + '_prop_source_qcew_naics2']
        outdf.loc[outdf['max' + variable] == outdf[
            "missing" + variable + "all"], "max" + variable + "_source"] = "missing_geography"

    if extrainvestigate:
        outdf = outdf[['geoindkey', 'max' + variable,"max"+variable+"_qcewsource", "max"+variable+"_source"]]
    else:

        outdf = outdf[['geoindkey', 'max' + variable, "max"+variable+"_source"]]
    return outdf


def get_varmaxmindf(df4dig, fulldf, variable="emp1",onlyqcew=True):
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
    #print(f'inside get_varmaxmin before get_varmin. columns of fulldf: {fulldf.columns}')
    if onlyqcew:
        fulldf=fulldf.loc[fulldf[variable+"_source"]=="qcew",:].copy()
    mindf = get_varmin(codes4naics=df4dig['geoindkey'], fulldf=fulldf, variable=variable)
    maxdf = get_varmax(codes4naics=df4dig['geoindkey'], fulldf=fulldf, variable=variable)
    # Merge all data
    df4_maxmin = df4dig.merge(maxdf, on="geoindkey", how="left") \
        .merge(mindf, on="geoindkey", how="left")
    df4_maxmin['min'+variable] = df4_maxmin['min'+variable].fillna(0)
    return df4_maxmin



def adjust_negative_diff(df4,df,justdrop=True):
    # get summary of countyXnaics6 cells by countyXnaics4 codes for estnum, wages, and emp3
    count6dig = get_codes_summary(df, groupbydigits=4, levelgrouped=6, variable="estnum", onlyQCEW=False,include_source=False)
    df4estnum = df4.merge(count6dig, on=['geo4naics'], how='left', indicator=False, suffixes=["", "_droplater"])
    df4estnum.drop(columns=[dropcol for dropcol in df4estnum.columns if "_droplater" in dropcol], errors="ignore",inplace=True)
    for vname in ['wages','emp1','emp2','emp3']:
        df4estnum[vname+'_perest']=df4estnum[vname]/df4estnum['estnum']
        #df4estnum['old_'+vname]=df4estnum[vname]
        df4estnum[vname]=df4estnum[vname+"_perest"]*df4estnum['estnum_sum6by4']
        df4estnum.drop(columns=[vname+"_perest"],inplace=True)
    df4estnum['estnum']=df4estnum['estnum_sum6by4']
    df4estnum.set_index(df4estnum['geoindkey'],inplace=True,drop=True)
    df.set_index(df['geoindkey'],inplace=True,drop=False)
    df.loc[df['geoindkey'].isin(df4estnum.index.to_list()),['wages','emp1','emp2','emp3','estnum']]=df4estnum[['wages','emp1','emp2','emp3','estnum']]
    df.drop(columns=['geoindkey'],inplace=True)
    df.reset_index(inplace=True,drop=False)
    df4estnum.drop(columns=['geoindkey'], inplace=True)
    df4estnum.reset_index(inplace=True,drop=False)
    df4estnum.drop(columns=[dropcol for dropcol in df4estnum.columns if "6by4" in dropcol], errors="ignore",inplace=True)
    df6=df.loc[df['agglvl_code']==78,:]
    df6['geo4naics']=df6['geoindkey'].str.slice(stop=-2)
    df6['geo5naics']=df6['geoindkey'].str.slice(stop=-1)
    df6['geo3naics'] = df6['geoindkey'].str.slice(stop=-3)

    count6digwages = get_codes_summary(df, groupbydigits=4, levelgrouped=6, variable="wages", onlyQCEW=False,
                                  include_source=True)
    count6digemp3 = get_codes_summary(df, groupbydigits=4, levelgrouped=6, variable="emp3", onlyQCEW=False,
                                       include_source=True)
    df4estnum=df4estnum.merge(count6digwages,on=['geo4naics'], how='left', indicator=True, suffixes=["", "_droplater"])
    df4estnum.drop(columns=[dropcol for dropcol in df4estnum.columns if "_droplater" in dropcol], errors="ignore",inplace=True)

    ## Get difference between county by NAICS4 and sum of known county by NAICS6
    df4estnum["wagesdiff"] = df4estnum["wages"].astype(float) - df4estnum['wages_sum6by4'].astype(float)
    df4 = df4estnum.merge(count6digemp3, on=['geo4naics'], how='left', indicator=False, suffixes=["", "_droplater"])
    df4["emp3diff"] = df4["emp3"].astype(float) - df4['emp3_sum6by4'].astype(float)

    ## summarize difference in establishment counts
    ## This showed no cells had more establishments at the countyXnaics6 level than at the countyXnaics4 level
    #print(f'# countyXnaics4 codes with all establishments accounted for: {df4estnum.loc[df4estnum["estnumdiff"]==0,:].shape[0]}')
    #print(pd.crosstab(missingestnum['from_cbp_missing_naics6']))
    #print(f'When There are missing wage values in the countyXnaics6 cells...')
    #print(f'# countyXnaics4 codes with more establishments at countyXnaics6 than countyXnaics4: {df4estnum.loc[(df4estnum["wages_missing6by4"]>0)&(df4estnum["estnumdiff"]<0),:].shape[0]}')
    #print(f'# countyXnaics4 codes with 1-10 establishments NOT accounted for in countyXnaics6: {df4estnum.loc[(df4estnum["wages_missing6by4"]>0)&(df4estnum["estnumdiff"]>0)&(df4estnum["estnumdiff"]<11),:].shape[0]}')
    #print(f'# countyXnaics4 codes with >10 establishments NOT accounted for in countyXnaics6: {df4estnum.loc[(df4estnum["wages_missing6by4"]>0)&(df4estnum["estnumdiff"]>10),:].shape[0]}')

    #print('Rows with greater than 11 establishments missing from countyXnaics6...')
    #print(df4estnum.loc[(df4estnum["wages_missing6by4"]>0)&(df4estnum["estnumdiff"] > 11), ['geoindkey','wages_cbp_flag','row_sources','estnum','estnumdiff','wagesdiff','wages_missing6by4','wages_propmissing6by4']])

    df4['I_emp3_negdiff']=((df4['emp3diff']<0)&(df4['emp3'].notna()))
    df4['I_wages_negdiff'] = ((df4['wagesdiff'] < 0) & (df4['wages'].notna()))
    df4['I_both_negdiff']=((df4['emp3diff']<0)&(df4['wagesdiff']<0)&(df4['wages'].notna()))
    print('2-way table of Indicators for emp3 difference is negative and wages difference is negative.')
    print(pd.crosstab(df4["I_emp3_negdiff"],df4['I_wages_negdiff']))

    # Check for countyXnaics4 codes which have no corresponding countyXnaics6 codes
    weirdones = df4.loc[df4["_merge"] == "left_only"]  # only appear in county by NAICS-4 codes
    geo6naicsweird = df6.loc[df6["geo4naics"].isin(weirdones['geo4naics']), :]
    for vname in ["wages","emp3"]:
        if len(geo6naicsweird) == 0:  ##If no county by NAICS6 codes, hard code the relevant values
            df4.loc[df4[vname + "_sum6by4"].isna(), vname + "_sum6by4"] = 0
            df4.loc[df4[vname + "_missing6by4"].isna(), vname + "_missing6by4"] = 0
            df4.loc[df4[vname + "_propmissing6by4"].isna(), vname + "_propmissing6by4"] = 1
            df4.loc[df4['count6by4codes'].isna(), 'count6by4codes'] = 0
        else:  # otherwise print diagnostic information
            print(f'count na in {vname}_sum6by4 {sum(df4.loc[:, vname + "_sum6by4"].isna())}')
            print(f'in {vname}_missing6by4 {sum(df4.loc[:, vname + "_missing6by4"].isna())}')
            raise Exception(f"Something wrong\n {geo6naicsweird.head()}")
        df4.drop(columns=["_merge",'emp3_missing6by4','emp3_propmissing6by4'], errors="ignore",inplace=True)
        df4.drop(columns=[dropcol for dropcol in df4.columns if "_droplater" in dropcol], errors="ignore",inplace=True)
        print(df4.columns)
    #df4['cbp_flags_equal']=df4['emp3_']
    ## Get difference between county by NAICS4 and sum of known county by NAICS6
    #df4[vname + "diff"] = df4[vname].astype(float) - df4[vname + '_sum6by4'].astype(float)
    negdiff = ((df4[vname+"diff"] < 0)&(df4[vname].notna()))
    source_notqcew = (df4[vname+'_source'] != "qcew")
    num_missing=df4['wages_missing6by4']

    ## When wagesdiff is negative...
    # CASE 1:
    # if there ARE missing countyXnaics6 cells, and the countyXnaics4 source is not "qcew",
    # then override vname, vname_source, and diff to be NA

    if justdrop:
        df4.loc[(negdiff) & (num_missing > 0) & (source_notqcew), vname] = np.nan
        df4.loc[(negdiff) & (num_missing > 0) & (source_notqcew), vname+"_source"] = np.nan
        df4.loc[(negdiff) & (num_missing > 0) & (source_notqcew), vname+"diff"] = np.nan
    else: ##### TO DO
        ## After some investigation and thought, I have decided that dropping county by NAICS-4 values which lead to negative differences is the best solution.
        ## There values will be imputed like the other missing county by NAICS-4 values.
        print('currently no method to adjust non-qcew County by NAICS-4 cells with negative differences and missing NAICS-6 cells. just dropping to refit.')
        df4.loc[(negdiff) & (num_missing > 0) & (source_notqcew), vname] = np.nan
        df4.loc[(negdiff) & (num_missing > 0) & (source_notqcew), vname + "_source"] = np.nan
        df4.loc[(negdiff) & (num_missing > 0) & (source_notqcew), vname + "diff"] = np.nan
    print(f'---- Adjusting incongruent county by NAICS-4 and by NAICS-6 {vname} ----')
    print(f'# negative difference in {vname}: {negdiff.sum()}\n# set cntyXnaics4 to NAN: {df4.loc[(negdiff) & (num_missing > 0) & (source_notqcew),:].shape[0]} (i.e. negative difference, >0 missing countyXnaics6, countyXnaics4 source NOT qcew)')# When missing>0:\n {pd.crosstab(negdiff[num_missing>0],df4.loc[num_missing>0,vname+"_source"])}')
    print(f'# set cntyXnaics4 to sum6by4: {df4.loc[(negdiff) & (num_missing == 0) & (source_notqcew),:].shape[0]} (i.e. negative difference, NO missing countyXnaics6, countyXnaics4 source NOT qcew)')
    subdf4qcew=df4.loc[(negdiff) & (df4[vname + "_source"] == "qcew"), :]
    print(f'# countyXnaics4 to adjust countyXnaics6: {subdf4qcew.shape[0]} (i.e. negative difference, countyXnaics4 source IS qcew)')
    #print(f'When missing==0:\n {pd.crosstab(negdiff[num_missing==0],df4.loc[num_missing==0,vname+"_source"])}')
    # CASE 2:
    # if there are NO countyXnaics6 cells, and the countyXnaics4 source is not "qcew",
    # then override vname=vname_sum6by4, vname_source='sum6by4', and diff=0
    df4.loc[(negdiff) & (num_missing == 0) & (source_notqcew), vname+"diff"] = 0
    df4.loc[(negdiff) & (num_missing == 0) & (source_notqcew), vname] = df4.loc[
    (negdiff) & (num_missing == 0) & (source_notqcew), vname+"_sum6by4"]
    df4.loc[(negdiff) & (num_missing == 0) & (source_notqcew), vname+"_source"] = "sum6by4"
    ### CASE 3a:
    ## if there vname_source is qcew at countyXnaics4 level and NO missing countyXnaics6,
    ## then must adjust county by NAICS-6 non-qcew values
    #subdf4,df6,df=adjust_nonqcew_negdiff_nomissing(subdf4=subdf4qcew[subdf4qcew[vname+'_missing6by4']==0,:],df6=df6,df=df,vname=vname)
    ## CASE 3b:
    # if there vname_source is qcew at countyXnaics4 level and >0 missing countyXnaics6,
    # then must adjust county by NAICS-6 non-qcew values

    subdf4, df6, df = adjust_nonqcew_negdiff_yesmissing(subdf4=subdf4qcew.loc[subdf4qcew[vname+'_missing6by4']>0,:],
                                                    df6=df6, df=df, vname=vname)

    ## When diff is POSITIVE, and only 1 countyXnaics6 is missing
    posdiff = (df4[vname + "diff"] > 0)
    # set the missing countyXnaics6 to the diff
    geo4naics_filldig6=df4.loc[(posdiff)&(num_missing==1),['geo4naics',vname+'diff']]
    geo4naics_filldig6['fill_source']=vname+"diff"
    df6 = df6.merge(geo4naics_filldig6, on="geo4naics", how="left")
    df6[vname] = df6[vname].fillna(df6[vname+"diff"])
    df6[vname+"_source"] = df6[vname+"_source"].fillna(df6['fill_source'])
    df6 = df6.drop(columns=[vname+"diff",'fill_source'])
    # AND update the df4 values to correspond
    df4.loc[(posdiff)&(num_missing==1),vname+'diff']=0
    df4.loc[(posdiff) & (num_missing == 1), vname + '_missing6by4'] = 0
    df4.loc[(posdiff) & (num_missing == 1), vname + '_sum6by4'] = df4.loc[(posdiff) & (num_missing == 1), vname]
    return df4,df6


def adjust_nonqcew_negdiff_nomissing(subdf4,df6,df,vname="wages",cbpflagdf=hardcode_cbp_flags):
    print(f'inside adjust_nonqcew_negdiff {vname}')
    subdf6=df6.loc[df6['geo4naics'].isin(subdf4['geo4naics']),:]
    print(subdf6.columns)
    print(subdf6.head())
    print(subdf6[vname+"_cbp_flag"].value_counts())

    stophere=True
    if stophere:
        raise Exception("stop here")
    return df4, df6, df

def adjust_nonqcew_negdiff_yesmissing(subdf4,df6,df,vname="wages",cbpflagdf=hardcode_cbp_flags):
    print(f'inside adjust_nonqcew_negdiff_yesmissing {vname}')
    df['geo4naics']=df['geoindkey'].str.slice(stop=-2)
    subdf4['prop_wages_from_qcew']=subdf4['count_wages_from_qcew']/subdf4['count6by4codes']
    print(subdf4.columns)
    #subdf4['wages_avgperest_qcew']=subdf4['sum_wages_qcew']/subdf4['count6by4codes']
    print(subdf4[['prop_wages_from_qcew','wages_missing6by4','wagesdiff']].describe())
    subdf4.drop(columns=['count_wages_from_cbp','wages_propmissing6by4','grouplevels'],inplace=True,errors="ignore")
    subdf6=df6.loc[df6['geo4naics'].isin(subdf4['geo4naics']),:]
    subdf6["vname_perest"]=subdf6[vname]/subdf6['estnum']
    subdf6['vname_perest_qcew']=subdf6['vname_perest']
    subdf6.loc[subdf6[vname+"_source"]!="qcew",'vname_perest_qcew']=np.nan
    subdf6['vname_peremp3']=subdf6[vname]/subdf6['emp3']
    subdf6['vname_peremp3_qcew']=subdf6['vname_peremp3']
    subdf6.loc[subdf6[vname+"_source"]!="qcew",'vname_peremp3_qcew']=np.nan
    subdf6gr=subdf6.groupby("geo4naics").agg(vname_avgperest=('vname_perest',"mean"),
                                             vname_medperest=('vname_perest', "median"),
                                             vname_avgperest_qcew=('vname_perest_qcew',"mean"),
                                             vname_medperest_qcew=('vname_perest_qcew',"median"),
                                             vname_avgperemp3=('vname_peremp3', "mean"),
                                             vname_medperemp3=('vname_peremp3', "median"),
                                             vname_avgperemp3_qcew=('vname_peremp3_qcew', "mean"),
                                             vname_medperemp3_qcew=('vname_peremp3_qcew', "median")
                                             )
    print(subdf6gr.head())
    subdf6gr.rename(columns={'vname_avgperest':vname+"_avgperest",
                             'vname_medperest':vname+"_medperest",
                             'vname_avgperest_qcew':vname+"_avgperest_qcew",
                             'vname_medperest_qcew':vname+"_medperest_qcew",
                             'vname_avgperemp3':vname+"_avgperemp3",
                             'vname_medperemp3':vname+"_medperemp3",
                             'vname_avgperemp3_qcew':vname+"_avgperemp3_qcew",
                             'vname_medperemp3_qcew':vname+"_medperemp3_qcew"},errors="ignore",inplace=True)
    subdf4=subdf4.merge(subdf6gr,on=['geo4naics'],how='left',indicator=False)
    print(subdf4[subdf4['count_wages_from_qcew']>2].head())
    #with Pool(processes=3) as pool:
    #    args = [(x, df6_toget, df4n) for x in test4dig]
    #    results = pool.starmap(process_chunk, args)
    # Combine results
    #df6_toget_imputed = pd.concat(results, ignore_index=True)
    # subdf=df.loc[df['geo4naics'].isin(subdf4['geo4naics']),:]
    # subcount6=get_codes_summary(subdf,groupbydigits=4, levelgrouped=6,variable=vname,include_source=False,onlyQCEW=True,perestab_stats=True)
    # subdf4gr = subdf4.merge(subcount6, on=['geo4naics'], how='left', indicator=False, suffixes=["_orig", "_qcewonly"])
    # subdf4gr.drop(columns=[dropcol for dropcol in subdf4gr.columns if "_droplater" in dropcol], errors="ignore",inplace=True)
    # print(subdf4gr.head())
    # subdf4gr[vname+"diff"]=subdf4gr[vname]-subdf4gr[vname+"_sum6by4"]
    # print(subdf4gr.head())
    # #print(subdf6.columns)
    # #print(subdf6.head())
    # #print(subdf6[vname+"_cbp_flag"].value_counts())

    stophere=True
    if stophere:
        raise Exception("stop here")
    return df4, df6, df

def avgestnum_source_adjustment(dfin,check_consistency=True,keep_only_filled_emp3=True):
    df=dfin.copy()
    if keep_only_filled_emp3:
        df=df.loc[df["emp3"].notna(),:]
    df=df.loc[df['estnum'].notna(),:]
    df[['wages_old', 'emp1_old', 'emp2_old', 'emp3_old', 'estnum_old']]=df[['wages', 'emp1', 'emp2', 'emp3', 'estnum']]
    df=adjust_by_estnum(df,groupbydigits=4,levelgrouped=6)
    df=adjust_by_estnum(df,groupbydigits=5,levelgrouped=6)
    df=adjust_by_estnum(df,groupbydigits=3,levelgrouped=4)
    #df=adjust_by_estnum(df,groupbydigits=2,levelgrouped=3)
    for vname in ["wages","emp1","emp2","emp3"]:
        df[vname]=df[vname+"_perestnum"]*df['estnum']
        df[vname]=df[vname].round(0)
    levelgrouped=6
    for grby_agglvl in [77,76,75]:
        groupbydigits=grby_agglvl-72
        labelstr="6by"+str(groupbydigits)
        if check_consistency:
            countlvldig = get_codes_summary(df, groupbydigits=groupbydigits, levelgrouped=levelgrouped, variable="estnum",
                                            onlyQCEW=False, include_source=False)
            grby_indic = (df['agglvl_code'] == grby_agglvl)
            dfgrby = df[grby_indic]
            dfestnum = dfgrby.merge(countlvldig, on=['geo' + str(groupbydigits) + 'naics'], how='left', indicator=False,
                                    suffixes=["", "_droplater"])
            dfestnum['estnum_diff']=dfestnum['estnum']-dfestnum['estnum_sum'+labelstr]
            notzero=(dfestnum['estnum_diff'].round(0)!=0).sum()
            if notzero != 0:
                print(
                    f'There are {notzero} countyXnaics{groupbydigits} cells with inconsistent establishment numbers to the countyXnaics6 cells. ( {100 * notzero / (dfestnum.shape[0])}%)')
                print(dfestnum.loc[dfestnum['estnum_diff'] != 0, ['geoindkey', 'estnum', 'estnum_sum' + labelstr,
                                                                  'estnum_missing' + labelstr, "estnum_diff",
                                                                  "estnum_source", "estnum_cbp", "estnum_qcew"]].head())
    countlvldig = get_codes_summary(df, groupbydigits=3, levelgrouped=4, variable="emp3",
                                    onlyQCEW=False, include_source=False)
    #print(countlvldig.describe())

    #print(countlvldig.loc[countlvldig["emp3_missing4by3"]>1,:].head())
    print(f'Currently {countlvldig.loc[countlvldig["emp3_missing4by3"]>1,:].shape[0]} emp3 values at countyXnaics3 level with only 1 missing countyXnaics4 cell.')
    # miss1=countlvldig.loc[countlvldig["emp3_missing4by3"] == 1, ["emp3_sum4by3", "geo3naics"]]
    # df75 = df.loc[df['agglvl_code']==75,["geo3naics","emp3","emp3_source"]].merge(miss1,on="geo3naics",how="left",indicator=False)
    # df75['emp3_diff']=df75["emp3"]-df75['emp3_sum4by3']
    # df=df.merge(df75[["geo3naics","emp3_diff"]],on="geo3naics",how="left",indicator=False)
    # df.loc[(df["geo3naics"].isin(df75['geo3naics']))&(df['agglvl_code']==76)&(df['emp3'].isna()),'emp3']=df.loc[(df["geo3naics"].isin(df75['geo3naics']))&(df['agglvl_code']==76)&(df['emp3'].isna()),'emp3_diff']
    # df.drop(columns="emp3_diff",inplace=True)
    # countlvldig = get_codes_summary(df, groupbydigits=3, levelgrouped=4, variable="emp3",
    #                                 onlyQCEW=False, include_source=False)
    # #print(countlvldig.describe())
    #
    # #print(countlvldig.loc[countlvldig["emp3_missing4by3"] ==1, :].head())
    # #print(f'Now {countlvldig.loc[countlvldig["emp3_missing4by3"]>1,:].shape[0]} emp3 values at countyXnaics3 level with only 1 missing countyXnaics4 cell.')
    #
    # #raise Exception("stop here")
    #print(df.loc[df["estnum"]!=df["estnum_old"],
    #             ["geoindkey","estnum","estnum_old","estnum_source","emp1","emp1_old","emp2","emp2_old","emp3","emp3_old","wages","wages_old","emp1_source","emp2_source","emp3_source","wages_source"]].head())
    #print(df.loc[df['geoindkey'].str.startswith("1001_113"),:].shape[0])
    #print(df.loc[df['geoindkey'].str.startswith("1001_113"),["geoindkey","estnum","estnum_old","estnum_source"]].head())



    return df

def adjust_by_estnum(dfin2,groupbydigits=4, levelgrouped=6,justestnum=True,check_consistency=True):
    df=dfin2.copy()
    if groupbydigits<2 or groupbydigits>5:
        raise Exception(f'Code does not currently support adjustments to the county level values. groupbydigits should be 2,3,4, or 5.')
    grby_agglvl=72+groupbydigits
    str_end_idx = -(6 - groupbydigits)
    df['geo'+str(groupbydigits)+'naics'] = df['geoindkey'].str.slice(stop=str_end_idx)
    countlvldig = get_codes_summary(df, groupbydigits=groupbydigits, levelgrouped=levelgrouped, variable="estnum", onlyQCEW=False,include_source=False)
    grby_indic=(df['agglvl_code']==grby_agglvl)
    dfgrby=df[grby_indic]
    dfestnum = dfgrby.merge(countlvldig, on=['geo'+str(groupbydigits)+'naics'], how='left', indicator=False, suffixes=["", "_droplater"])
    dfestnum.drop(columns=[dropcol for dropcol in dfestnum.columns if "_droplater" in dropcol], errors="ignore",
                   inplace=True)
    labelstr=str(levelgrouped)+"by"+str(groupbydigits)
    if justestnum: #just adjusting the establishment number (adjust the variable values in avgestnum_source_adjustment function)
        dfestnum['estnum'] = dfestnum['estnum_sum'+labelstr]
        df = df.merge(dfestnum[['geoindkey', 'estnum']], on="geoindkey", how="left",
                      indicator=False, suffixes=["", "_adj"])
        df.loc[df['agglvl_code']==grby_agglvl,'estnum']=df.loc[df['agglvl_code']==grby_agglvl, 'estnum_adj']
        df.drop(columns='estnum_adj',inplace=True,errors="ignore")
    else:
        for vname in ['wages', 'emp1', 'emp2', 'emp3']:
            vname_grby_source=dfestnum[vname+"_source"]
            vname_grby_source[vname_grby_source=="qwi"]="qcew"
            estnum_s=dfestnum["estnum_qcew"]
            estnum_s[vname_grby_source=="cbp"]=dfestnum.loc[vname_grby_source=="cbp","estnum_cbp"]
            dfestnum[vname + '_perest'] = dfestnum[vname] /estnum_s
            # df4estnum['old_'+vname]=df4estnum[vname]
            dfestnum[vname] = dfestnum[vname + "_perest"] * dfestnum['estnum_sum'+labelstr]
            dfestnum.drop(columns=[vname + "_perest"], inplace=True)
        dfestnum['estnum'] = dfestnum['estnum_sum'+labelstr]

        df = df.merge(dfestnum[['geoindkey', 'wages', 'emp1', 'emp2', 'emp3', 'estnum']], on="geoindkey", how="left",
                      indicator=False, suffixes=["", "_adj"])
        df.loc[df['agglvl_code'] == grby_agglvl, ['wages', 'emp1', 'emp2', 'emp3', 'estnum']] = df.loc[
            df['agglvl_code'] == grby_agglvl, ['wages_adj', 'emp1_adj', 'emp2_adj', 'emp3_adj', 'estnum_adj']]
        df.drop(columns=['wages_adj', 'emp1_adj', 'emp2_adj', 'emp3_adj', 'estnum_adj'], inplace=True, errors="ignore")
        #df=df.merge(dfestnum[['geoindkey','wages', 'emp1', 'emp2', 'emp3','estnum']],on="geoindkey",how="left",indicator=False,suffixes=["_old",""])
        #df.loc[grby_indic,['old_wages', 'old_emp1', 'old_emp2', 'old_emp3','old_estnum']]=df.loc[grby_indic,['wages', 'emp1', 'emp2', 'emp3','estnum']]
        #df.loc[grby_indic,['wages', 'emp1', 'emp2', 'emp3','estnum']]=dfestnum[['wages', 'emp1', 'emp2', 'emp3','estnum']]
    if check_consistency:
        countlvldig = get_codes_summary(df, groupbydigits=groupbydigits, levelgrouped=levelgrouped, variable="estnum",
                                        onlyQCEW=False, include_source=False)
        grby_indic = (df['agglvl_code'] == grby_agglvl)
        dfgrby = df[grby_indic]
        dfestnum = dfgrby.merge(countlvldig, on=['geo' + str(groupbydigits) + 'naics'], how='left', indicator=False,
                                suffixes=["", "_droplater"])
        dfestnum['estnum_diff']=dfestnum['estnum']-dfestnum['estnum_sum'+labelstr]
        notzero=(dfestnum['estnum_diff'].round(0)!=0).sum()

        if notzero!=0:
            print(f'There are {notzero} countyXnaics{groupbydigits} cells with inconsistent establishment numbers to the countyXnaics{levelgrouped} cells. ( {100*notzero/(dfestnum.shape[0])}%)')
            print(dfestnum.loc[dfestnum['estnum_diff']!=0,['geoindkey','estnum','estnum_sum'+labelstr,'estnum_missing'+labelstr,"estnum_diff","estnum_source","estnum_cbp","estnum_qcew"]].head())
        # if not justestnum:
        #     for vname in ["wages","emp1","emp2","emp3"]:
        #         countlvldig = get_codes_summary(df, groupbydigits=groupbydigits, levelgrouped=levelgrouped,
        #                                         variable=vname,
        #                                         onlyQCEW=False, include_source=False)
        #         dfestnum = dfestnum.merge(countlvldig, on=['geo' + str(groupbydigits) + 'naics'], how='left', indicator=False,
        #                                 suffixes=["", "_droplater"])
        #         dfestnum[vname+'_diff'] = dfestnum[vname] - dfestnum[vname+'_sum' + labelstr]
        #         negdiff = ((dfestnum[vname].notna())&(dfestnum[vname+'_diff'].round(0)< 0)).sum()
        #         #print(
        #         #    f'{vname} in {labelstr}: {negdiff}  negative differences (i.e. {100*negdiff/(dfestnum[vname].notna().sum())}%)')
    return df




def adjust_nonqcew_negdiff_yesmissing_per_naics4():

    return np.nan #get_6naics_per4(x, df6=df6_toget, df4imp=df4n)