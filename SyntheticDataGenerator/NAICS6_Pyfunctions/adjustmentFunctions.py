import statsmodels.api as sm
from formulaic import Formula
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath('./'))
from GeneralFunctions import custom_predict
from hierarchy_geoindkey import *

# with open('./config.yaml','r') as configFile:
#     config = yaml.safe_load(configFile)
#     employmentConfig = config['employmentConfig']
pd.set_option('mode.chained_assignment', None)
# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Flag for development/testing mode - set to False in production
PRINTHEADS = False

# Hard-coded NAICS codes excluded from CBP (Census Bureau of Population) data
EXCLUDED_CBP = [
    "92----",   # Public administration
    "111///",   # Crop production
    "112///",   # Animal production
    "482///",   # Rail transportation
    "491///",   # Postal service
    "814///",   # Private households
    "525110",   # Open-end investment funds
    "525120",   # Closed-end investment funds
    "525190",   # Other financial vehicles
    "525920",   # Trusts, estates, and agency accounts
    "541120"    # Offices of other holding companies
]


# ============================================================================
# FUNCTIONS FOR ADJUSTING TO CORRECT INCONSISTENCIES AND TO ACCOUNT FOR SOURCE OR QUARTER
# ============================================================================


def quarter_source_adjustment(data,
                              generalConfig,
                              response,
                              quarterConfig=None,
                              formula=None,
                              adjust_source=True,
                              source="CBP",
                              rseed=None):
    '''
    Fit an OLS regression model for employment data adjustment across sources or quarters.

    PURPOSE:
        Creates an OLS model to predict employment values when direct data is missing.
        Handles two scenarios:
        1. Adjusting between different data sources (CBP to QCEW, etc.)
        2. Adjusting for quarterly differences within the same source

    WORKFLOW:
        1. Data Filtering: Remove suppressed/invalid data
            - Exclude rows where employment (sEmp/sEmpEnd) is suppressed
            - Exclude rows with ind_level = "A" (invalid aggregation level)

        2. Initial Model Fitting:
            - Use formula (from config or provided) to build OLS model
            - Default formula: 'Emp ~ EmpEnd + estnum + C(sector) + C(state)'

        3. Outlier Detection & Removal:
            - Calculate Cook's distance (threshold configurable, default: 1.0)
            - Calculate Studentized residuals (threshold configurable)
            - Remove observations exceeding thresholds

        4. Model Refitting:
            - Refit model on cleaned dataset without influential points

    PARAMETERS:
        data (pd.DataFrame):
            Input dataset containing employment metrics and covariates

        generalConfig (dict):
            From overall configuration file
            General configuration settings

        response (str):
            Name of the response variable to predict (e.g., 'Emp', 'wages_qcew')

        quarterConfig (dict, optional):
            From overall configuration file
            Configuration specific to quarterly adjustments
            Contains 'EMP_OLS_FORMULA', 'WAGE_OLS_FORMULA', 'DIAGNOSTIC_PLOTS'

        formula (str, optional):
            Custom regression formula (R/patsy style)
            If not provided, uses default from config or hardcoded defaults

        adjust_source (bool):
            True: Adjust between data sources
            False: Adjust for quarterly differences

        source (str):
            Name of data source ('CBP', 'QCEW', 'QWI') - used for labeling

        rseed (int, optional):
            Random seed for reproducibility

    RETURNS:
        Model object (statsmodels.OLS fitted model)
            - Can be used with custom_predict() to generate predictions
            - Contains coefficients, residuals, and diagnostic information

    SIDE EFFECTS:
        - Modifies input dataframe 'data' in-place:
            * Fills missing response values with predictions
            * Adds '{response}_source' column tracking data source
        - Prints model summary and diagnostic information
        - May remove influential points, adjusting dataset size
    '''

    # Set random seed for reproducibility if provided
    if rseed is not None:
        np.random.seed(rseed)
    #create y+quarter variable
    if "year_qtr" not in data.columns:
        data['year_qtr'] = data['year'] + data['qtr'].astype(float).multiply(0.25)

    # Create working copy to avoid modifying original
    tempdata = data.copy()

    if adjust_source:  # adjusting from one data source to another
        # ====================================================================
        # SCENARIO 1: ADJUSTING BETWEEN DATA SOURCES (Within same quarter)
        # ====================================================================
        if formula is None:
            formula = response + "~." #use all variables

        usenewsource = data.loc[:, response].isna()
        subdata = tempdata.loc[~usenewsource, :].copy()
    else:
        # ====================================================================
        # SCENARIO 2: ADJUSTING FOR QUARTERLY DIFFERENCES
        # ====================================================================
        tempdata['year_qtr_diff'] = tempdata['year_qtr_cbp'].astype(float) - tempdata['year_qtr'].astype(float)

        # Build formula stem based on whether data spans multiple quarters
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
            if response =="wages_qcew":
                formula = formula_stem + "wages_cbp+np.log10(estnum_cbp)+np.log10(estnum)+emp3_cbp+agglvl_code+naics2"
            else:
                formula = formula_stem + "emp3_cbp+np.log10(estnum_cbp)+np.log10(estnum)+wages_cbp+agglvl_code+naics2"

            # ensure variable type is correct
            # Ensure correct data types for numeric variables
            numeric_cols = [
                "year", "wages_cbp", "estnum_cbp", "estnum",
                "wages_qcew", "emp1_qcew", "emp2_qcew", "emp3_qcew", "emp3_cbp"
            ]
            for col in numeric_cols:
                tempdata[col] = tempdata[col].astype(float)

            # Ensure correct data types for categorical variables
            categorical_cols = [
                'qtr', 'qtr_cbp', 'wages_cbp_flag', 'emp3_cbp_flag',
                'agglvl_code', 'naics2', 'naics3', 'naics4', 'naics5'
            ]
            for col in categorical_cols:
                tempdata[col] = tempdata[col].astype("category")

        # dataset to fit model on
        subdata = tempdata[
            (~tempdata['wages_cbp'].isna()) &
            (~tempdata['wages_qcew'].isna()) &
            (~tempdata['emp3_cbp'].isna())].copy()

    # Create design matrices (gets the variables ready for fitting in statsmodels.OLS) using the formula
    # and perform initial model fitting
    y_pre, X_pre = Formula(formula).get_model_matrix(subdata)
    model = sm.OLS(y_pre, X_pre).fit()

    if adjust_source:  # adjusting datasources
        print("Model to adjust " + source + "  " + response)
        print(model.summary())

        # Get predictions and standard errors for rows with missing response
        pred, se_fit = custom_predict(tempdata[usenewsource], model, rseed=rseed)

        # Fill missing response values with rounded predictions
        data.loc[usenewsource, response] = np.round(pred, decimals=0)

        # Track data source for imputed values
        if response + "_source" not in data.columns:
            data.loc[:, response + "_source"] = ""
        data.loc[usenewsource, response + "_source"] = source.lower()
        data.loc[data[response].isna(), response + "_source"] = ""

    else: #adjust for quarter
        if quarterConfig is not None and quarterConfig['DIAGNOSTIC_PLOTS'] is not None:
            save_diagnostic_plots(model, formula, quarterConfig['DIAGNOSTIC_PLOTS'])
        print("Model to adjust CBP " + response + " to quarter " + str(generalConfig['QTR']))
        print(model.summary())

        split_response = response.split("_")
        split_response.pop()
        response_stem = "_".join(split_response)
        data.loc[subdata.index.tolist(), response_stem + "_cbp"] = model.fittedvalues()

    return data


def get_varmin(codes4naics, fulldf, variable="emp1",onlyqcew=False):
    '''
    What is the point?
        get_varmin() calculates lower bounds for variable ("emp1","emp2","emp3",or "wages") using 6-digit NAICS summaries
    Inputs:
        1. codes4naics - Array of 4-digit NAICS codes
        2. fulldf - Complete dataset with wage information
        3. variable - string which indicates which variable we are getting the minimum of ("emp1","emp2","emp3","wages")
    Returns:
        DataFrame with geoindkey and calculated minwage values
    '''
    # Get 6-digit NAICS summaries
    #print(f"in varmin, codes NAICS are: {codes4naics}")
    #print(fulldf.head())
    tomerge6dig = get_codes_summary(dfin=fulldf, groupbydigits=4, levelgrouped=6, variable=variable,include_estab_emp3_stats=False,naicsdf=codes4naics,onlyQCEW=onlyqcew)
    #print(tomerge6dig.head())
    # Create minwage column (0 if no data available)
    tomerge6dig['min' + variable] = np.where(tomerge6dig[variable + '_sum6by4'].isna(), 0,
                                             tomerge6dig[variable + '_sum6by4'])
    tomerge6dig['geoindkey'] = tomerge6dig['geo4naics'].astype(str) + "//"
    tomerge6dig = tomerge6dig[['geoindkey', 'min' + variable]]
    return tomerge6dig


def adjust_geo4naics_varvalues(fitdf, dfmaxmin=None, variable="emp1", fulldf=None, onlyqcew=False,
                               minonly=True):
    '''
    What is the point?
        adjust_geo4naics_varvalues() constrains wage/emp estimates to stay within min/max bounds
    Inputs:
        1. fitdf - DataFrame with estimates
        2. dfmaxmin - DataFrame with min/max bounds (if none, then fulldf must be the full data without the estimates)
        3. variable- string name of variable to be adjusted
        4. adjust_indic- series of indicators to determine which of fitdf[variable] can be adjusted.
        5. fulldf- if dfmaxmin is not provided, them fulldf must be the full data without the estimates
        6. onlyqcew- if True indicates only variable values with source "qcew" will be used to set min and max values
        7. minonly- if True indicates the variable will only be adjusted with the minimum
    Returns:
        DataFrame with adjusted wage values
    '''
    # Merge with min/max bounds
    if dfmaxmin is None:
        fulldf['geo4naics'] = fulldf['geoindkey'].str.slice(stop=-2)
        df4 = fulldf[fulldf['agglvl_code'] == 76].copy()
        dfmaxmin = get_varmaxmindf(df4dig=df4, fulldf=fulldf, variable=variable, onlyqcew=onlyqcew)
    if 'geo4naics' not in fitdf.columns:
        fitdf['geo4naics'] = fitdf['geoindkey'].str.slice(stop=-2)
    maxmindf = dfmaxmin[['geo4naics', 'min' + variable, 'max' + variable,
                         "max" + variable + "_source"]].copy()

    # maxmindf = dfmaxmin[['geo4naics', 'min' + variable, 'max' + variable,'max'+variable+'_qcewsource',"max"+variable+"_source"]].copy()
    fitdf = fitdf.merge(maxmindf, on='geo4naics', how='left')
    fitdf['min' + variable + '_source'] = 'hierarchy'
    fitdf.loc[fitdf['min' + variable] == 0, 'min' + variable + '_source'] = 'structural'
    # if stabvals is not None and "emp" in variable:
    #    fitdf.loc[:, "min" + variable] = np.fmin(fitdf["min" + variable].to_numpy(), stabvals.to_numpy())
    #    fitdf.loc[fitdf['min'+variable]==stabvals,'min_source']="stable_emp"

    if PRINTHEADS:
        ## check given qcew values
        fitdf['value_status'] = "within calculated bounds"
        fitdf.loc[(fitdf[variable] < fitdf['min' + variable]), 'value_status'] = "below calculated min"
        fitdf.loc[(fitdf[variable] > fitdf['max' + variable]), 'value_status'] = "above calculated max"
        print(
            f'When adjusting {variable}: \n{pd.crosstab(fitdf["value_status"], fitdf[variable + "_source"], dropna=False)}')
        abovedf = fitdf.loc[(fitdf["value_status"] == "above calculated max"), [variable + "_source",
                                                                                "max" + variable + "_qcewsource",
                                                                                "max" + variable + "_source"]].groupby(
            [variable + "_source", "max" + variable + "_source"]).describe()
        print(f'Above Calculated Max prop from qcew:\n {abovedf}')
        abovedf = fitdf.loc[
            (fitdf["value_status"] == "above calculated max"), [variable + "_source", "max" + variable + "_source"]]
        print(
            f'Above Calculated Max prop from qcew:\n {pd.crosstab(abovedf[variable + "_source"], abovedf["max" + variable + "_source"])}')
        fitdf.drop(columns="value_status", inplace=True)
        print(
            f'Check Maxes\n{fitdf.loc[fitdf["max" + variable].notna(), [variable, variable + "_source", "min" + variable, "max" + variable]].head()}')

    # Apply constraints
    if minonly:
        fitdf[variable] = fitdf[variable].clip(lower=fitdf['min' + variable].astype(float))
    else:
        fitdf[variable] = fitdf[variable].clip(
            lower=fitdf['min' + variable].astype(float),
            upper=fitdf['max' + variable].astype(float)
        )
    # fixed=fitdf.loc[below_min_notqcew.index.values,[variable,variable+"_source",'min'+variable,'max'+variable,'min_source']]
    # print(f'Fixed bounds?\n{fixed.head()}')
    # fitdf = fitdf.drop(columns=['min' + variable, 'max' + variable,'min_source'])
    return fitdf


def get_varmax(codes4naics, fulldf, variable="emp1",onlyqcew=False):
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
        "max" + variable: np.nan,
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

    tomergedf3[variable + "_naics3"] = tomergedf3[
        variable]  # np.where(tomergedf3[variable].notna(), np.nan, tomergedf3[variable])
    tomergedf3[variable + "_source_naics3"] = tomergedf3[variable + "_source"]
    tomergedf3 = tomergedf3[["geo3naics", "estnum", variable + "_naics3", variable + "_source_naics3"]]
    # Merge 3-digit data
    outdf = outdf.merge(tomergedf3, on='geo3naics', how='left', suffixes=('', '_naics3'))
    # For missing values, try 2-digit sector level
    notmaxcodes = outdf[outdf[variable + '_naics3'].isna()]['geo2naics'].tolist()
    fulldf['geo2naics'] = fulldf['geoindkey'].str.slice(stop=-4)
    tomergedf2 = fulldf[fulldf['geo2naics'].isin(notmaxcodes) &
                        fulldf['geoindkey'].str.contains(r"_[0-9]{2}[^0-9]{4}")].copy()
    tomergedf2[variable + '_naics2'] = tomergedf2[
        variable]  # np.where(tomergedf2[variable].notna(), np.nan, tomergedf2[variable])
    tomergedf2[variable + '_source_naics2'] = tomergedf2[variable + "_source"]
    tomergedf2 = tomergedf2[['geo2naics', 'estnum', variable + '_naics2', variable + "_source_naics2"]]
    # Calculate differences between sector and summed 3-digit wages
    tomergedf3[variable + '_naics3'] = tomergedf3[variable + '_naics3'].astype(float)
    tomergedf3['geo2naics'] = tomergedf3['geo3naics'].str[:-1]  # Extract sector codes
    tomergedf3grouped = tomergedf3.groupby('geo2naics', as_index=False).agg(sumvar3=(variable + '_naics3', 'sum'))

    extrainvestigate = False
    if extrainvestigate:
        ## extra investigation
        tallysource = tomergedf3.groupby(['geo2naics', variable + '_source_naics3'], dropna=False).size().to_frame(
            "count")
        tallysource['prop'] = tallysource['count'] / tallysource.groupby(level=0)['count'].transform('sum')
        tallysource = tallysource.reset_index().pivot_table(
            index='geo2naics', columns=variable + "_source_naics3", values=["count", "prop"],
            dropna=False)  # fill_value=0, dropna=False)
        tallysource.columns = tallysource.columns.map(lambda index: f'{variable}_{index[0]}_source_{index[1]}_naics3')
        tallysource = tallysource.reset_index()
        # print(f'tallysource in tomergedf3 groupby geo2naics head after pivot\n{tallysource.head(10)}')
        tomergedf3 = tomergedf3grouped.merge(tallysource, on='geo2naics', how="left")
    else:
        tomergedf3 = tomergedf3grouped
    tomergedf2 = tomergedf2.merge(tomergedf3, on='geo2naics', how='left')
    tomergedf2['missing_' + variable + '_naics2'] = tomergedf2[variable + '_naics2'].astype(float) - tomergedf2[
        'sumvar3']
    # print(f"describe missing geo2naics \n{tomergedf3['missing_'+variable+'_naics2'].describe()}")
    if extrainvestigate:
        tomergedf2 = tomergedf2[['geo2naics', 'missing_' + variable + '_naics2', 'estnum', variable + '_naics2',
                                 variable + '_source_naics2', variable + '_prop_source_qcew_naics3']]
    else:

        tomergedf2 = tomergedf2[['geo2naics', 'missing_' + variable + '_naics2', 'estnum', variable + '_naics2',
                                 variable + '_source_naics2']]
    # Merge sector-level data
    outdf = outdf.merge(tomergedf2, on='geo2naics', how='left', suffixes=('', '_naics2'))
    # print(f'outdf head before getting max \n {outdf.head()}')
    outdf['max' + variable] = outdf.apply(
        lambda row: row[variable + '_naics3'] if pd.notna(row[variable + '_naics3']) else row[
            'missing_' + variable + '_naics2'], axis=1)
    if extrainvestigate:
        outdf['max' + variable + "_qcewsource"] = outdf[variable + '_prop_source_qcew_naics3']
        outdf.loc[(outdf['max' + variable] == outdf[variable + "_naics3"]) & (
                outdf['max' + variable + '_source'] == "qcew"), 'max' + variable + "_qcewsource"] = 1
        outdf.loc[(outdf['max' + variable] == outdf[variable + "_naics3"]) & (
                outdf['max' + variable + '_source'] != "qcew"), 'max' + variable + "_qcewsource"] = 0

    outdf['max' + variable + "_source"] = ""
    outdf.loc[outdf['max' + variable] == outdf[variable + "_naics3"], "max" + variable + "_source"] = "geo3naics"
    outdf.loc[outdf['max' + variable] == outdf[
        "missing_" + variable + "_naics2"], "max" + variable + "_source"] = "missing_geo2naics"

    # print(f'inside get_varmax, head of outdf after max{variable} added when max{variable} is not na \n {outdf.loc[outdf["max"+variable].notna(),:].head()}')
    outdf = outdf.drop(columns=[variable + '_naics2'])
    fulldf[variable] = fulldf[variable].astype(float)
    # For remaining missing values, use county-wide totals
    max_allind_allcounty = fulldf[fulldf['agglvl_code'] == 76][variable].max(skipna=True)
    # print(f'number of maxes from allind_allcounty {outdf["max"+variable].isna().sum()}')
    if outdf['max' + variable].isna().sum() > 0:
        # print(f'number of maxes from allind_allcounty {outdf["max"+variable].isna().sum()}')
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
            tallysource = tomergedf2.groupby(['geography', variable + '_source_naics2'], dropna=False).size().to_frame(
                "count")
            tallysource['prop'] = tallysource['count'] / tallysource.groupby(level=0)['count'].transform('sum')
            tallysource = tallysource.reset_index().pivot_table(
                index='geography', columns=variable + "_source_naics2", values=["count", "prop"],
                dropna=False)  # fill_value=0, dropna=False)
            tallysource.columns = tallysource.columns.map(
                lambda index: f'{variable}_{index[0]}_source_{index[1]}_naics2')
            tallysource = tallysource.reset_index()
            # print(f'tallysource in tomergedf3 groupby geo2naics head after pivot\n{tallysource.head(10)}')
            tomergedf2 = tomergedf2group.merge(tallysource, on='geography', how="left")
        else:
            tomergedf2 = tomergedf2group
        tomergedfall = tomergedfall.merge(tomergedf2, on="geography", how="left")
        tomergedfall['missing' + variable + 'all'] = tomergedfall[variable + 'all'].astype(float) - tomergedfall[
            'sum' + variable + '2'].astype(float)

        if extrainvestigate:
            tomergedfall = tomergedfall[['geography', 'missing' + variable + 'all', 'estnum', variable + 'all',
                                         variable + '_prop_source_qcew_naics2']]
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
        outdf = outdf[['geoindkey', 'max' + variable, "max" + variable + "_qcewsource", "max" + variable + "_source"]]
    else:

        outdf = outdf[['geoindkey', 'max' + variable, "max" + variable + "_source"]]
    return outdf


def get_varmaxmindf(df4dig, fulldf, variable="emp1", onlyqcew=False):
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
    # print(f'inside get_varmaxmin before get_varmin. columns of fulldf: {fulldf.columns}')
    if onlyqcew:
        fulldf = fulldf.loc[fulldf[variable + "_source"] == "qcew", :].copy()

    mindf = get_varmin(codes4naics=df4dig['geoindkey'], fulldf=fulldf, variable=variable,onlyqcew=onlyqcew)
    maxdf = get_varmax(codes4naics=df4dig['geoindkey'], fulldf=fulldf, variable=variable,onlyqcew=onlyqcew)
    # Merge all data
    df4_maxmin = df4dig.merge(maxdf, on="geoindkey", how="left") \
        .merge(mindf, on="geoindkey", how="left")
    df4_maxmin['min' + variable] = df4_maxmin['min' + variable].fillna(0)
    return df4_maxmin


def adjust_negative_diff(df4, df, justdrop=True):
    # get summary of countyXnaics6 cells by countyXnaics4 codes for estnum, wages, and emp3
    count6dig = get_codes_summary(df, groupbydigits=4, levelgrouped=6, variable="estnum", onlyQCEW=False,
                                  include_source=False,include_estab_emp3_stats=False)
    df4estnum = df4.merge(count6dig, on=['geo4naics'], how='left', indicator=False, suffixes=["", "_droplater"])
    df4estnum.drop(columns=[dropcol for dropcol in df4estnum.columns if "_droplater" in dropcol], errors="ignore",
                   inplace=True)
    for vname in ['wages', 'emp1', 'emp2', 'emp3']:
        df4estnum[vname + '_perest'] = df4estnum[vname] / df4estnum['estnum']
        # df4estnum['old_'+vname]=df4estnum[vname]
        df4estnum[vname] = df4estnum[vname + "_perest"] * df4estnum['estnum_sum6by4']
        df4estnum.drop(columns=[vname + "_perest"], inplace=True)
    df4estnum['estnum'] = df4estnum['estnum_sum6by4']
    df4estnum.set_index(df4estnum['geoindkey'], inplace=True, drop=True)
    df.set_index(df['geoindkey'], inplace=True, drop=False)
    df.loc[df['geoindkey'].isin(df4estnum.index.to_list()), ['wages', 'emp1', 'emp2', 'emp3', 'estnum']] = df4estnum[
        ['wages', 'emp1', 'emp2', 'emp3', 'estnum']]
    df.drop(columns=['geoindkey'], inplace=True)
    df.reset_index(inplace=True, drop=False)
    df4estnum.drop(columns=['geoindkey'], inplace=True)
    df4estnum.reset_index(inplace=True, drop=False)
    df4estnum.drop(columns=[dropcol for dropcol in df4estnum.columns if "6by4" in dropcol], errors="ignore",
                   inplace=True)
    df6 = df.loc[df['agglvl_code'] == 78, :]
    df6['geo4naics'] = df6['geoindkey'].str.slice(stop=-2)
    df6['geo5naics'] = df6['geoindkey'].str.slice(stop=-1)
    df6['geo3naics'] = df6['geoindkey'].str.slice(stop=-3)

    count6digwages = get_codes_summary(df, groupbydigits=4, levelgrouped=6, variable="wages", onlyQCEW=False,
                                       include_source=True,include_estab_emp3_stats=False)
    count6digemp3 = get_codes_summary(df, groupbydigits=4, levelgrouped=6, variable="emp3", onlyQCEW=False,
                                      include_source=True,include_estab_emp3_stats=False)
    df4estnum = df4estnum.merge(count6digwages, on=['geo4naics'], how='left', indicator=True,
                                suffixes=["", "_droplater"])
    df4estnum.drop(columns=[dropcol for dropcol in df4estnum.columns if "_droplater" in dropcol], errors="ignore",
                   inplace=True)

    ## Get difference between county by NAICS4 and sum of known county by NAICS6
    df4estnum["wagesdiff"] = df4estnum["wages"].astype(float) - df4estnum['wages_sum6by4'].astype(float)
    df4 = df4estnum.merge(count6digemp3, on=['geo4naics'], how='left', indicator=False, suffixes=["", "_droplater"])
    df4["emp3diff"] = df4["emp3"].astype(float) - df4['emp3_sum6by4'].astype(float)

    ## summarize difference in establishment counts
    ## This showed no cells had more establishments at the countyXnaics6 level than at the countyXnaics4 level
    # print(f'# countyXnaics4 codes with all establishments accounted for: {df4estnum.loc[df4estnum["estnumdiff"]==0,:].shape[0]}')
    # print(pd.crosstab(missingestnum['from_cbp_missing_naics6']))
    # print(f'When There are missing wage values in the countyXnaics6 cells...')
    # print(f'# countyXnaics4 codes with more establishments at countyXnaics6 than countyXnaics4: {df4estnum.loc[(df4estnum["wages_missing6by4"]>0)&(df4estnum["estnumdiff"]<0),:].shape[0]}')
    # print(f'# countyXnaics4 codes with 1-10 establishments NOT accounted for in countyXnaics6: {df4estnum.loc[(df4estnum["wages_missing6by4"]>0)&(df4estnum["estnumdiff"]>0)&(df4estnum["estnumdiff"]<11),:].shape[0]}')
    # print(f'# countyXnaics4 codes with >10 establishments NOT accounted for in countyXnaics6: {df4estnum.loc[(df4estnum["wages_missing6by4"]>0)&(df4estnum["estnumdiff"]>10),:].shape[0]}')

    # print('Rows with greater than 11 establishments missing from countyXnaics6...')
    # print(df4estnum.loc[(df4estnum["wages_missing6by4"]>0)&(df4estnum["estnumdiff"] > 11), ['geoindkey','wages_cbp_flag','row_sources','estnum','estnumdiff','wagesdiff','wages_missing6by4','wages_propmissing6by4']])

    df4['I_emp3_negdiff'] = ((df4['emp3diff'] < 0) & (df4['emp3'].notna()))
    df4['I_wages_negdiff'] = ((df4['wagesdiff'] < 0) & (df4['wages'].notna()))
    df4['I_both_negdiff'] = ((df4['emp3diff'] < 0) & (df4['wagesdiff'] < 0) & (df4['wages'].notna()))
    print('2-way table of Indicators for emp3 difference is negative and wages difference is negative.')
    print(pd.crosstab(df4["I_emp3_negdiff"], df4['I_wages_negdiff']))

    # Check for countyXnaics4 codes which have no corresponding countyXnaics6 codes
    weirdones = df4.loc[df4["_merge"] == "left_only"]  # only appear in county by NAICS-4 codes
    geo6naicsweird = df6.loc[df6["geo4naics"].isin(weirdones['geo4naics']), :]
    for vname in ["wages", "emp3"]:
        if len(geo6naicsweird) == 0:  ##If no county by NAICS6 codes, hard code the relevant values
            df4.loc[df4[vname + "_sum6by4"].isna(), vname + "_sum6by4"] = 0
            df4.loc[df4[vname + "_missing6by4"].isna(), vname + "_missing6by4"] = 0
            df4.loc[df4[vname + "_propmissing6by4"].isna(), vname + "_propmissing6by4"] = 1
            df4.loc[df4['count6by4codes'].isna(), 'count6by4codes'] = 0
        else:  # otherwise print diagnostic information
            print(f'count na in {vname}_sum6by4 {sum(df4.loc[:, vname + "_sum6by4"].isna())}')
            print(f'in {vname}_missing6by4 {sum(df4.loc[:, vname + "_missing6by4"].isna())}')
            raise Exception(f"Something wrong\n {geo6naicsweird.head()}")
        df4.drop(columns=["_merge", 'emp3_missing6by4', 'emp3_propmissing6by4'], errors="ignore", inplace=True)
        df4.drop(columns=[dropcol for dropcol in df4.columns if "_droplater" in dropcol], errors="ignore", inplace=True)
        print(df4.columns)
    # df4['cbp_flags_equal']=df4['emp3_']
    ## Get difference between county by NAICS4 and sum of known county by NAICS6
    # df4[vname + "diff"] = df4[vname].astype(float) - df4[vname + '_sum6by4'].astype(float)
    negdiff = ((df4[vname + "diff"] < 0) & (df4[vname].notna()))
    source_notqcew = (df4[vname + '_source'] != "qcew")
    num_missing = df4['wages_missing6by4']

    ## When wagesdiff is negative...
    # CASE 1:
    # if there ARE missing countyXnaics6 cells, and the countyXnaics4 source is not "qcew",
    # then override vname, vname_source, and diff to be NA

    if justdrop:
        df4.loc[(negdiff) & (num_missing > 0) & (source_notqcew), vname] = np.nan
        df4.loc[(negdiff) & (num_missing > 0) & (source_notqcew), vname + "_source"] = np.nan
        df4.loc[(negdiff) & (num_missing > 0) & (source_notqcew), vname + "diff"] = np.nan
    else:  ##### TO DO
        ## After some investigation and thought, I have decided that dropping county by NAICS-4 values which lead to negative differences is the best solution.
        ## There values will be imputed like the other missing county by NAICS-4 values.
        print(
            'currently no method to adjust non-qcew County by NAICS-4 cells with negative differences and missing NAICS-6 cells. just dropping to refit.')
        df4.loc[(negdiff) & (num_missing > 0) & (source_notqcew), vname] = np.nan
        df4.loc[(negdiff) & (num_missing > 0) & (source_notqcew), vname + "_source"] = np.nan
        df4.loc[(negdiff) & (num_missing > 0) & (source_notqcew), vname + "diff"] = np.nan
    print(f'---- Adjusting incongruent county by NAICS-4 and by NAICS-6 {vname} ----')
    print(
        f'# negative difference in {vname}: {negdiff.sum()}\n# set cntyXnaics4 to NAN: {df4.loc[(negdiff) & (num_missing > 0) & (source_notqcew), :].shape[0]} (i.e. negative difference, >0 missing countyXnaics6, countyXnaics4 source NOT qcew)')  # When missing>0:\n {pd.crosstab(negdiff[num_missing>0],df4.loc[num_missing>0,vname+"_source"])}')
    print(
        f'# set cntyXnaics4 to sum6by4: {df4.loc[(negdiff) & (num_missing == 0) & (source_notqcew), :].shape[0]} (i.e. negative difference, NO missing countyXnaics6, countyXnaics4 source NOT qcew)')
    subdf4qcew = df4.loc[(negdiff) & (df4[vname + "_source"] == "qcew"), :]
    print(
        f'# countyXnaics4 to adjust countyXnaics6: {subdf4qcew.shape[0]} (i.e. negative difference, countyXnaics4 source IS qcew)')
    # print(f'When missing==0:\n {pd.crosstab(negdiff[num_missing==0],df4.loc[num_missing==0,vname+"_source"])}')
    # CASE 2:
    # if there are NO countyXnaics6 cells, and the countyXnaics4 source is not "qcew",
    # then override vname=vname_sum6by4, vname_source='sum6by4', and diff=0
    df4.loc[(negdiff) & (num_missing == 0) & (source_notqcew), vname + "diff"] = 0
    df4.loc[(negdiff) & (num_missing == 0) & (source_notqcew), vname] = df4.loc[
        (negdiff) & (num_missing == 0) & (source_notqcew), vname + "_sum6by4"]
    df4.loc[(negdiff) & (num_missing == 0) & (source_notqcew), vname + "_source"] = "sum6by4"
    ### CASE 3a:
    ## if there vname_source is qcew at countyXnaics4 level and NO missing countyXnaics6,
    ## then must adjust county by NAICS-6 non-qcew values
    # subdf4,df6,df=adjust_nonqcew_negdiff_nomissing(subdf4=subdf4qcew[subdf4qcew[vname+'_missing6by4']==0,:],df6=df6,df=df,vname=vname)
    ## CASE 3b:
    # if there vname_source is qcew at countyXnaics4 level and >0 missing countyXnaics6,
    # then must adjust county by NAICS-6 non-qcew values

    subdf4, df6, df = adjust_nonqcew_negdiff_yesmissing(
        subdf4=subdf4qcew.loc[subdf4qcew[vname + '_missing6by4'] > 0, :],
        df6=df6, df=df, vname=vname)

    ## When diff is POSITIVE, and only 1 countyXnaics6 is missing
    posdiff = (df4[vname + "diff"] > 0)
    # set the missing countyXnaics6 to the diff
    geo4naics_filldig6 = df4.loc[(posdiff) & (num_missing == 1), ['geo4naics', vname + 'diff']]
    geo4naics_filldig6['fill_source'] = vname + "diff"
    df6 = df6.merge(geo4naics_filldig6, on="geo4naics", how="left")
    df6[vname] = df6[vname].fillna(df6[vname + "diff"])
    df6[vname + "_source"] = df6[vname + "_source"].fillna(df6['fill_source'])
    df6 = df6.drop(columns=[vname + "diff", 'fill_source'])
    # AND update the df4 values to correspond
    df4.loc[(posdiff) & (num_missing == 1), vname + 'diff'] = 0
    df4.loc[(posdiff) & (num_missing == 1), vname + '_missing6by4'] = 0
    df4.loc[(posdiff) & (num_missing == 1), vname + '_sum6by4'] = df4.loc[(posdiff) & (num_missing == 1), vname]
    return df4, df6


def adjust_nonqcew_negdiff_nomissing(subdf4, df6, df, vname="wages"):
    print(f'inside adjust_nonqcew_negdiff {vname}')
    subdf6 = df6.loc[df6['geo4naics'].isin(subdf4['geo4naics']), :]
    print(subdf6.columns)
    print(subdf6.head())
    print(subdf6[vname + "_cbp_flag"].value_counts())

    stophere = True
    if stophere:
        raise Exception("stop here")
    return df4, df6, df


def adjust_nonqcew_negdiff_yesmissing(subdf4, df6, df, vname="wages"):
    print(f'inside adjust_nonqcew_negdiff_yesmissing {vname}')
    df['geo4naics'] = df['geoindkey'].str.slice(stop=-2)
    subdf4['prop_wages_from_qcew'] = subdf4['count_wages_from_qcew'] / subdf4['count6by4codes']
    print(subdf4.columns)
    # subdf4['wages_avgperest_qcew']=subdf4['sum_wages_qcew']/subdf4['count6by4codes']
    print(subdf4[['prop_wages_from_qcew', 'wages_missing6by4', 'wagesdiff']].describe())
    subdf4.drop(columns=['count_wages_from_cbp', 'wages_propmissing6by4', 'grouplevels'], inplace=True, errors="ignore")
    subdf6 = df6.loc[df6['geo4naics'].isin(subdf4['geo4naics']), :]
    subdf6["vname_perest"] = subdf6[vname] / subdf6['estnum']
    subdf6['vname_perest_qcew'] = subdf6['vname_perest']
    subdf6.loc[subdf6[vname + "_source"] != "qcew", 'vname_perest_qcew'] = np.nan
    subdf6['vname_peremp3'] = subdf6[vname] / subdf6['emp3']
    subdf6['vname_peremp3_qcew'] = subdf6['vname_peremp3']
    subdf6.loc[subdf6[vname + "_source"] != "qcew", 'vname_peremp3_qcew'] = np.nan
    subdf6gr = subdf6.groupby("geo4naics").agg(vname_avgperest=('vname_perest', "mean"),
                                               vname_medperest=('vname_perest', "median"),
                                               vname_avgperest_qcew=('vname_perest_qcew', "mean"),
                                               vname_medperest_qcew=('vname_perest_qcew', "median"),
                                               vname_avgperemp3=('vname_peremp3', "mean"),
                                               vname_medperemp3=('vname_peremp3', "median"),
                                               vname_avgperemp3_qcew=('vname_peremp3_qcew', "mean"),
                                               vname_medperemp3_qcew=('vname_peremp3_qcew', "median")
                                               )
    print(subdf6gr.head())
    subdf6gr.rename(columns={'vname_avgperest': vname + "_avgperest",
                             'vname_medperest': vname + "_medperest",
                             'vname_avgperest_qcew': vname + "_avgperest_qcew",
                             'vname_medperest_qcew': vname + "_medperest_qcew",
                             'vname_avgperemp3': vname + "_avgperemp3",
                             'vname_medperemp3': vname + "_medperemp3",
                             'vname_avgperemp3_qcew': vname + "_avgperemp3_qcew",
                             'vname_medperemp3_qcew': vname + "_medperemp3_qcew"}, errors="ignore", inplace=True)
    subdf4 = subdf4.merge(subdf6gr, on=['geo4naics'], how='left', indicator=False)
    print(subdf4[subdf4['count_wages_from_qcew'] > 2].head())
    # with Pool(processes=3) as pool:
    #    args = [(x, df6_toget, df4n) for x in test4dig]
    #    results = pool.starmap(process_chunk, args)
    # Combine results
    # df6_toget_imputed = pd.concat(results, ignore_index=True)
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

    #stophere = True
    #if stophere:
    #    raise Exception("stop here")
    return df4, df6, df




def avgestnum_source_adjustment(dfin, check_consistency=True, naicsdf=None):
    """
        Adjust employment and wages based on establishment counts across aggregation levels.

        This function systematically adjusts employment and wage data by using establishment
        counts (estnum) as a scaling factor across different NAICS aggregation levels
        (6-digit, 5-digit, 4-digit, 3-digit, 2-digit).

        The workflow:
        1. Filter data to rows with valid establishment counts
        2. Adjust 6-digit NAICS codes using 5-digit aggregates
        3. Adjust 5-digit NAICS codes using 4-digit aggregates
        4. Continue down to 2-digit NAICS codes
        5. Recalculate employment and wage values using adjusted estnum
        6. Check consistency at each level if requested

        Args:
            dfin (pd.DataFrame):
                Input dataframe with employment data, establishment counts, and NAICS codes.

            check_consistency (bool):
                If True, verify consistency between aggregation levels after adjustment.
                Prints diagnostic information about inconsistencies.
                Default is True.


            naicsdf (pd.DataFrame, optional):
                Lookup table for NAICS code hierarchies.
                Default is None.

        Returns:
            pd.DataFrame:
                Dataframe with adjusted establishment counts and recalculated employment/wage values.
                Rows with missing establishment counts are removed.

        Side Effects:
            - Creates backup columns: 'wages_old', 'emp1_old', 'emp2_old', 'emp3_old', 'estnum_old'
            - Prints consistency check results if check_consistency=True
            - Removes rows with missing establishment numbers
        """
    df = dfin.copy()

    #keep only rows with establishment numbers
    df = df.loc[df['estnum'].notna(), :]

    # Apply hierarchical adjustments from 6-digit down to 2-digit NAICS
    df=adjust_estnum(df,groupbydigits=5,levelgrouped=6,naicsdf=naicsdf)
    df=adjust_estnum(df,groupbydigits=4,levelgrouped=5,naicsdf=naicsdf)
    df=adjust_estnum(df,groupbydigits=3,levelgrouped=4,naicsdf=naicsdf)
    df=adjust_estnum(df,groupbydigits=2,levelgrouped=3,naicsdf=naicsdf)

    # Recalculate employment and wage totals using adjusted estnum
    for vname in ["wages", "emp1", "emp2", "emp3"]:
        df[vname] = df[vname + "_perestnum"] * df['estnum']
        df[vname] = df[vname].round(0)

    return df


def adjust_estnum(dfin2,
                  groupbydigits=4, levelgrouped=6,
                  naicsdf=None,
                  printmore=False):
    """
        Adjust employment data based on establishment count aggregates across NAICS levels.

        This function adjusts employment and wage values for a specific geographic-industry
        grouping by ensuring they are consistent with aggregate counts at a higher NAICS level.

        Two adjustment modes:
        1. justestnum=True: Only adjust establishment counts based on aggregates
        2. justestnum=False: Adjust all employment/wage metrics using per-establishment ratios

        Args:
            dfin2 (pd.DataFrame):
                Input dataframe with employment data and geographic-industry codes.

            groupbydigits (int):
                Number of NAICS digits to group by (2-5).
                2=2-digit, 3=3-digit, 4=4-digit, 5=5-digit NAICS level.
                Default is 4.

            levelgrouped (int):
                Number of NAICS digits in the detailed level to aggregate (must be > groupbydigits).
                Default is 6.

            naicsdf (pd.DataFrame, optional):
                NAICS code lookup table for handling special cases like 2-digit with dashes.
                Default is None.

            printmore (bool):
                If True, print additional diagnostic messages during adjustment.
                Default is False.

        Returns:
            pd.DataFrame:
                Dataframe with adjusted values at the specified aggregation level.
                Rows with missing establishment counts are removed.

        Raises:
            Exception: If groupbydigits is not in range 2-5.
        """

    df = dfin2.copy()

    #validate input parameters
    if groupbydigits < 2 or groupbydigits > 5:
        raise Exception(
            f'Code does not currently support adjustments to the county level values. groupbydigits should be 2,3,4, or 5.')
    if levelgrouped<=groupbydigits or levelgrouped not in [3,4,5,6]:
        raise Exception(f'levelgrouped {levelgrouped} must be greater than groupbydigits {groupbydigits} and in values 3,4,5,6')

    # Calculate aggregation level code and create grouping column
    grby_agglvl = 72 + groupbydigits
    str_end_idx = -(6 - groupbydigits)
    df['geo' + str(groupbydigits) + 'naics'] = df['geoindkey'].str.slice(stop=str_end_idx)

    # Special handling for 2-digit NAICS with dashes
    if groupbydigits == 2:
        df = geo2naics_dash_handler(df, naicsdf)
    countlvldig = get_codes_summary(df, groupbydigits=groupbydigits, levelgrouped=levelgrouped, variable="estnum",
                                    onlyQCEW=False, include_source=False, naicsdf=naicsdf,include_estab_emp3_stats=False)

    # Extract rows at target aggregation level and merge with aggregates
    grby_indic = (df['agglvl_code'] == grby_agglvl)
    dfgrby = df[grby_indic]
    dfestnum = dfgrby.merge(countlvldig, on=['geo' + str(groupbydigits) + 'naics'], how='left', indicator=False,
                            suffixes=["", "_droplater"])
    dfestnum.drop(columns=[dropcol for dropcol in dfestnum.columns if "_droplater" in dropcol], errors="ignore",
                  inplace=True)
    labelstr = str(levelgrouped) + "by" + str(groupbydigits)
    excluded_cbp_stem = [str(stem).replace("-", "").replace("/", "") for stem in EXCLUDED_CBP]

    dfestnum['estnum_new'] = dfestnum['estnum_sum' + labelstr]
    df = df.merge(dfestnum[['geoindkey', 'estnum_new']], on="geoindkey", how="left", indicator=False)

    # Check for missing values at this aggregation level
    numna = df.loc[
            (df['agglvl_code'] == grby_agglvl) &
            (df['estnum_new'].isna()) &
            (~df["geoindkey"].str.startswith(tuple(excluded_cbp_stem))),
            :].shape[0]
    if numna > 0 and printmore:
        print(
            f"for {labelstr} estnum: there are {numna} at agglevel {grby_agglvl} which are NA based on sum{labelstr}.")
    df.loc[(df['agglvl_code'] == grby_agglvl) & (df['estnum_new'].notna()), 'estnum'] = df.loc[
        (df['agglvl_code'] == grby_agglvl) & (df['estnum_new'].notna()), 'estnum_new']
    df.drop(columns='estnum_new', inplace=True, errors="ignore")

    df = df[df["estnum"].notna()]
    return df


