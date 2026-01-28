import pandas as pd
import numpy as np
import scipy.stats
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from formulaic import Formula
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os


sys.path.append(os.path.abspath("./NAICS6_Pyfunctions/"))
from plottingFunctions import *

## Function is incomplete. It formats the fitted model as a latex style string to be used as suptitle of
## matplotlib plots with usetex=True. Currently it does not handle transformation formatting or polynomial term formatting.
def tex_format_model(formula_str,modelfit,digits=2,useparams=False):
    response = modelfit.model.endog_names.strip()
    terms = []
    if useparams:
        predictors= modelfit.model.exog_names
        params=modelfit.params
        print(f'predictors {predictors} params {params}')

        for var,beta in zip(predictors,params):
            beta_round=f"{beta:.{digits}f}"
            if var.lower() in ["intercept","const"]:
                terms.append(beta_round)
            else:
                format_var=re.sub(r'_',r'\\_',var)
                terms.append(f"{beta_round}\\,{format_var}")
    else:
        temp_rhs=formula_str.split("~")[1]
        if "-1" in temp_rhs:
            const=""
            temp_rhs=re.sub("-1","",temp_rhs)
        else:
            const=rf"$\beta_0$"
        terms.append(const)
        predictors=temp_rhs.split("+")
        i=1
        for var in predictors:
            format_var=re.sub(r'_',r'\\_',var.strip())
            terms.append(rf"$\beta_{i}$ {format_var}")
        print(f'predictors {predictors}')

    rhs=" + ".join(terms)
    texout=rf"$\hat{{{response}}} =$ {rhs}"
    print(texout)
    return texout


def features_from_formula_str(formula_str,data):

    if "~" in formula_str:
        lhs, rhs=formula_str.split("~",1)
        response=feature_to_colname(lhs.strip(),data)
    elif "=" in formula_str:
        lhs, rhs = formula_str.split("=", 1)
        response=feature_to_colname(lhs.strip(),data)
    else:
        rhs=formula_str
        response=None
    predictors=[]
    splitplus=rhs.split("+")
    for feature in splitplus:
        cname=feature_to_colname(feature,data)
        predictors.append(cname)
    return list(set(predictors)), response


def transform_feature(feature, df):
    if feature in df.columns:
        return df[feature]
    elif '[T.' in feature:
        base_var = feature.split('(',1)[1].split(')')[0].strip()
        category = feature.split('[T.')[1].split(']')[0].strip()
        return (df.loc[:,base_var] == category).astype(float)
    elif 'C(' in feature:
        base_var = feature.split('(',1)[1].split(')')[0].strip()
        category = feature.split('[')[1].split(']')[0].strip()
        return (df.loc[:,base_var] == category).astype(float)
    elif feature.split("[",1)[0] in df.columns:
        base_var=feature.split("[",1)[0]
        category = feature.split('[')[1].split(']')[0].strip()
        return (df.loc[:, base_var] == category).astype(float)
    elif feature.split("(",1)[1].split("[",1)[0] in df.columns:
        base_var=feature.split("(",1)[1].split("[",1)[0]
        category = feature.split('[')[1].split(']')[0].strip()
        return (df.loc[:,base_var] == category).astype(float)
    elif 'np.sqrt' in feature:
        internalvar = feature.split('(')[1].split(')')[0].strip()
        return np.sqrt(df.loc[:,internalvar].astype(float))
    elif 'np.log' in feature:
        internalvar = feature.split('(')[1].split(')')[0].strip()
        return np.log(df.loc[:,internalvar].astype(float))
    else:
        raise Exception(f'Something wrong with feature. {feature}.\nIt should be: in df columns; a categorical, log, or sqrt transformation; or a polynomial term.')

def feature_to_colname(feature, df):
    if feature in df.columns:
        return feature
    elif feature.replace(" ","") in [cname.replace(" ","") for cname in df.columns.tolist()]:
        npspacecnames=[cname.replace(" ","") for cname in df.columns.tolist()]
        colname=[cname for cname,nospacename in zip(df.columns.tolist(),npspacecnames) if nospacename==feature.replace(" ","")]
        return colname[0]
    elif 'poly' in feature and "(" in feature:
        internal_var=feature.split("(",1)[1].split(',')[0]
        return feature_to_colname(internal_var,df)
    elif '[T.' in feature or feature.strip().startswith("C("):
        return feature_to_colname(feature.split('(',1)[1].split(')')[0].strip(),df)
    elif '[' in feature:
        return feature_to_colname(feature.split('[', 1)[0].strip(), df)
    elif 'np.sqrt' in feature:
        return feature_to_colname(feature.split('(')[1].split(')')[0].strip(),df)
    elif 'np.log' in feature:
        return feature_to_colname(feature.split('(')[1].split(')')[0].strip(),df)
    else:
        raise Exception(f'Something wrong with feature. {feature}.\nIt should be: in df columns; a categorical, log, or sqrt transformation; or a polynomial term.')


def polynomial_handling(feature,X,df):
    inside_poly = feature.split('(', 1)[1].split(',')
    base_var = inside_poly[0].strip()
    degree_sect=inside_poly[1].split('[')[0]
    degree_pre = [char for char in degree_sect if char.isdigit()]
    degree = "".join(degree_pre)

    nospacecolnames = [var.replace(" ", "") for var in X.columns]
    # Only process if we haven't already created these columns
    if f'poly({base_var.replace(" ", "")},degree={degree})[1]' not in nospacecolnames:
        if "np.sqrt(" in base_var:
            internal_var = base_var.split("(")[1].split(")")[0].strip()
            transformsqrt=np.sqrt(df.loc[X.index,internal_var].astype(float))
            #nan_mask = np.isnan(transformsqrt)
            #index_drop = [idx for idx, nabool in zip(transformsqrt.index.tolist(), transformsqrt) if np.isnan(nabool)]
            #X.drop(index_drop, inplace=True)
            #transformsqrt.drop(index_drop, inplace=True)
            vals = transformsqrt.values.reshape(-1, 1)
        elif "np.log(" in base_var:
            internal_var = base_var.split("(")[1].split(")")[0].strip()
            transformlog = np.log(df.loc[X.index,internal_var].astype(float))
            #nan_mask = np.isnan(transformlog)
            #index_drop = [idx for idx, nabool in zip(transformlog.index.tolist(), transformlog) if np.isnan(nabool)]
            #X.drop(index_drop, inplace=True)
            #transformlog.drop(index_drop, inplace=True)
            vals = transformlog.values.reshape(-1, 1)
        else:
            vals_prep = df.loc[X.index,base_var].astype(float)
            #nan_mask=np.isnan(vals_prep)
            #index_drop=[idx for idx,nabool in zip(vals_prep.index.tolist(),vals_prep) if np.isnan(nabool)]
            #X.drop(index_drop,inplace=True)
            #vals_prep.drop(index_drop, inplace=True)
            vals=vals_prep.values.reshape(-1, 1)
        # Create polynomial features

        poly = PolynomialFeatures(degree=int(degree), include_bias=False)
        raw_poly = poly.fit_transform(vals)
        # Center and orthogonalize using QR decomposition
        centered = raw_poly - raw_poly.mean(axis=0)
        Q, R = np.linalg.qr(centered)
        # Ensure consistent sign
        signs = np.sign(Q[0, :])
        Q = Q * signs
        # Store all polynomial terms
        for i in range(int(degree)):
            if "degree=" in feature.replace(" ",""):
                X[f'poly({base_var.replace(" ", "")},degree={degree})[{i + 1}]'] = Q[:, i]
            else:
                X[f'poly({base_var.replace(" ", "")},{degree})[{i + 1}]'] = Q[:, i]

    return X

def custom_predict(data, ols_model,rseed=None):
    '''
    What is the point?
        custom_predict() provides predictions and standard errors from a statsmodels OLS model,
        handling complex formula specifications that include:
        - Interaction terms
        - Polynomial terms
        - Categorical variables
        - Regular numeric predictors
    
    Why is this needed?
        The standard statsmodels predict() method requires the exact design matrix used in fitting.
        This function reconstructs that matrix from the original formula specification using new data.
        Also returns standard errors and handles QR decomposition.
    
    Inputs:
        1. df - pd.DataFrame containing predictor variables
        2. ols_model - Fitted statsmodels OLS model object
    
    Steps:
        1. Extract model feature names from the fitted model
        2. Initialize output DataFrame
        3. Process each feature type:
           a) Interaction terms (handles 'var1:var2' syntax)
           b) Polynomial terms (handles 'poly(var, degree=3)' syntax)
           c) Categorical variables (handles 'C(var)[T.level]' syntax)
           d) Regular numeric predictors
        4. Add intercept column
        5. Ensure column order matches original model
        6. Generate predictions
        7. Calculate standard errors in chunks to manage memory
    
    Special Handling:
        - Polynomial terms are orthogonalized using QR decomposition
        - Categorical variables are one-hot encoded
        - Numeric columns are explicitly converted to float
    
    Returns:
        A tuple containing:
        1. pred - Array of predicted values
        2. se_fit - Array of standard errors for each prediction
    '''
    if rseed is not None:
        np.random.seed(rseed)
    # Get feature names from the fitted model
    model_features = ols_model.model.exog_names
    df=data.copy()
    pred_idx=df.index.tolist()
    ## get relevant index
    for feature in model_features:
        if feature == 'Intercept':
            # Add intercept column
            #X['Intercept'] = 1
            continue
        elif "*" in feature in feature:
            featsplit=feature.split("*",1)
            featurecolname1=feature_to_colname(featsplit[0],df)
            featurecolname2=feature_to_colname(featsplit[1],df)
            missing_mask1=df[featurecolname1].isna()
            missing_mask2 = df[featurecolname2].isna()
            temp_pred_idx = set(pred_idx) - set(df.loc[missing_mask1, :].index.tolist())
            pred_idx = list(temp_pred_idx - set(df.loc[missing_mask2].index.tolist()))
        elif ":" in feature in feature:
            featsplit=feature.split(":",1)
            featurecolname1=feature_to_colname(featsplit[0],df)
            featurecolname2=feature_to_colname(featsplit[1],df)
            missing_mask1=df[featurecolname1].isna()
            missing_mask2 = df[featurecolname2].isna()
            temp_pred_idx = set(pred_idx) - set(df.loc[missing_mask1, :].index.tolist())
            pred_idx=list(temp_pred_idx-set(df.loc[missing_mask2].index.tolist()))
        else:
            featurecolname=feature_to_colname(feature,df)
            missing_mask=df[featurecolname].isna()
            pred_idx=list(set(pred_idx)-set(df.loc[missing_mask,:].index.tolist()))
    nopred_idx=set(df.index.tolist())-set(pred_idx)
    X = pd.DataFrame(index=pred_idx)
    df=df.loc[pred_idx,:]
    # Process each feature in the model 
    for feature in model_features:
        # Skip intercept - we'll add it later
        if feature == 'Intercept':
            X['Intercept'] = 1
        elif "poly(" in feature or "poly (" in feature: #polynomial terms
            X = polynomial_handling(feature, X, df)
        # Handle interaction terms
        elif ':' in feature or "*" in feature:
            astcount=feature.count("*")
            coloncount=feature.count(":")
            if astcount+coloncount>1:
                raise Exception(f"Only two-way interations supported. {feature} has {astcount+coloncount}.")
            elif astcount>1:
                var1, var2 = feature.split("*")
            else:
                var1, var2 = feature.split(':')
            var1col=transform_feature(var1,df)
            var2col=transform_feature(var2,df)
            X[feature]=var1col.astype(float)*var2col.astype(float)
            #featurena=np.isnan(X[feature])
            #index_to_remove=[idx for idx,nabool in zip(X.index.tolist(),featurena) if not nabool]
            #X.drop(index_to_remove,inplace=True)
        else:
            X[feature]=transform_feature(feature,df)
            #featurena = np.isnan(X[feature])
            #index_to_remove = [idx for idx, nabool in zip(X.index.tolist(), featurena) if not nabool]
            #X.drop(index_to_remove, inplace=True)


    # Ensure columns match original model order
    nospace_features=[feat.replace(" ","") for feat in model_features]
    X.columns=[cname.replace(" ","") for cname in X.columns.tolist()]
    X = X[nospace_features]

    # Generate predictions
    pred = ols_model.predict(X)


    # Calculate standard errors of prediction
    cov_matrix = ols_model.cov_params().values
    X_np = X.values
    chunk_size = 1000 ## Process in chunks to keep memory from blowing up
    n = len(X_np)
    se_fit = np.empty(n)
    for i in range(0, n, chunk_size):
        chunk = X_np[i:i+chunk_size]
        # Calculate (X @ cov_matrix) @ X.T for each row
        x_cov = chunk @ cov_matrix
        se_fit[i:i+chunk_size] = np.sqrt(np.sum(x_cov * chunk, axis=1))
    fullpred=pd.DataFrame(index=data.index.tolist())
    fullpred['pred']=np.nan
    fullpred['se']=np.nan
    fullpred.loc[pred_idx,'pred']=pred
    fullpred.loc[pred_idx,'se']=se_fit
    return fullpred['pred'], fullpred['se']

def possible_variables(data,response):
    naidx=data[response].isna()
    missing_y=data[naidx]
    possible_vars=list()
    for cvar in data.columns:
        if cvar==response:
            pass
        else:
            numna=missing_y[cvar].isna().sum()
            if numna==0:
                if data[cvar].nunique()>1:
                    possible_vars.append(cvar)
    return possible_vars




def subset_model_data(data,formula_str):
    # Retrieve variables from OLS formula
    checkvars = Formula(formula_str).required_variables
    for cvar in checkvars:
        #if it can be easily made from geoindkey, make it
        if cvar not in data.columns and cvar in ["sector",'naics2',"naics3","naics4"]:
            indkey=data['geoindkey'].astype(str).str.replace('[0-9]*_',"",regex=True)
            if cvar=='sector':
                data['sector']=indkey.astype(str).str[:2]
            elif cvar=='naics2':
                data['naics2'] = indkey.astype(str).str[:2]
            elif cvar=='naics3':
                data['naics3'] = indkey.astype(str).str[:3]
            else:
                data['naics4'] = indkey.astype(str).str[:4]
        assert cvar in data.columns, f"{cvar} needed for ols formula but is not in the data columns."
        if cvar+"_flag" in data.columns:
            if data[cvar+"_flag"].dtype=="object":
                data=data[data["s"+cvar]==1].copy() #not suppressed in QWI
            else:
                data = data[data[cvar + "_nf"].isin(["G", "H", "J"])].copy()
        if "C("+cvar in formula_str.replace(" ",""): #if it is categorical
            continue
        else: #otherwise see if it should be made into a float
            pass
            #if data[cvar].astype(str).str.isnumeric().any():
            #    data[cvar]=data[cvar].astype(float)
    #print(list(checkvars))
    #print(data.columns)
    #print(data.shape[0])
    data=data.loc[:,list(checkvars)].copy()
    #print(data.head())
    outdf=data[data.notna().all(axis=1)]
    print(outdf.shape[0])
    #print(outdf.head())
    return data[data.notna().all(axis=1)]


def get_model(data,formula_str,cooks_thresh,studentresid_thresh,include_multicolinearity=False,diagnostic_plots=None,output_removed=False,return_summary_and_diagnostics=False):
    '''
    What is the point?
        get_model() creates an OLS model that predicts values based on various
        predictors specified in formula_str. It filters out outliers and can save diagnostic plots based on Config values
    Steps:
        1. Filters input data to include only rows where variables are not suppressed or missing
        2. Initial model fitting
            - Use formula specified in config.yaml
              to construct the design matrix to fit an OLS model. (model_pre).
        3. Influential point/Outlier detection
            - Compute Cook's distance for each observation and filter out observations where Cook's
              Distance exceeds the threshold set in config.yaml. (default: 1)
              Compute Studentized Residuals for each observation and filter out observations where they
              exceed the threshold set in config.yaml
        4. Refit model after removing influential points
    Configurable Parameters:
        The regression formula and Cook's disitance thresholds are both configurable via config.yaml
    Returns:
        1. model  -  (statsmodel.OLS)
            - Used with custom_predict in get_m1emp() to predict month 1 employment counts
        2. Prints a message if any influential points are removed.
            - Helpful Diagnostic
        3. Can save the dataset the model is fit on as a csv for more exploration of prediction models and
            save the diagnostic plots from the final model
    '''


    # Step 1
    # Retrieve OLS formula from config.yaml
    formula=formula_str
    Config={'COOKS_THRESH':cooks_thresh,'OUTLIER_THRESH':studentresid_thresh,'DIAGNOSTIC_PLOTS':diagnostic_plots}

    subdf = subset_model_data(data,formula_str)

    # Create design matrices (gets the variables ready for fitting in statsmodels.OLS) using the formula
    # and perform initial model fitting

    y_pre, X_pre = Formula(formula).get_model_matrix(subdf)
    # corrmat=X_pre.corr()
    # for cname in corrmat.columns:
    #     corrvals=np.abs(corrmat[cname])
    #     highcorr=corrmat.loc[corrvals>=0.75,]
    #     if highcorr.shape[0]>0:
    #         #toprint=highcorr.apply(lambda  row:f"{row.name} ({row[cname]})")
    #         print(f'{cname}: high correlation with {[rname for rname in highcorr.index.values if rname!="Intercept" and rname!=cname]} ({highcorr[cname].values})')
    # print(f'X_pre head is \n{X_pre.head()}\n and describe is\n{X_pre.describe()}\n and Y_pre head is\n{y_pre.head()}')
    model_pre = sm.OLS(y_pre, X_pre).fit()
    #print(f'model before outlier handling \n {model_pre.summary()}')

    # Calculate Cook's distance for each observation
    influence = OLSInfluence(model_pre)
    cooks_d=influence.cooks_distance[0]
    student_resid = influence.resid_studentized_internal
    # Identify and remove indices of influential points. (Cook's Distance > threshold)
    # Threshold is configurable in config.yaml -> employmentConfig -> 'COOKS_THRESH'
    influential_indices = [i for i, d in enumerate(cooks_d) if d > Config['COOKS_THRESH']]
    outliers = [i for i, r in enumerate(student_resid) if np.abs(r) > Config['OUTLIER_THRESH']]
    if influential_indices:
        if output_removed:
            if len(influential_indices)<=6:
                outliertext=["Filtered out the following indices due to influence (Cook's Distance):",
                            ", ".join([str(x) for x in influential_indices])]
            else:
                outliertext = [f"Filtered out {len(influential_indices)} outliers due to influence (Cook's Distance)."]
        else:
            print("Filtered out the following indices due to influence (Cook's Distance):", influential_indices)
    elif output_removed:
        outliertext=["Filtered out 0 outliers due to influence (Cook's Distance)."]
    if outliers:
        if output_removed:
            outliertext.append("# of outliers filtered (Studentized Residuals):"+str(len(outliers))+'|'+str(np.round((len(outliers)/len(subdf)),3) * 100)+'%')
        else:
            print("# of outliers filtered (Studentized Residuals):", len(outliers), '|', np.round((len(outliers)/len(subdf)),3) * 100, '%')
    rows_to_drop = influential_indices + outliers

    del X_pre, y_pre, model_pre
    # Rebuild design matrices without influential points and perform final model fitting.
    y, X = Formula(formula).get_model_matrix(subdf.drop(subdf.index[rows_to_drop]))
    if include_multicolinearity:
        if return_summary_and_diagnostics:
            vifdf, multtext = find_multicollinearity(X, return_text=True)
            outliertext=outliertext+multtext
        else:
            vifdf= find_multicollinearity(X, return_text=False)


    model = sm.OLS(y, X).fit()
    if Config['DIAGNOSTIC_PLOTS'] is not None:
        save_diagnostic_plots(model,formula_str,Config['DIAGNOSTIC_PLOTS'])
    # end. return fitted model.
    #print("Done fitting")
    if output_removed:
        if return_summary_and_diagnostics:
            return model, outliertext, model.summary().as_text(), model.fittedvalues, model.resid
        else:
            return model, outliertext
    else:
        if return_summary_and_diagnostics:
            return model, model.summary().as_text(), model.fittedvalues, model.resid
        else:
            return model

######### IDEA NOTE TO SELF: ######
## use predicted values from OLS model as parameters for dirichlet divider...
####### END NOTE TO SELF ###########

def dirichlet_divider(params,total,size=1,rseed=None,param_lb=1e-10):
    if rseed is not None:
        np.random.seed(rseed)
    # Get Dirichlet parameters based on m3emp distribution
    if len(params) > 1:  # Multiple parameters
        print(len(params))
        print(params)
        checknonneg=params<0
        print(checknonneg.sum())
        if checknonneg.sum()>0:
            raise Exception(f"Parameters must be non-negative: {params}")
        checkpos=params>0
        if checkpos.sum()==0:
            params=[1]*len(params)
        else:
            minnonzero=min(params[params>0])
            paramszeros_indic=(params==0).astype(int)
            print(paramszeros_indic)
            if paramszeros_indic.sum()>0:
                if minnonzero<1 and param_lb<1: #make sure lower bound is much lower than minnonzero
                    params[paramszeros_indic]=param_lb*minnonzero# if x==0 else x for x in params] #change zeros to small values
                elif param_lb<minnonzero:
                    params[paramszeros_indic] = param_lb# if x == 0 else x for x in params]
                elif param_lb<1:
                    params[paramszeros_indic]=param_lb*minnonzero# if x==0 else x for x in params] #change zeros to small values
                #params=[param_lb*minnonzero if x==0 else x for x in params] #change zeros to small values
                else:
                    raise Exception(f"parameter lower bound 'param_lb'={param_lb} should be <1 or < minimum non-zero 'param' value ({minnonzero}).")
        print(params)
        if isinstance(params,np.ndarray) and len(params)>1:
            rprops=scipy.stats.dirichlet.rvs(params,size=size)
        else:
            rprops = np.random.dirichlet(params, size=size)
        # Split the m1emp value proportionally using rprops
        return np.round(rprops.flatten() * total).astype(float)
    else:
        return total



def find_multicollinearity(df, target=None, vif_threshold=5.0, return_text=False,ignore_intercept=True):
    """
    Identify variables with multicollinearity using Variance Inflation Factor (VIF).

    Parameters:
        df (pd.DataFrame): DataFrame containing features (and optionally the target).
        target (str): Optional target column name to exclude.
        vif_threshold (float): VIF threshold above which variables are considered multicollinear.

    Returns:
        pd.DataFrame: Table of VIF values sorted descending.
    """
    # Drop target if provided
    if target and target in df.columns:
        X = df.drop(columns=[target])
    else:
        X = df.copy()

    # Add constant term for intercept
    #X = sm.add_constant(X)

    # Calculate VIF for each variable
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    if ignore_intercept:
        vif_data=vif_data.loc[vif_data["Variable"]!="Intercept",:].copy()

    # Remove the constant from results
    vif_data = vif_data[vif_data["Variable"] != "const"].reset_index(drop=True)

    # Sort and flag variables with high VIF
    vif_data = vif_data.sort_values("VIF", ascending=False)
    high_vif = vif_data[vif_data["VIF"] > vif_threshold]


    if return_text:
        if high_vif.empty:
            outtext=f"\nModel Predictors with VIF > {vif_threshold} (potential multicollinearity): None detected"
        else:
            categories_indic=high_vif["Variable"].str.contains("[T.")
            if categories_indic.sum()==0:
                outtext=f"\nModel Predictors with VIF > {vif_threshold} (potential multicollinearity):\n{high_vif}"
            else:
                outtext=f"\nModel Predictors with VIF > {vif_threshold} (potential multicollinearity):\n{high_vif}"

                #new_high_vif=high_vif.loc[~categories_indic,:]
                #categories=high_vif.loc[categories_indic,"Variable"].str.split("[",expand=True)[0]



        return vif_data,outtext
    else:
        print(f"\nModel Predictors with VIF > {vif_threshold} (potential multicollinearity):")
        if high_vif.empty:
            print("None detected.")
        else:
            print(high_vif)
        return vif_data


def write_pipe_table(df, filename,include_index=True):
    """
    Writes a pandas DataFrame to a file in pipe-formatted table
    df : The data to write.
    filename : Path to the output file.
    """
    if include_index:
        df.reset_index(drop=False,inplace=True)
    # Create pipe-style header
    header = "| " + " | ".join([str(col) for col in df.columns]) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    #get max character count

    # Create pipe-style rows
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " \t| ".join([rv if isinstance(rv,str) else f"{rv:.0f}" for rv in row.values]) + " \t|")

    table_text = "\n".join([header, separator] + rows) + "\n"

    # Determine write mode based on file existence
    file_exists = os.path.exists(filename)
    write_mode = "a" if file_exists else "w"

    with open(filename, write_mode, encoding="utf-8") as f:
        if file_exists:
            f.write("\n")  # add blank row before appending
        f.write(table_text)


