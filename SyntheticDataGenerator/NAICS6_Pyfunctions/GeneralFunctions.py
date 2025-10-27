import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import OLSInfluence
from formulaic import Formula
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def custom_predict(df, ols_model,rseed=None):
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
    X = pd.DataFrame(index=df.index)
    # Process each feature in the model 
    for feature in model_features:
        # Skip intercept - we'll add it later
        if feature == 'Intercept':
            continue
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
            if '[T.' in var1:
                base_var1 = var1.split('C(')[1].split(')')[0]
                category1 = var1.split('[T.')[1].split(']')[0]
                if '[T.' in var2:
                    base_var2 = var2.split('C(')[1].split(')')[0]
                    category2 = var2.split('[T.')[1].split(']')[0]
                    X[feature] = (df[base_var1] == category1).astype(float) * (df[base_var2] == category2).astype(float)
                else:
                    X[feature] = (df[base_var1] == category1).astype(float) *df[var2].astype(float)
            elif '[T.' in var2:
                base_var2 = var2.split('C(')[1].split(')')[0]
                category2 = var2.split('[T.')[1].split(']')[0]
                X[feature] = (df[base_var2] == category2).astype(float) * df[var1].astype(float)
            elif var1 in df.columns and var2 in df.columns:
                X[feature] = df[var1].astype(float) * df[var2].astype(float)
            else:
                raise Exception(f"Cannot find variables {var1} and {var2} in data to use in interation {feature}.")
        # Handle polynomial terms (e.g., 'poly(var, degree=3)')
        elif 'poly' in feature and "(" in feature:
            inside_poly = feature.split('(')[1].split(',')
            base_var=inside_poly[0].strip()
            degree_pre=[char for char in inside_poly[1] if char.isdigit()]
            degree="".join(degree_pre)
            nospacecolnames=[var.replace(" ","") for var in X.columns]
            # Only process if we haven't already created these columns
            if f'poly({base_var.replace(" ","")},degree={degree})[1]' not in nospacecolnames:
                vals = df[base_var].astype(float).values.reshape(-1, 1)
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
                    X[f'poly({base_var.replace(" ","")},degree={degree})[{i+1}]'] = Q[:, i]
        # Handle categorical variables (e.g., 'C(var)[T.level]')
        elif '[T.' in feature:
            base_var = feature.split('C(')[1].split(')')[0]
            category = feature.split('[T.')[1].split(']')[0]
            X[feature] = (df[base_var] == category).astype(float)
        # Handle regular numeric predictors
        elif feature in df.columns:
            X[feature] = df[feature].astype(float)
    # Add intercept column
    X['Intercept'] = 1
    # Ensure columns match original model order
    X = X[model_features]
    # Generate predictions
    pred = ols_model.predict(X)
    # Calculate standard errors of prediction
    cov_matrix = ols_model.cov_params().values
    X_np = X.values
    chunk_size = 1000 ## Process in chunks to keeep memory from blowing up
    n = len(X_np)
    se_fit = np.empty(n)
    for i in range(0, n, chunk_size):
        chunk = X_np[i:i+chunk_size]
        # Calculate (X @ cov_matrix) @ X.T for each row
        x_cov = chunk @ cov_matrix
        se_fit[i:i+chunk_size] = np.sqrt(np.sum(x_cov * chunk, axis=1))
    return pred, se_fit

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
    # Retrieve OLS formula from config.yaml
    checkvars = Formula(formula_str).required_variables
    for cvar in checkvars:
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
        if "s"+cvar in data.columns:
            data=data[data["s"+cvar]==1] #not suppressed in QWI
        elif cvar+"_nf" in data.columns:
            data=data[data[cvar+"_nf"].isin(["G","H","J"])] #is noise-infused but present in CBP
        if "C("+cvar in formula_str.replace(" ",""):
            iscat=True
        else:
            data[cvar]=data[cvar].astype(float)
    data=data[list(checkvars)]
    return data[data.notna().all(axis=1)]

def save_diagnostic_plots(model,title,filename):
    fitted=model.fittedvalues
    resid=model.resid

    fig, axes=plt.subplots(1,2,figsize=(12,6))
    ## QQ plot for normality
    qqplot(resid,line='s',ax=axes[0])
    axes[0].set_title("Residual Q-Q Plot")

    ## Resid vs fitted
    sns.residplot(x=fitted,y=resid,lowess=True,line_kws={'color':'red','lw':1},scatter_kws={'alpha':0.4},ax=axes[1])
    axes[1].axhline(0,color='grey',linewidth=1)
    axes[1].set_xlabel("Fitted")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Fitted vs. Residuals")

    plt.suptitle(title,fontsize=12,y=1.05)
    plt.tight_layout()
    plt.savefig(filename,dpi=300,bbox_inches='tight')
    plt.close()
    print("Save regression diagnostic plots as "+filename)

def get_model(data,formula_str,cooks_thresh,studentresid_thresh,diagnostic_plots=None,output_removed=False,return_summary_and_diagnostics=False):
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
    #if Config['MODEL_DATA_FILE'] is not None:
    #    combinedf.to_csv(Config['MODEL_DATA_FILE'])



    # Create design matrices (gets the variables ready for fitting in statsmodels.OLS) using the formula
    # and perform initial model fitting
    y_pre, X_pre = Formula(formula).get_model_matrix(subdf)
    model_pre = sm.OLS(y_pre, X_pre).fit()

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
    subdffull = subdf.drop(subdf.index[rows_to_drop])
    # Rebuild design matrices without influential points and perform final model fitting.
    y, X = Formula(formula).get_model_matrix(subdffull)
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
        #print(params)
        checknonneg=params<0
        if checknonneg.sum()>0:
            raise Exception(f"Parameters must be non-negative: {params}")
        checkpos=params>0
        if checkpos.sum()==0:
            params=[1]*len(params)
        else:
            minnonzero=min(params[params>0])
            if minnonzero<1 and param_lb<1: #make sure lower bound is much lower than minnonzero
                params=[param_lb*minnonzero if x==0 else x for x in params] #change zeros to small values
            elif param_lb<minnonzero:
                params = [param_lb if x == 0 else x for x in params]
            elif param_lb<1:
                params=[param_lb*minnonzero if x==0 else x for x in params] #change zeros to small values
            else:
                raise Exception(f"parameter lower bound 'param_lb'={param_lb} should be <1 or < minimum non-zero 'param' value ({minnonzero}).")
        rprops = np.random.dirichlet(params, size=size)
        # Split the m1emp value proportionally using rprops
        return np.round(rprops.flatten() * total).astype(float)
    else:
        return total
