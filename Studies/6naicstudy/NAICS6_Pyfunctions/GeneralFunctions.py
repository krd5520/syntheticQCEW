import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

def custom_predict(df, ols_model):
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
    # Get feature names from the fitted model
    model_features = ols_model.model.exog_names
    X = pd.DataFrame(index=df.index)
    # Process each feature in the model 
    for feature in model_features:
        # Skip intercept - we'll add it later
        if feature == 'Intercept':
            continue
        # Handle interaction terms
        elif ':' in feature:
            var1, var2 = feature.split(':')
            if var1 in df.columns and var2 in df.columns:
                X[feature] = df[var1].astype(float) * df[var2].astype(float)
        # Handle polynomial terms (e.g., 'poly(var, degree=3)')
        elif 'poly(' in feature:
            base_var = feature.split('(')[1].split(',')[0]
            # Only process if we haven't already created these columns
            if f'poly({base_var}, degree=3)[1]' not in X.columns:
                vals = df[base_var].astype(float).values.reshape(-1, 1)
                # Create polynomial features
                poly = PolynomialFeatures(degree=3, include_bias=False)
                raw_poly = poly.fit_transform(vals)
                # Center and orthogonalize using QR decomposition
                centered = raw_poly - raw_poly.mean(axis=0)
                Q, R = np.linalg.qr(centered)
                # Ensure consistent sign
                signs = np.sign(Q[0, :])
                Q = Q * signs
                # Store all polynomial terms
                for i in range(3):
                    X[f'poly({base_var}, degree=3)[{i+1}]'] = Q[:, i]
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

