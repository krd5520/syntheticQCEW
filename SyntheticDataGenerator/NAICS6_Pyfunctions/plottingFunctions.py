import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from formulaic import Formula
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns
import warnings
import sys
import os
import re


sys.path.append(os.path.abspath("./NAICS6_Pyfunctions/"))
from GeneralFunctions import *
from hierarchy_geoindkey import *
#from adjustmentFunctions import *


def save_diagnostic_plots(model,title,filename):
    #title=tex_format_model(title,model)
    fitted=model.fittedvalues
    resid=model.resid
    response=model.model.endog_names
    ycol=model.model.endog

    fig, axes=plt.subplots(1,3,figsize=(18,6))
    ## QQ plot for normality
    qqplot(resid,line='s',ax=axes[0])
    axes[0].set_title("Residual Q-Q Plot")

    ## Resid vs fitted
    sns.residplot(x=fitted,y=resid,lowess=True,line_kws={'color':'red','lw':1},scatter_kws={'alpha':0.4},ax=axes[1])
    axes[1].axhline(0,color='grey',linewidth=1)
    axes[1].set_xlabel("Fitted")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Fitted vs. Residuals")

    ## Observed vs Fitted
    coef = np.polyfit(fitted, ycol, 1) # Best-fit line
    poly = np.poly1d(coef)
    axes[2].scatter(fitted, ycol, alpha=0.6, label="Data")
    axes[2].plot(fitted, poly(fitted), linewidth=2, label="Best-fit line")
    min_val = min(ycol.min(), fitted.min())     # 45-degree perfect-fit line
    max_val = max(ycol.max(), fitted.max())
    axes[2].plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Ideal fit")

    axes[2].set_xlabel("Fitted")
    axes[2].set_ylabel("Observed Values")
    axes[2].set_title("Observed vs Fitted")
    axes[2].legend()


    plt.suptitle(title,fontsize=12,wrap=True) #,y=1.05,usetex=True)
    plt.tight_layout()
    plt.savefig(filename,dpi=500,bbox_inches='tight')
    plt.close('all')
    print("Save regression diagnostic plots as "+filename)


def plot_hist_with_normal(series, filename="histogram.png", bins=30):
        """
        Plot a histogram of a pandas Series with a normal density curve overlay.

        Parameters:
            series (pd.Series): Input data.
            filename (str): Output filename for the saved PNG.
            bins (int): Number of bins for the histogram.
        """
        # Drop missing values
        series = series.dropna()

        # Compute mean and standard deviation
        mean = series.mean()
        std = series.std()

        # Create histogram
        plt.figure(figsize=(8, 5))
        count, bins_edges, _ = plt.hist(series, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black')

        # Create normal density curve
        x = np.linspace(series.min(), series.max(), 200)
        y = norm.pdf(x, mean, std)
        plt.plot(x, y, 'r-', linewidth=2, label='Normal PDF')

        # Add labels and title
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title(f"Histogram with Normal Curve (mean={mean:.2f}, std={std:.2f})")
        plt.legend()

        # Save the plot
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close('all')
        print(f"Saved histogram with normal curve as '{filename}'")


def plot_regression_fit(df, fitmodel, y_col=None,filename=None):
    """
    Plots observed vs predicted values for a multiple regression model.
    df : pandas DataFrame
        Data containing the features and target.
    model : fitted regression model
        Any model with a .predict() method (sklearn or statsmodels).
    y_col : str
        Name of the target variable in `df`.
    """
    if y_col is None:
        y_col=fitmodel.model.endog_names
    print(y_col)
    y_obs = df[y_col] #observed values

    # Handle statsmodels and sklearn
    try:
        y_pred = fitmodel.predict(df.drop(columns=[y_col]))
    except Exception:
        y_pred = fitmodel.predict(df)

    # Best-fit line
    coef = np.polyfit(y_pred, y_obs, 1)
    poly = np.poly1d(coef)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_pred, y_obs, alpha=0.6, label="Data")
    plt.plot(y_pred, poly(y_pred), linewidth=2, label="Best-fit line")

    # 45-degree perfect-fit line
    min_val = min(y_obs.min(), y_pred.min())
    max_val = max(y_obs.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Ideal fit")

    plt.xlabel("Predicted Values")
    plt.ylabel("Observed Values")
    plt.title("Observed vs Predicted")
    plt.legend()
    plt.grid(True)

    if filename is not None:
        plt.savefig(filename, dpi=300)
        plt.close('all')
        print(f"Saved observed vs fitted as '{filename}'")
    else:
        plt.show()
    return plt


