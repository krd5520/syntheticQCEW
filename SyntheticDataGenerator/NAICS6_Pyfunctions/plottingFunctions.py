import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from formulaic import Formula
import scipy.stats as stats
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


##Function produced with assistance from AI
def save_diagnostic_plots(model,title,filename,cook_threshold=None,stud_threshold=None):
    #title=tex_format_model(title,model)
    fitted=model.fittedvalues
    resid=model.resid
    response=model.model.endog_names
    ycol=model.model.endog
    fitted_arr = np.array(fitted)
    resid_arr = np.array(resid)
    ycol_arr=np.array(ycol)

    n = len(fitted)
    all_indices = np.arange(n)
    influence = OLSInfluence(model)
    n = int(model.nobs)

    cooks_d = influence.cooks_distance[0]
    stud_resid = influence.resid_studentized_internal

    # if cook_threshold is None:
    #    cook_threshold = 4 / n

    obs_index = np.arange(n)

    if cook_threshold is not None:
        cook_outliers = obs_index[cooks_d > cook_threshold]
    else:
        cook_outliers=None
    if stud_threshold is not None:
        stud_outliers = obs_index[np.abs(stud_resid) > stud_threshold]
    else:
        stud_outliers=None
    # Build index sets
    outlier_set = set(stud_outliers) if stud_outliers is not None else set()
    influential_set = set(cook_outliers) if cook_outliers is not None else set()
    special_set = outlier_set | influential_set
    normal_idx = [i for i in all_indices if i not in special_set]
    outlier_only_idx = [i for i in all_indices if i in outlier_set and i not in influential_set]
    influential_only_idx = [i for i in all_indices if i in influential_set and i not in outlier_set]
    both_idx = [i for i in all_indices if i in outlier_set and i in influential_set]

    # Helper to scatter subgroups with distinct styles
    def scatter_groups(ax, x, y, alpha=0.5, size=20):
        ax.scatter(x[normal_idx], y[normal_idx], color='black', alpha=alpha, s=size, label='Normal')
        if outlier_only_idx:
            ax.scatter(x[outlier_only_idx], y[outlier_only_idx], color='orange',
                       marker='^', s=size * 2, alpha=0.9, label='Outlier')
        if influential_only_idx:
            ax.scatter(x[influential_only_idx], y[influential_only_idx], color='steelblue',
                       marker='D', s=size * 2, alpha=0.9, label='Influential')
        if both_idx:
            ax.scatter(x[both_idx], y[both_idx], color='fuchsia',
                       marker='*', s=size * 3, alpha=0.9, label='Outlier & Influential')

    fig, axes=plt.subplots(1,2,figsize=(8,3.8))
    ## QQ plot for normality
    qqplot(resid,line='s',ax=axes[0],markerfacecolor="black",markeredgecolor="black",marker='o',alpha=0.9)
    # if special_set: #if there are outliers or influential points
    #     # Recompute QQ theoretical quantiles to mark special points
    #     theoretical,sortedresid=stats.probplot(resid)
    #     #sorted_idx = np.argsort(resid)
    #     # rank = np.argsort(sorted_idx)  # rank of each original index in sorted order
    #     # n_pts = len(resid)
    #     # theoretical = pd.Series(stats.norm.ppf((rank + 1) / (n_pts + 1)))
    #     # theoretical.index=sorted_idx
    #     # standardized_resid = (resid - resid.mean()) / resid.std()
    #     for idx, color, marker, label in [
    #          (outlier_only_idx, 'orange', '^', 'Outlier'),
    #          (influential_only_idx, 'steelblue', 'D', 'Influential'),
    #          (both_idx, 'fuchsia', '*', 'Outlier & Influential'),
    #     ]:
    #         print(f"in qqplot outliers for {label} the min and max idx is {min(idx)}, {max(idx)}, the theoretical and standardized resid types are {type(theoretical),type(sortedresid)}")
    #     #     try:
    #     #         print(theoretical.index)
    #     #     except:
    #     #         print("thoeretical has no index")
    #     #
    #     #     try:
    #     #         print(standardized_resid.index)
    #     #     except:
    #     #         print("standardized resid has no index")
    #     #
    #         if idx:
    #             theoidx=theoretical[idx]
    #             stresidx=sortedresid[idx]
    #             axes[0].scatter(theoretical[[int(i) for i in idx]], sortedresid[[int(i) for i in idx]],
    #                             color=color, marker=marker, s=60, zorder=5, label=label)
    #     axes[0].legend(fontsize=7)

    axes[0].set_title("Residual Q-Q Plot")
    axes[1].axhline(0,color='grey',linewidth=1)
    ## Resid vs fitted
    if special_set:
        # Plot lowess line manually so we can layer scatter on top
        scatter_groups(axes[1], fitted_arr, resid_arr, alpha=0.4)
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(resid_arr, fitted_arr, frac=2 / 3)
        axes[1].plot(smoothed[:, 0], smoothed[:, 1], color='crimson', lw=1.5,linestyle='--')
        axes[1].legend(fontsize=7)
    else:
        sns.residplot(x=fitted,y=resid,lowess=True,
                      line_kws={'color':'crimson','lw':1.5,'linestyle':'--'},
                      scatter_kws={'alpha':0.4,'color':'black'},ax=axes[1])

    axes[1].set_xlabel("Fitted",fontsize=10)
    axes[1].set_ylabel("Residuals",fontsize=10)
    axes[1].set_title("Fitted vs. Residuals",fontsize=10)

    # ## Observed vs Fitted
    # if special_set:
    #     scatter_groups(axes[2], fitted_arr, ycol_arr, alpha=0.6)
    #     axes[2].legend(fontsize=7)
    # else:
    #     axes[2].scatter(fitted_arr, ycol_arr, alpha=0.6, label='Data')
    #
    # try:
    #     coef = np.polyfit(fitted_arr, ycol_arr, 4)
    #     poly = np.poly1d(coef)
    #     axes[2].plot(fitted_arr, poly(fitted_arr), linewidth=2, label='Best-fit line')
    # except:
    #     print("Could not fit 'best fit line' for observed vs fitted plot.")
    #
    # min_val = min(ycol_arr.min(), fitted_arr.min())
    # max_val = max(ycol_arr.max(), fitted_arr.max())
    # axes[2].plot([min_val, max_val], [min_val, max_val], linestyle='--', label='Ideal fit')
    # axes[2].set_xlabel("Fitted")
    # axes[2].set_ylabel("Observed Values")
    # axes[2].set_title("Observed vs Fitted")
    # axes[2].legend()


    plt.suptitle(title,fontsize=12,wrap=True) #,y=1.05,usetex=True)
    plt.tight_layout()
    plt.savefig(filename,dpi=500,bbox_inches='tight')
    plt.close('all')
    print("Save regression diagnostic plots as "+filename)

## Function produced with the assistance of AI
def plot_outlier_diagnostics(model, title,cook_threshold=None, stud_threshold=None, figsize=(8, 3.8), plotstem=None):
    """
    Plot Cook's Distance and Studentized Residuals with outlier thresholds.

    Parameters
    ----------
    model       : fitted OLS model from statsmodels (sm.OLS(...).fit())
    cook_threshold : float or None
                    Threshold for Cook's distance. If None, uses 4/n as the
                    common rule-of-thumb.
    stud_threshold : float
                    Threshold for absolute studentized residuals (default 3.0).
    figsize     : tuple, figure size.

    Returns
    -------
    fig, axes   : matplotlib Figure and Axes array
    outliers    : dict with keys 'cooks' and 'studentized', each a list of
                  observation indices exceeding the respective threshold.
    """
    influence = OLSInfluence(model)
    n = int(model.nobs)

    cooks_d = influence.cooks_distance[0]
    stud_resid = influence.resid_studentized_internal

    #if cook_threshold is None:
    #    cook_threshold = 4 / n

    obs_index = np.arange(n)
    if cook_threshold is not None:
        cook_outliers = obs_index[cooks_d > cook_threshold]
    else:
        cook_outliers=None
    if stud_threshold is not None:
        stud_outliers = obs_index[np.abs(stud_resid) > stud_threshold]
    else:
        stud_outliers=None

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Cook's Distance ──────────────────────────────────────────────────────
    ax = axes[0]
    markerline, stemlines, baseline = ax.stem(
        obs_index, cooks_d, linefmt="black", markerfmt="o", basefmt=" "
    )
    plt.setp(markerline, markersize=4, color="black",alpha=0.5)
    plt.setp(stemlines, linewidth=0.8,alpha=0.9)
    #for i in range(len(cooks_d)):
    #    if cooks_d.iloc[i]>cook_threshold:
    #        plt.setp(list(stemlines)[i],color='steelblue')
            #plt.setp(markerline[i],markerfacecolor='steelblue',markeredgecolor='steelblue')
    # Overlay outlier markers in steelblue
    if cook_outliers is not None and len(cook_outliers) > 0:
        ax.plot(
            cook_outliers,
            cooks_d.iloc[cook_outliers],
            'o',
            markersize=4,
            color='steelblue',
            alpha=0.9,
            zorder=3  # draw on top of the stem markers
        )

    if cook_threshold is not None:
        ax.axhline(cook_threshold, color="crimson", linestyle="--", linewidth=1.5,
                   label=f"Threshold = {cook_threshold:.4f}  (4/n)")

        # for idx in cook_outliers:
        #     ax.annotate(str(idx), xy=(idx, cooks_d[idx]),
        #                 xytext=(4, 4), textcoords="offset points",
        #                 fontsize=8, color="crimson")

    ax.set_title("Cook's Distance", fontsize=10, fontweight="bold")
    ax.set_xlabel("Observation Index",fontsize=10)
    ax.set_ylabel("Cook's Distance",fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(-1, n)

    # ── Studentized Residuals ────────────────────────────────────────────────
    ax = axes[1]
    if stud_threshold is not None:
        colors = np.where(np.abs(stud_resid) > stud_threshold, "orange", "black")
    else:
        colors=np.where(np.abs(stud_resid)>-1,"black","black")
    ax.scatter(obs_index, stud_resid, c=colors, s=20, zorder=3,alpha=0.8)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="-")
    if stud_threshold is not None:
        ax.axhline( stud_threshold, color="crimson", linestyle="--", linewidth=1.5,
                    label=f"±{stud_threshold} threshold")
        ax.axhline(-stud_threshold, color="crimson", linestyle="--", linewidth=1.5)

        # for idx in stud_outliers:
        #     ax.annotate(str(idx), xy=(idx, stud_resid[idx]),
        #                 xytext=(4, 4), textcoords="offset points",
        #                 fontsize=8, color="crimson")

    ax.set_title("Studentized Residuals", fontsize=10, fontweight="bold")
    ax.set_xlabel("Observation Index",fontsize=10)
    ax.set_ylabel("Studentized Residual",fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(-1, n)



    plt.suptitle(title,fontsize=12,wrap=True) #,y=1.05,usetex=True)
    if plotstem is not None:
        filename=plotstem+"_outlier_threshold_plots.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=500, bbox_inches='tight')

        print("Save regression diagnostic plots as " + filename)


    #plt.show()

    #outliers = {"cooks": cook_outliers.tolist(), "studentized": stud_outliers.tolist()}
    #print(f"Cook's distance outliers  (>{cook_threshold:.4f}): {outliers['cooks']}")
    #print(f"Studentized residual outliers (|r|>{stud_threshold}): {outliers['studentized']}")


    return fig, axes#, outliers





#Function produced with AI assistance
def plot_hist_with_normal(series, filename="histogram.png", bins=30,title_stem=None):
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
        count, bins_edges, _ = plt.hist(series, bins=bins, density=True, alpha=0.6, color='steelblue', edgecolor='black')

        # Create normal density curve
        x = np.linspace(series.min(), series.max(), 200)
        y = stats.norm.pdf(x, mean, std)
        plt.plot(x, y, 'r-', linewidth=2, label='Normal PDF')

        # Add labels and title
        plt.xlabel("Value")
        plt.ylabel("Density")
        if title_stem is not None:
            plt.title(f"{title_stem} (Normal Curve mean={mean:.2f}, std={std:.2f})",wrap=True)
        else:
            plt.title(f"Histogram with Normal Curve (mean={mean:.2f}, std={std:.2f})",wrap=True)
        plt.legend()

        # Save the plot
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close('all')
        print(f"Saved histogram with normal curve as '{filename}'")


#Function produced with AI assistance
def plot_normality(series, filename="histogram.png", bins=30,title_stem=None):
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

        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
        ## QQ plot for normality
        qqplot(series, line='s', ax=axes[0], markerfacecolor="steelblue", markeredgecolor="steelblue", marker='o', alpha=0.9)
        axes[0].set_title("Q-Q Plot for Normality",fontsize=9)
        # Create histogram
        #plt.figure(figsize=(8, 5))
        count, bins_edges, _ = axes[1].hist(series, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black')

        # Create normal density curve
        x = np.linspace(series.min(), series.max(), 200)
        y = stats.norm.pdf(x, mean, std)
        axes[1].plot(x, y, 'r-', linewidth=2, label='Normal PDF')

        # Add labels and title
        axes[1].set_title(f"Histogram with Normal Curve (mean={mean:.2f}, std={std:.2f})",wrap=True,fontsize=9)
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Density")
        if title_stem is not None:
            plt.suptitle(f"{title_stem}",wrap=True,fontsize=10)
        else:
            plt.suptitle(f"Normality Plots for Series with mean={mean:.2f}, std={std:.2f}",wrap=True,fontsize=10)


        # Save the plot
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close('all')
        print(f"Saved histogram with normal curve as '{filename}'")


## Function written with AI assistance
def plot_regression_fit(df, fitmodel=None,y_pred=None, y_col=None,filename=None,cook_outliers=None, stud_outliers=None,):
    """
    Plots observed vs predicted values for a multiple regression model.
    df : pandas DataFrame
        Data containing the features and target.
    model : fitted regression model
        Any model with a .predict() method (sklearn or statsmodels).
    y_col : str
        Name of the target variable in `df`.
    """
    fig, axes = plt.subplots(1, 1, figsize=(6,5))



    if y_pred is None:
        if fitmodel is None:
            raise Exception("Either y_pred or fitmodel is needed to plot the fitted verse observed points.")
        else:
            # Handle statsmodels and sklearn
            try:
                y_pred = fitmodel.predict(df.drop(columns=[y_col]))
            except Exception:
                y_pred = fitmodel.predict(df)

    if y_col is None:
        y_col=fitmodel.model.endog_names
    #print(y_col)
    y_obs = df[y_col] #observed values

    n = len(y_pred)
    all_indices = np.arange(n)
    outlier_set = set(stud_outliers) if stud_outliers is not None else set()
    influential_set = set(cook_outliers) if cook_outliers is not None else set()
    special_set = outlier_set | influential_set
    normal_idx = [i for i in all_indices if i not in special_set]
    outlier_only_idx = [i for i in all_indices if i in outlier_set and i not in influential_set]
    influential_only_idx = [i for i in all_indices if i in influential_set and i not in outlier_set]
    both_idx = [i for i in all_indices if i in outlier_set and i in influential_set]

    # Best-fit line
    coef = np.polyfit(y_pred, y_obs, 1)
    poly = np.poly1d(coef)

    # Helper to scatter subgroups with distinct styles
    def scatter_groups(ax, x, y, alpha=0.5, size=30):
        ax.scatter(x[normal_idx], y[normal_idx], color='black', alpha=alpha, s=size, label='Normal')
        if outlier_only_idx:
            ax.scatter(x[outlier_only_idx], y[outlier_only_idx], color='orange',
                       marker='^', s=size * 2, alpha=0.9, label='Outlier')
        if influential_only_idx:
            ax.scatter(x[influential_only_idx], y[influential_only_idx], color='steelblue',
                       marker='D', s=size * 2, alpha=0.9, label='Influential')
        if both_idx:
            ax.scatter(x[both_idx], y[both_idx], color='fuchsia',
                       marker='*', s=size * 3, alpha=0.9, label='Outlier & Influential')

    ycol_arr=np.array(y_col)
    fitted_arr=np.array(y_pred)
    ## Observed vs Fitted
    if special_set:
        scatter_groups(axes[0], fitted_arr, ycol_arr, alpha=0.6)
        axes[0].legend(fontsize=7)
    else:
        axes[0].scatter(fitted_arr, ycol_arr, alpha=0.6, label='Data')

    try:
        coef = np.polyfit(fitted_arr, ycol_arr, 1)
        poly = np.poly1d(coef)
        axes[0].plot(fitted_arr, poly(fitted_arr), linewidth=2, label='Best-fit line',color="orange")
    except:
        print("Could not fit 'best fit line' for observed vs fitted plot.")

    min_val = min(ycol_arr.min(), fitted_arr.min())
    max_val = max(ycol_arr.max(), fitted_arr.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], linestyle='--', label='Ideal fit',color="crimson")
    axes[0].set_xlabel("Fitted")
    axes[0].set_ylabel("Observed Values")
    axes[0].set_title("Observed vs Fitted")
    axes[0].legend()

    # plt.figure(figsize=(6, 5))
    # plt.scatter(y_pred, y_obs, alpha=0.6, label="Data")
    # plt.plot(y_pred, poly(y_pred), linewidth=2, label="Best-fit line")
    #
    # # 45-degree perfect-fit line
    # min_val = min(y_obs.min(), y_pred.min())
    # max_val = max(y_obs.max(), y_pred.max())
    # plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Ideal fit")
    #
    # plt.xlabel("Predicted Values")
    # plt.ylabel("Observed Values")
    # plt.title("Observed vs Predicted")
    # plt.legend()
    plt.grid(True)

    if filename is not None:
        plt.savefig(filename, dpi=300)
        plt.close('all')
        print(f"Saved observed vs fitted as '{filename}'")
    else:
        plt.show()
    return plt


