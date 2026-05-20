import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import numpy as np
import os

try:
    df = pd.read_csv("../OrigQCEWStudy/nj34_qdb_2016_1.csv")
    distinct_cnty = df['cnty'].unique()
    print("Distinct cnty values:")
    print(sorted(distinct_cnty))
except FileNotFoundError:
    print("Error: File not found.")


def load_combined_data(directory):
    dfs = []

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)

            # Read CSV file
            df = pd.read_csv(filepath)
            dfs.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df

real_nj_data = load_combined_data('./TrueQCEW/NewJersey/')
real_nj_data = real_nj_data[real_nj_data['qtr'] == 1].copy()

real_nj_data['state'] = real_nj_data['area_fips'].astype(str).str[:2]  # First 2 digits
real_nj_data['cnty'] = real_nj_data['area_fips'].astype(str).str[2:]   # Last 3 digits

# Remove the area_fips column
real_nj_data = real_nj_data.drop(columns=['area_fips'])
# List of columns to keep
columns_to_keep = [
    'year',
    'qtr',
    'state',
    'cnty',
    'own_code',
    'industry_code',
    'month1_emplvl',
    'month2_emplvl',
    'month3_emplvl',
    'total_qtrly_wages',
    'avg_wkly_wage',
    'agglvl_code',
    'disclosure_code'
]
real_nj_data = real_nj_data[columns_to_keep]
real_nj_data = real_nj_data.rename(columns={'month1_emplvl':'m1emp',
                                            'month2_emplvl':'m2emp',
                                            'month3_emplvl':'m3emp'
                                           })
print(real_nj_data.head())
synth_nj_df = pd.read_csv('./nj34_qdb_2016_1.csv')

def compare_df()
synth_nj_df_22_sums = synth_nj_df_22.groupby(
    ['year', 'qtr', 'state', 'cnty', 'own', 'naics_sector']
).agg({
    'm1emp': 'sum',
    'm2emp': 'sum',
    'm3emp': 'sum'
}).reset_index()

real_nj_22 = real_nj_data[
    (real_nj_data['industry_code'].str[:2]=='22') &
    (real_nj_data['industry_code'].astype(str).str.len() == 2) &
    (real_nj_data['own_code'] == 5)
][['year', 'qtr', 'state', 'cnty', 'industry_code','own_code', 'm1emp', 'm2emp', 'm3emp','disclosure_code']].copy()


real_nj_22['cnty'] = real_nj_22['cnty'].astype(int)
real_nj_22['state'] = real_nj_22['state'].astype(int)

# Merge the dataframes on county
merged_df = pd.merge(
    synth_nj_df_22_sums,
    real_nj_22,
    on=['year', 'qtr', 'state', 'cnty'],
    suffixes=('_synth', '_real')
)

# Calculate differences
merged_df['m1emp_diff'] = merged_df['m1emp_synth'] - merged_df['m1emp_real']
merged_df['m2emp_diff'] = merged_df['m2emp_synth'] - merged_df['m2emp_real']
merged_df['m3emp_diff'] = merged_df['m3emp_synth'] - merged_df['m3emp_real']

# Select relevant columns for the result
result_df_22 = merged_df[['year', 'qtr', 'state', 'cnty',
                      'm1emp_synth', 'm1emp_real', 'm1emp_diff',
                      'm2emp_synth', 'm2emp_real', 'm2emp_diff',
                      'm3emp_synth', 'm3emp_real', 'm3emp_diff']]

real_nj_data_2dig = real_nj_data[
    (real_nj_data['industry_code'].str.len() == 2) &
    (real_nj_data['own_code'] == 5)
]
synth_nj_df_2dig = synth_nj_df.groupby(
    ['year', 'qtr', 'state', 'cnty', 'own', 'naics_sector']
).agg({
    'm1emp': 'sum',
    'm2emp': 'sum',
    'm3emp': 'sum'
}).reset_index()

# Convert to string first to handle any non-numeric values, then to int
synth_nj_df_2dig['naics_sector'] = synth_nj_df_2dig['naics_sector'].astype(str).str.strip().astype(int)
real_nj_data_2dig['industry_code'] = real_nj_data_2dig['industry_code'].astype(str).str.strip().astype(int)
# First get the intersection of common NAICS codes
common_codes = set(synth_nj_df_2dig['naics_sector']).intersection(set(real_nj_data_2dig['industry_code']))

# Filter synthetic data to only keep rows with common codes
synth_nj_df_2dig = synth_nj_df_2dig[synth_nj_df_2dig['naics_sector'].isin(common_codes)]

# Filter real data to only keep rows with common codes
real_nj_data_2dig = real_nj_data_2dig[real_nj_data_2dig['industry_code'].isin(common_codes)]

real_nj_data_2dig['cnty'] = real_nj_data_2dig['cnty'].astype(int)
real_nj_data_2dig['state'] = real_nj_data_2dig['state'].astype(int)
real_nj_data_2dig['naics_sector'] = real_nj_data_2dig['industry_code']
# Merge the dataframes on county
merged_df = pd.merge(
    synth_nj_df_2dig,
    real_nj_data_2dig,
    on=['year', 'qtr', 'state', 'cnty', 'naics_sector'],
    suffixes=('_synth', '_real')
)

# Calculate differences
merged_df['m1emp_diff'] = merged_df['m1emp_synth'] - merged_df['m1emp_real']
merged_df['m2emp_diff'] = merged_df['m2emp_synth'] - merged_df['m2emp_real']
merged_df['m3emp_diff'] = merged_df['m3emp_synth'] - merged_df['m3emp_real']

# Select relevant columns for the result
result_df_2dig = merged_df[['year', 'qtr', 'state', 'cnty', 'naics_sector',
                      'm1emp_synth', 'm1emp_real', 'm1emp_diff',
                      'm2emp_synth', 'm2emp_real', 'm2emp_diff',
                      'm3emp_synth', 'm3emp_real', 'm3emp_diff']]
result_df_2dig = result_df_2dig[result_df_2dig['m1emp_real'] != 0]


def empresultout(result_df, naicslevel):
    # Summary stats for differences
    diff_stats = result_df[['m1emp_diff', 'm2emp_diff', 'm3emp_diff']].describe()
    print("Difference Statistics:")
    print(diff_stats)

    # Correlation matrix
    corr_matrix = result_df[['m1emp_diff', 'm2emp_diff', 'm3emp_diff']].corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix)

    # By NAICS sector
    sector_stats = result_df.groupby(naicslevel)[['m1emp_diff', 'm2emp_diff', 'm3emp_diff']].median()
    print("\nMedian Differences by NAICS Sector:")
    print(sector_stats)

    ##### Plots #####
    ## Boxplots (without fliers)
    result_df['m1emp_reldiff'] = (result_df['m1emp_synth'] - result_df['m1emp_real']) / result_df['m1emp_real']
    result_df['m2emp_reldiff'] = (result_df['m2emp_synth'] - result_df['m2emp_real']) / result_df['m2emp_real']
    result_df['m3emp_reldiff'] = (result_df['m3emp_synth'] - result_df['m3emp_real']) / result_df['m3emp_real']

    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    sns.boxplot(data=result_df[['m1emp_diff', 'm2emp_diff', 'm3emp_diff']], showfliers=False, ax=ax1)
    ax1.set_title(f'Distribution of Employment Differences (Synthetic - Real) ({naicslevel})')
    ax1.set_ylabel('Difference in Employment Counts')
    ax1.set_xlabel('Employment Metric')
    ax1.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax1.set_xticks([0, 1, 2], ['M1EMP', 'M2EMP', 'M3EMP'])

    sns.boxplot(data=result_df[['m1emp_reldiff', 'm2emp_reldiff', 'm3emp_reldiff']], showfliers=False, ax=ax2)
    ax2.set_title(f'Relative Employment Differences\n(Synthetic - Real) / (Real) ({naicslevel})')
    ax2.set_ylabel('Relative Difference')
    ax2.set_xlabel('Employment Metric')
    ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xticks([0, 1, 2], ['M1EMP', 'M2EMP', 'M3EMP'])

    plt.tight_layout()
    plt.show()

    ## Boxplots (with fliers)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    sns.boxplot(data=result_df[['m1emp_diff', 'm2emp_diff', 'm3emp_diff']], showfliers=True, ax=ax1)
    ax1.set_title(f'Distribution of Employment Differences (Synthetic - Real) ({naicslevel})')
    ax1.set_ylabel('Difference in Employment Counts')
    ax1.set_xlabel('Employment Metric')
    ax1.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax1.set_xticks([0, 1, 2], ['M1EMP', 'M2EMP', 'M3EMP'])

    sns.boxplot(data=result_df[['m1emp_reldiff', 'm2emp_reldiff', 'm3emp_reldiff']], showfliers=True, ax=ax2)
    ax2.set_title(f'Relative Employment Differences\n(Synthetic - Real) / (Real) ({naicslevel})')
    ax2.set_ylabel('Relative Difference')
    ax2.set_xlabel('Employment Metric')
    ax2.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xticks([0, 1, 2], ['M1EMP', 'M2EMP', 'M3EMP'])

    plt.tight_layout()
    plt.show()

    ## Scatter plots with regression
    metrics = ['m1emp', 'm2emp', 'm3emp']
    plt.figure(figsize=(15, 5))

    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i + 1)
        sns.scatterplot(data=result_df, x=f'{metric}_real', y=f'{metric}_synth', s=60, alpha=0.7)

        X = sm.add_constant(result_df[f'{metric}_real'])
        model = sm.OLS(result_df[f'{metric}_synth'], X).fit()
        predictions = model.get_prediction(X)
        pred_frame = predictions.summary_frame(alpha=0.05)

        plt.plot(result_df[f'{metric}_real'], model.predict(X), color='blue',
                 label=f'y = {model.params[1]:.2f}x + {model.params[0]:.2f}\nR² = {model.rsquared:.2f}')
        plt.fill_between(result_df[f'{metric}_real'], pred_frame['obs_ci_lower'], pred_frame['obs_ci_upper'],
                         color='blue', alpha=0.2, label='95% CI')
        max_val = max(result_df[f'{metric}_real'].max(), result_df[f'{metric}_synth'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect agreement')

        plt.title(f'Synthetic vs Real Employment ({metric.upper()}) ({naicslevel})')
        plt.xlabel('Real Employment')
        plt.ylabel('Synthetic Employment')
        plt.legend()

    plt.tight_layout()
    plt.show()

    ## Employment distribution histograms (linear and log-scale)
    plt.figure(figsize=(15, 4))
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i + 1)
        sns.histplot(result_df[f'{metric}_real'], color='blue', kde=True, label='Real', alpha=0.6)
        sns.histplot(result_df[f'{metric}_synth'], color='orange', kde=True, label='Synthetic', alpha=0.6)
        plt.title(f'Distribution of Real vs Synthetic Employment ({metric.upper()}) ({naicslevel})')
        plt.xlabel('Employment')
        plt.ylabel('Count')
        plt.legend()
    plt.tight_layout()
    plt.show()

    # Log-scale version using log1p
    plt.figure(figsize=(15, 4))
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i + 1)
        sns.histplot(np.log1p(result_df[f'{metric}_real']), color='blue', kde=True, label='Real', alpha=0.6)
        sns.histplot(np.log1p(result_df[f'{metric}_synth']), color='orange', kde=True, label='Synthetic', alpha=0.6)
        plt.title(f'Log-Scale Distribution of Real vs Synthetic Employment ({metric.upper()}) ({naicslevel})')
        plt.xlabel('log(1 + Employment)')
        plt.ylabel('Count')
        plt.legend()
    plt.tight_layout()
    plt.show()

empresultout(result_df_2dig, 'naics_sector')

real_nj_data_3dig = real_nj_data[
    (real_nj_data['industry_code'].str.len() == 3) &
    (real_nj_data['own_code'] == 5)
]
synth_nj_df_3dig = synth_nj_df.groupby(
    ['year', 'qtr', 'state', 'cnty', 'own', 'naics3']
).agg({
    'm1emp': 'sum',
    'm2emp': 'sum',
    'm3emp': 'sum'
}).reset_index()
# Convert to string first to handle any non-numeric values, then to int
synth_nj_df_3dig['naics3'] = synth_nj_df_3dig['naics3'].astype(str).str.strip().astype(int)
real_nj_data_3dig['industry_code'] = real_nj_data_3dig['industry_code'].astype(str).str.strip().astype(int)
# First get the intersection of common NAICS codes
common_codes = set(synth_nj_df_3dig['naics3']).intersection(set(real_nj_data_3dig['industry_code']))

# Filter synthetic data to only keep rows with common codes
synth_nj_df_3dig = synth_nj_df_3dig[synth_nj_df_3dig['naics3'].isin(common_codes)]

# Filter real data to only keep rows with common codes
real_nj_data_3dig = real_nj_data_3dig[real_nj_data_3dig['industry_code'].isin(common_codes)]
real_nj_data_3dig['cnty'] = real_nj_data_3dig['cnty'].astype(int)
real_nj_data_3dig['state'] = real_nj_data_3dig['state'].astype(int)
real_nj_data_3dig['naics3'] = real_nj_data_3dig['industry_code']
# Merge the dataframes on county
merged_df = pd.merge(
    synth_nj_df_3dig,
    real_nj_data_3dig,
    on=['year', 'qtr', 'state', 'cnty', 'naics3'],
    suffixes=('_synth', '_real')
)

# Calculate differences
merged_df['m1emp_diff'] = merged_df['m1emp_synth'] - merged_df['m1emp_real']
merged_df['m2emp_diff'] = merged_df['m2emp_synth'] - merged_df['m2emp_real']
merged_df['m3emp_diff'] = merged_df['m3emp_synth'] - merged_df['m3emp_real']

# Select relevant columns for the result
result_df_3dig = merged_df[['year', 'qtr', 'state', 'cnty', 'naics3',
                      'm1emp_synth', 'm1emp_real', 'm1emp_diff',
                      'm2emp_synth', 'm2emp_real', 'm2emp_diff',
                      'm3emp_synth', 'm3emp_real', 'm3emp_diff']]
result_df_3dig = result_df_3dig[result_df_3dig['m1emp_real'] != 0]

empresultout(result_df_3dig, 'naics3')