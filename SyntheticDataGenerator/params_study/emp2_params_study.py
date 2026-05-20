import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import statsmodels.api as sm
import sys
import os
from stargazer.stargazer import Stargazer
import importlib
import random
sys.path.append(os.path.abspath("../NAICS6_Pyfunctions/"))
from EmploymentFunctions import *
from GeneralFunctions import *
from WageFunctions import *
from NAICS6functions import *
from get_microdata import *
from MicrodataPostprocessing import *
from generate_NAICS6 import *
from config_reader import *
from CBP_QWI_download import *
from download_QCEWdata import *
from preprocess_combine import *
from adjustmentFunctions import *

#DATASETloc = "DataDiag/PythonPreprocessOut/combine_data_subset.csv"
pd.options.mode.chained_assignment = None  # default='warn'

naicsdf = process_naics_file("DataDiag/2012_codes.csv", code_col=1, code_sep=',')
configfile="DataDiag/param_model_select/param_select_config.yaml"

generalConfig, microdataConfig, preprocessConfig, employmentConfig, wageConfig,supplementaryConfig, quarterConfig  = check_config(configfile)
foldername = preprocessConfig['DATA_IN_FOLDER']
# createcombinedstart=time.time()
# df=combine_qwi_cbp_qcew(rawfile=foldername + preprocessConfig['CBPDATA'],
#                         imputedfile=foldername + preprocessConfig['IMPUTECBP'],
#                         qwifolder=foldername + preprocessConfig['QWIDIR'],
#                         outfilename=generalConfig['COMBINED_DATA'],
#                         diagnosticsfile=preprocessConfig["DIAGNOSTIC_FILE"],
#                         generalConfig=generalConfig,
#                         preprocessConfig=preprocessConfig,
#                         quarterConfig=quarterConfig,
#                         supplementaryConfig=supplementaryConfig,
#                         outfilepath=preprocessConfig['OUTPATH'],
#                         year=generalConfig['YEAR'],
#                         naicsdf=naicsdf)
#
# #df = pd.read_csv(DATASETloc, dtype=str)# nrows=100000)
# df = df.iloc[:, 1:] # Remove index
#
# #cntydata = pd.read_csv("DataDiag/DataIn/rural_urban_fips.txt", sep='\t')
# #cntydata["FIPS"] = pd.to_numeric(
# #        cntydata["FIPS"], errors="coerce").round(0).astype(int).astype(str)
# #df['geography'] = df['geoindkey'].str.extract(r'^([^_]+)')  # split['_'].str[0]
# #df = df.merge(cntydata, right_on="FIPS", left_on="geography", how="left")
#
# dfsave = df.copy()
# columns_to_convert = ['emp1', 'emp2','emp3','wages','estnum',"avg_month_emp_wages", 'year',"Population_2010"]
# df[columns_to_convert] = df[columns_to_convert].astype(float)
# df=fill_from_geoindkey(df,naicsdata=naicsdf)
# df6 = df[df['geoindkey'].str.contains("_[0-9]{6}", regex=True)].copy()
# df4 = df[df['geoindkey'].str.contains("_[0-9]{4}[^0-9]{2}", regex=True)].copy()
#
# for vname in ["emp1", "wages"]:
#     count6dig = get_codes_summary(df, groupbydigits=4, levelgrouped=6, variable=vname)
#     df4 = df4.merge(count6dig, on=['geo4naics'], how='left', indicator=True, suffixes=["", "_droplater"])
#     weirdones = df4.loc[df4["_merge"] == "left_only"]  # only appear in county by NAICS-4 codes
#     geo6naicsweird = df6.loc[df6["geo4naics"].isin(weirdones['geo4naics']), :]
#     if len(geo6naicsweird) == 0:  ##If no county by NAICS6 codes, hard code the relevant values
#         df4.loc[df4[vname + "_sum6by4"].isna(), vname + "_sum6by4"] = 0
#         df4.loc[df4[vname + "_missing6by4"].isna(), vname + "_missing6by4"] = 0
#         df4.loc[df4[vname + "_propmissing6by4"].isna(), vname + "_propmissing6by4"] = 1
#         df4.loc[df4['count6by4codes'].isna(), 'count6by4codes'] = 0
#         df4.drop(columns=[vname+"_notna","estnum_notna","emp3_notna"],inplace=True,errors="ignore")
#     else:  # otherwise print diagnostic information
#         print(f'count na in {vname}_sum6by4 {sum(df4.loc[:, vname + "_sum6by4"].isna())}')
#         print(f'in {vname}_missing6by4 {sum(df4.loc[:, vname + "_missing6by4"].isna())}')
#         raise Exception(f"Something wrong\n {geo6naicsweird.head()}")
#     df4.drop(columns=["_merge"], inplace=True)
#     df4.drop(columns=[dropcol for dropcol in df4.columns if "_droplater" in dropcol], errors="ignore", inplace=True)
#
#     ## Get difference between county by NAICS4 and sum of known county by NAICS6
#     df4[vname + "diff"] = df4[vname].astype(float) - df4[vname + '_sum6by4'].astype(float)
#
#
#
# datavardict={"qcew":["emp1","emp3","wages","estnum"],
#              "cbp":["emp3","wages","estnum"],
#              "qwi":["emp1","emp3"]}
# for key,value in datavardict.items():
#     df4.drop(columns=[vname+"_"+key for vname in datavardict[key]],inplace=True)
#     if key!="old":
#         df4.drop(columns=[vname+"_perestnum_"+key for vname in datavardict[key] if vname!="estnum"],inplace=True)
#         if key!="qcew":
#             df4.drop(columns=[vname + "_" +key+"_flag" for vname in datavardict[key] if vname!="estnum"], inplace=True)
# contcols2=[x for x in df4.columns if "perestnum" in x]
# df4[contcols2] = df4[contcols2].astype(float)
#
# moredrop=["year","qtr","naics5","avg_month_emp_wages_flag","row_sources","County_Name","Description","year_qtr",
#           "geo5naics","ind_level","agglvl_code","grouplevels","FIPS","year_qtr_cbp",
#           "sum_emp1_","sum_emp1_qcew","count_emp1_from_","count_emp1_from_qcew",
#           "sum_wages_","sum_wages_qcew","count_wages_from_","count_wages_from_qcew","sum_wages_cbp","count_wages_from_cbp"]
#
#
# df4.drop(columns=moredrop,inplace=True,errors="ignore")
# summarize_dataframe(df4)
#
#df4.to_csv("DataDiag/param_model_select/subset_combined_cntyN4.csv")

df4 = pd.read_csv("../DataDiag/param_model_select/subset_combined_cntyN4.csv", dtype=str)# nrows=100000)
df4=df4.loc[df4["State"].isin(generalConfig["STATES"]),:].copy()
#summarize_dataframe(df4)
floatcols=["emp2_qcew","avg_month_emp_wages","Population_2010","RUCC_2013","estnum","emp3","emp2","emp1","wages",
           "emp1_perestnum","emp2_perestnum","emp3_perestnum","wages_perestnum","emp1diff","wagesdiff"]+[cname for cname in df4.columns if "6by4" in cname]
df4[floatcols] = df4[floatcols].astype(float)
