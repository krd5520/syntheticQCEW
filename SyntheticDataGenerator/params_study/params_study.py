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

sys.path.append(os.path.abspath("./NAICS6_Pyfunctions/"))
from model_investigation import *
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

DATASETloc = "DataDiag/PythonPreprocessOut/combine_data_subset.csv"
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
# df4.to_csv("DataDiag/param_model_select/subset_combined_cntyN4.csv")

#raise Exception("stop here")
df4 = pd.read_csv("./DataDiag/param_model_select/subset_combined_cntyN4.csv", dtype=str)# nrows=100000)
df4=df4.loc[df4["State"].isin(generalConfig["STATES"]),:].copy()
#summarize_dataframe(df4)
floatcols=["emp2_qcew","avg_month_emp_wages","Population_2010","RUCC_2013","estnum","emp3","emp2","emp1","wages",
           "emp1_perestnum","emp2_perestnum","emp3_perestnum","wages_perestnum","emp1diff","wagesdiff"]+[cname for cname in df4.columns if "6by4" in cname]
df4[floatcols] = df4[floatcols].astype(float)

#testdf4=df4.sample(5000,ignore_index=True)
#summarize_dataframe(testdf4)
# ############## Employment Counts ################
# # Step 1: Create employment prediction model
# #empfunc.employmentConfig.TESTFORMULAS = True
employment_config=employmentConfig
# model_dict={"emp1_model":["Response: Month 1 Employment", "emp1 ~ poly(np.sqrt(emp3),3) + poly(np.log10(estnum),2)+emp1_sum6by4 + C(state) + C(sector)+np.log10(Population_2010)+C(RUCC_2013)+ emp3_perestnum"],
#         "sqrtemp1_model":["Response: sqrt(Month 1 Employment)", "np.sqrt(emp1) ~ poly(np.sqrt(emp3),3) + poly(np.log10(estnum),2)+np.sqrt(emp1_sum6by4) + C(state) + C(sector)+np.log10(Population_2010)+C(RUCC_2013)+ np.sqrt(emp3_perestnum)"],
#         "log1pemp1_model":["Response: ln(1+ Month 1 Employment)", "np.log1p(emp1) ~ poly(np.log1p(emp3),3) + poly(np.log10(estnum),2)+np.log1p(emp1_sum6by4) + C(state) + C(sector)+np.log10(Population_2010)+C(RUCC_2013)+ np.log1p(emp3_perestnum)"],
#         "emp1diff_model":["Response: Month 1 Employment Suppression Difference","emp1diff ~ emp1_propmissing6by4+count6by4codes+emp1_propmissing6by4*count6by4codes+poly(np.sqrt(emp3_emp1_missing6by4),3)+poly(np.log10(estnum_emp1_missing6by4),2)+C(sector)+C(state)+np.log10(Population_2010)+C(RUCC_2013)+emp3_perestnum"],
#         "sqrtemp1diff_model":["Response: sqrt(Month 1 Employment Suppression Difference)", "np.sqrt(emp1diff) ~ emp1_propmissing6by4+count6by4codes+ emp1_propmissing6by4*count6by4codes+poly(np.sqrt(emp3_emp1_missing6by4),3)+poly(np.log10(estnum_emp1_missing6by4),2)+C(sector)+C(state)+np.log10(Population_2010)+C(RUCC_2013)+np.sqrt(emp3_perestnum)"],
#         "log1pemp1diff_model":["Response: ln(1+Month 1 Employment Suppression Difference)","np.log1p(emp1diff) ~ emp1_propmissing6by4+count6by4codes+emp1_propmissing6by4*count6by4codes+poly(np.log1p(emp3_emp1_missing6by4),3)+poly(np.log10(estnum_emp1_missing6by4),2)+C(sector)+C(state)+np.log10(Population_2010)+C(RUCC_2013)+np.log1p(emp3_perestnum)"]}
#temp=compare_models(data=df4,modelsdict=model_dict,config=employmentConfig,foldername="DataDiag/param_model_select",tablefname="response_emp1_compare.tex",
#                    csvfname="response_emp1_compare.csv",
#                    diagplotstem="DiagnosticPlots/month1employment_diagnostics_")


random.seed(2)
# ## Repeat process but look at different predictors
# model_dict_empdiff1={"log1pemp1diff_basemodel":["Base Model", "np.log1p(emp1diff) ~ emp1_propmissing6by4+count6by4codes+emp1_missing6by4+poly(np.log1p(emp3_emp1_missing6by4),3)+poly(np.log10(estnum_emp1_missing6by4),2)+C(sector)+C(state)+np.log10(Population_2010)+C(RUCC_2013)+np.log1p(emp3_perestnum)"],
# "log1pemp1diff_noRUCC":["Remove RUCC", "np.log1p(emp1diff) ~ emp1_propmissing6by4+count6by4codes+emp1_missing6by4+poly(np.log1p(emp3_emp1_missing6by4),3)+poly(np.log10(estnum_emp1_missing6by4),2)+C(sector)+C(state)+np.log10(Population_2010)+np.log1p(emp3_perestnum)"],
# 'log1pemp1diff_reduced':["+Remove count6by4codes", "np.log1p(emp1diff) ~ emp1_propmissing6by4+emp1_missing6by4+poly(np.log1p(emp3_emp1_missing6by4),3)+poly(np.log10(estnum_emp1_missing6by4),2)+C(sector)+C(state)+np.log10(Population_2010)+np.log1p(emp3_perestnum)"],
# 'log1pemp1diff_reduced_supersector_swap':["Reduced predictors. Use supersector", "np.log1p(emp1diff) ~ emp1_propmissing6by4+poly(np.log1p(emp3_emp1_missing6by4),3)+poly(np.log10(estnum_emp1_missing6by4),2)+C(supersector)+C(state)+np.log10(Population_2010)+np.log1p(emp3_perestnum)"]}#,
# #                     'log1pemp1diff_reduced_supersector_swap_removeVIF':["Option 4: Reduced Predictors Further, Use Supersector", "np.log1p(emp1diff) ~ emp1_propmissing6by4+emp1_missing6by4+poly(np.log10(estnum_emp1_missing6by4),2)+C(supersector)+C(state)+np.log10(Population_2010)+np.log1p(emp3_perestnum)"]}
# # model_dict_empdiff2={'log1pemp1diff_reduced_supersector_swap':["Option 3: Reduced Predictors, Use Supersector", "np.log1p(emp1diff) ~ emp1_propmissing6by4+emp1_missing6by4+poly(np.log1p(emp3_emp1_missing6by4),3)+poly(np.log10(estnum_emp1_missing6by4),2)+C(supersector)+C(state)+np.log10(Population_2010)+np.log1p(emp3_perestnum)"],
# #                      'log1pemp1diff_reduced_interaction_supersector_swap': ["Option 4: Reduced Predictors, Use Supersector, Add Interaction",
# #                                                           "np.log1p(emp1diff) ~ emp1_propmissing6by4+emp1_missing6by4+poly(np.log1p(emp3_emp1_missing6by4),3)+poly(np.log10(estnum_emp1_missing6by4),2)+C(supersector)+C(state)+np.log10(Population_2010)+np.log10(Population_2010):np.log10(estnum_emp1_propmissing6by4)+np.log1p(emp3_perestnum)"],
# #                   'log1pemp1diff_reduced_interaction_supersector_swap_higherpoly':["Option 5: Reduced Predictors, Use Supersector, Add Interaction, Increase Polynomial Terms",
# #     "np.log1p(emp1diff) ~ poly(emp1_propmissing6by4,3)+poly(emp1_missing6by4,3)+poly(np.log1p(emp3_emp1_missing6by4),3)+poly(np.log10(estnum_emp1_missing6by4),3)+C(supersector)+C(state)+poly(np.log10(Population_2010),3)+np.log10(Population_2010):np.log10(estnum_emp1_propmissing6by4)+np.log1p(emp3_perestnum)"]
# # }#,
#
# temp=compare_models(data=df4,modelsdict=model_dict_empdiff1,config=employmentConfig,foldername="DataDiag/param_model_select",tablefname="logp1_emp1diff_compare1.tex",
#                    csvfname="logp1_emp1diff_compare1.csv",
#                    diagplotstem="DiagnosticPlots/month1employment_diagnostics_")
# # temp=compare_models(data=df4,modelsdict=model_dict_empdiff2,config=employmentConfig,foldername="DataDiag/param_model_select",tablefname="logp1_emp1diff_compare2.tex",
# #                     csvfname="logp1_emp1diff_compare2.csv",
# #                     diagplotstem="DiagnosticPlots/month1employment_diagnostics_")


#################### Month 2 Employment Noise Investigation ##########################

nbins=100
df4_emp2_all=df4.loc[df4['emp2'].notna(),['emp1','emp2','emp3']]
df4_emp2_all['midpoint_emp']=df4_emp2_all['emp1']+(0.5*(df4_emp2_all['emp3']-df4_emp2_all['emp1']))
df4_emp2_all['diff_3_2_emp']=df4_emp2_all['emp3']-df4_emp2_all['emp2']
df4_emp2_all['diff_mid_emp']=df4_emp2_all['midpoint_emp']-df4_emp2_all['emp2']
df4_emp2_all['avg_first_last_emp']=0.5*(df4_emp2_all['emp1']+df4_emp2_all['emp3'])
df4_emp2_all['abs_diff']=abs(df4_emp2_all['emp3']-df4_emp2_all['emp1'])



plot_hist_with_normal(df4_emp2_all['diff_3_2_emp'],
                      filename="DataDiag/param_model_select/emp2_noise_plots/all_diff_3_2_emp_histogram.png",
                      bins=nbins,title_stem="Month 2 and 3 Employment Difference",figsize=(6.5,3))
# plot_hist_with_normal(df4_emp2_all['diff_mid_emp'],
#                       filename="DataDiag/param_model_select/emp2_noise_plots/all_diff_mid_emp_histogram.png",
#                       bins=nbins,title_stem="Month 2 Employment Difference From Midpoint", figsize=(8,4))

df4_emp2=df4_emp2_all.loc[(df4_emp2_all['emp1']>0)&(df4_emp2_all['emp3']>0),:].copy()
df4_empsequal=df4_emp2.loc[df4_emp2['abs_diff']==0,:].copy()
df4_emp2=df4_emp2.loc[df4_emp2['abs_diff']>0,:].copy()

df4_emp2['emp2_proposed']=df4_emp2['abs_diff']/df4_emp2['avg_first_last_emp']
df4_emp2['emp2_proposed_flip']=df4_emp2['avg_first_last_emp']/df4_emp2['abs_diff']


createvars={"proposed":"emp2_proposed",
            "avgemp":"avg_first_last_emp",
            "absdiff":"abs_diff",
            "absdiff_p_avgemp":["abs_diff","avg_first_last_emp"]}
for key, value in createvars.items():
    overvar=df4_emp2[value]
    if key=="absdiff_p_avgemp":
        overvar=overvar.iloc[:,0]+overvar.iloc[:,1]
    df4_emp2[f'emp_center0_{key}']=df4_emp2['diff_3_2_emp']/np.sqrt(overvar)
    df4_emp2[f'emp_center0_{key}_flip'] = df4_emp2['diff_3_2_emp'] * np.sqrt(overvar)
    #df4_emp2[f'emp_center0_{key}_sq']=df4_emp2['diff_3_2_emp']/(overvar**2)
    #df4_emp2[f'emp_center0_{key}_flip_sq'] = df4_emp2['diff_3_2_emp'] * (overvar**2)
    df4_emp2[f'emp_center0_{key}_p1']=df4_emp2['diff_3_2_emp']/np.sqrt(1+overvar)
    df4_emp2[f'emp_center0_{key}_flip_p1'] = df4_emp2['diff_3_2_emp'] * np.sqrt(overvar+1)
    #df4_emp2[f'emp_center0_{key}_p1_sq'] = df4_emp2['diff_3_2_emp'] /((1 + overvar)**2)
    df4_emp2[f'emp_center0_{key}_flip_p1_sq'] = df4_emp2['diff_3_2_emp'] * ((overvar + 1)**2)

for key, value in createvars.items():
    overvar=df4_emp2[value]
    if key=="absdiff_p_avgemp":
        overvar=overvar.iloc[:,0]+overvar.iloc[:,1]
    df4_emp2[f'emp_center0_{key}']=np.sqrt(overvar)
    df4_emp2[f'emp_center0_{key}_flip'] = (1/np.sqrt(overvar))
    #df4_emp2[f'emp_center0_{key}_sq']=df4_emp2['diff_3_2_emp']/(overvar**2)
    #df4_emp2[f'emp_center0_{key}_flip_sq'] = df4_emp2['diff_3_2_emp'] * (overvar**2)
    df4_emp2[f'emp_center0_{key}_p1']=np.sqrt(1+overvar)
    df4_emp2[f'emp_center0_{key}_flip_p1'] = (1/np.sqrt(overvar+1))
    #df4_emp2[f'emp_center0_{key}_p1_sq'] = df4_emp2['diff_3_2_emp'] /((1 + overvar)**2)
    #df4_emp2[f'emp_centermid_{key}_flip_p1_sq'] = df4_emp2['diff_mid_emp'] * ((overvar + 1)**2)


df4_empsequal['emp_center0_avgemp']=np.sqrt(df4_empsequal['avg_first_last_emp'])
#df4_empsequal['emp_center0_avgemp_sq']=df4_empsequal['diff_3_2_emp']/((df4_empsequal['avg_first_last_emp'])**2)
df4_empsequal['emp_center0_avgemp_flip']=(1/np.sqrt(df4_empsequal['avg_first_last_emp']))
#df4_empsequal['emp_center0_avgemp_flip_sq']=df4_empsequal['diff_3_2_emp']*((df4_empsequal['avg_first_last_emp'])**2)
df4_empsequal['emp_center0_avgemp_p1']=np.sqrt(1+df4_empsequal['avg_first_last_emp'])
#df4_empsequal['emp_center0_avgemp_p1_sq']=df4_empsequal['diff_3_2_emp']/((1+df4_empsequal['avg_first_last_emp'])**2)
df4_empsequal['emp_center0_avgemp_flip_p1']=(1/np.sqrt(1+df4_empsequal['avg_first_last_emp']))
#df4_empsequal['emp_center0_avgemp_flip_p1_sq']=df4_empsequal['diff_3_2_emp']*((1+df4_empsequal['avg_first_last_emp'])**2)
df4_empsequal['emp_center0_avgemp']=np.sqrt(df4_empsequal['avg_first_last_emp'])
#df4_empsequal['emp_center0_avgemp_sq']=df4_empsequal['diff_3_2_emp']/((df4_empsequal['avg_first_last_emp'])**2)
df4_empsequal['emp_center0_avgemp_flip']=(1/np.sqrt(df4_empsequal['avg_first_last_emp']))
#df4_empsequal['emp_center0_avgemp_flip_sq']=df4_empsequal['diff_3_2_emp']*((df4_empsequal['avg_first_last_emp'])**2)
df4_empsequal['emp_center0_avgemp_p1']=np.sqrt(1+df4_empsequal['avg_first_last_emp'])
#df4_empsequal['emp_center0_avgemp_p1_sq']=df4_empsequal['diff_3_2_emp']/((1+df4_empsequal['avg_first_last_emp'])**2)
df4_empsequal['emp_center0_avgemp_flip_p1']=(1/np.sqrt(1+df4_empsequal['avg_first_last_emp']))
#df4_empsequal['emp_center0_avgemp_flip_p1_sq']=df4_empsequal['diff_3_2_emp']*((1+df4_empsequal['avg_first_last_emp'])**2)



# plot_normality(df4_emp2['diff_3_2_emp'],
#                filename="DataDiag/param_model_select/emp2_noise_plots/unequal_g0_diff_emp3_normality.png",
#                bins=nbins,
#                title_stem="Month 2 Employment Difference from Month 3 Employment Subset Unequal >0",
#                figsize=(8,4))
# plot_hist_with_normal(df4_emp2['diff_3_2_emp'],
#                       filename="DataDiag/param_model_select/emp2_noise_plots/unequal_g0_diff_emp3_histogram.png",
#                       bins=nbins,title_stem="Month 2 Employment Difference From Month 3 Employment Subset Unequal >0",
#                       figsize=(8,4))
#
# plot_hist_with_normal(df4_empsequal['diff_3_2_emp'],
#                       filename="DataDiag/param_model_select/emp2_noise_plots/empsequal_g0_diff_3_2_emp_histogram.png",
#                       bins=nbins,title_stem="Month 2 Employment Difference From Equal >0 Midpoint",
#                       figsize=(8,4))

titledict={"proposed":"Abs. Difference over Avg. Employment",
           "proposed_flip": "Avg. Employment over Abs. Difference",
           "avgemp":"Avg. Employment",
           "avgemp_flip":"1/(Avg. Employment)",
           "absdiff":"Abs. Difference",
           "absdiff_flip":"1/(Abs. Difference)",
           "absdiff_p_avgemp":"Abs. Difference + Avg. Employment",
           "absdiff_p_avgemp_flip":"1/(Abs. Difference + Avg. Employment)",
           "proposed_p1": "1+Abs. Difference over Avg. Employment",
           "proposed_flip_p1": "1+Avg. Employment over Abs. Difference",
           "avgemp_p1": "1+Avg. Employment",
           "avgemp_flip_p1": "1/(1+Avg. Employment)",
           "absdiff_p1": "1+Abs. Difference",
           "absdiff_flip_p1": "1/(1+Abs. Difference)",
           "absdiff_p_avgemp_p1": "1+Abs. Difference + Avg. Employment",
           "absdiff_p_avgemp_flip_p1": "1/(1+Abs. Difference + Avg. Employment)",
           }

normalitytest=pd.DataFrame({"varname":titledict.keys(),"label":titledict.values(),
                            "anderson_stat":np.nan*len(titledict.keys()),
                            "jb_stat":np.nan*len(titledict.keys()),
                            "jb_pval":np.nan*len(titledict.keys())})
# for vname in ["proposed","absdiff","avgemp","absdiff_p_avgemp"]:
#     for sfx in ["_flip","","_p1","_flip_p1"]:
#         for pwr in [1,2,3,4]:
#             if pwr==1:
#                 pwrstem=""
#             else:
#                 pwrstem="_"+str(pwr)
#             labcol="("+titledict[f'{vname}{sfx}']+")^("+str(pwr)+"/2)"
#
#             for centerval in ['diff_mid_emp','diff_3_2_emp']:
#
#                 vseries=df4_emp2[centerval]/(df4_emp2[f'emp_center0_{vname}{sfx}']**pwr)
#                 res=stats.anderson(vseries.to_numpy(),'norm')#,stats.norm.cdf)
#
#                 temp_stat, temp_p=stats.jarque_bera(vseries.to_numpy())
#         #sw_stat, sw_p=stats.shapiro(df4_emp2[f'emp_center0_{vname}{sfx}'].to_numpy())
#                 if normalitytest.loc[normalitytest['varname']==f'{centerval}_{vname}{sfx}{pwrstem}',:].shape[0]==0:
#                     normalitytest.loc[len(normalitytest)]={'varname':f'{centerval}_{vname}{sfx}{pwrstem}',
#                                                    'label':labcol,
#                                                    'anderson_stat':res.statistic,
#                                                    'jb_stat':temp_stat,
#                                                    'jb_pval':temp_p}
#                 else:
#                     normalitytest.loc[normalitytest['varname'] == f'{centerval}_{vname}{sfx}{pwrstem}', ['anderson_stat','jb_stat','jb_pval']]=[res.statistic,temp_stat,temp_p]
#
#         #normalitytest.loc[normalitytest['varname'] == f'{vname}{sfx}', 'sw_stat'] = sw_stat
#         #normalitytest.loc[normalitytest['varname'] == f'{vname}{sfx}', 'sw_pval'] = sw_p
#
#         #normalitytest.loc[normalitytest['varname']==f'{vname}{sfx}','critval']=",".join(str(res.critical_values))
#         #normalitytest.loc[normalitytest['varname']==f'{vname}{sfx}','signlvl']=",".join(str(res.significance_level))
#
#                 plot_normality(vseries,
#                        filename="DataDiag/param_model_select/emp2_noise_plots/emp_center_"+centerval+"_"+vname+sfx+"_"+str(pwr)+"_emp_normality.png",
#                        bins=nbins,
#                        title_stem="Month 2 Employment Unequal >0 Subset: Difference over "+labcol,
#                                figsize=(8,3))

# print("Unequal Subset")
# print(normalitytest.sort_values(by="jb_stat").head(15))
# normalitytest_emp3=normalitytest.copy()
#
# print(f"Difference From Month 3 Employment Unequal >0: \n{stats.anderson(df4_emp2['diff_3_2_emp'].to_numpy(),dist='norm')}")#,stats.norm.cdf))
# temp_stat, temp_p = stats.jarque_bera(df4_emp2[f'diff_3_2_emp'].to_numpy())
# #sw_stat, sw_p = stats.shapiro(df4_emp2['diff_3_2_emp'].to_numpy())
# print(f"Jarque-Bera: stat={temp_stat} p={temp_p}")#; Shapiro-Wilks: stat={sw_stat} p={sw_p}")
#
# print("Equal Nonzero Subset")
# print(f"Difference From Midpoint Equal >0: \n{stats.anderson(df4_empsequal['diff_3_2_emp'].to_numpy(),dist='norm')}")#,stats.norm.cdf))
# temp_stat, temp_p = stats.jarque_bera(df4_empsequal[f'diff_3_2_emp'].to_numpy())
# #sw_stat, sw_p = stats.shapiro(df4_empsequal['diff_3_2_emp'].to_numpy())
# print(f"Jarque-Bera: stat={temp_stat} p={temp_p}")#; Shapiro-Wilks: stat={sw_stat} p={sw_p}")

# plot_normality(df4_empsequal['diff_3_2_emp'], filename="DataDiag/param_model_select/emp2_noise_plots/empsequal_diff_3_2_emp_normality.png", bins=nbins,
#                title_stem="Month 2 Employment Difference from Midpoint Equal >0 Subset")
printlater=[]
# for sfx in ["_flip", "", "_p1", "_flip_p1"]:
#     for pwr in [1, 2, 3, 4]:
#         if pwr == 1:
#             pwrstem = ""
#         else:
#             pwrstem = "_" + str(pwr)
#         labcol = "(" + titledict[f'avgemp{sfx}'] + ")^(" + str(pwr) + "/2)"
#
#         #tstem = titledict[f'{vname}{sfx}']
#         vseries = df4_empsequal['diff_mid_emp']/(df4_empsequal[f'emp_center0_avgemp{sfx}'] ** pwr)
#         res = stats.anderson(vseries.to_numpy(), 'norm')  # ,stats.norm.cdf)
#
#         temp_stat, temp_p = stats.jarque_bera(vseries.to_numpy())
#         # sw_stat, sw_p=stats.shapiro(df4_emp2[f'emp_center0_{vname}{sfx}'].to_numpy())
#         printlater.append(
#             f"{labcol} normality test anderson={res.statistic}, jb={temp_stat}, jb_pvalue={temp_p}")  # ,stats.norm.cdf)}")
#         plot_normality(vseries,
#                        filename="DataDiag/param_model_select/emp2_noise_plots/empsequal_center0_avgemp" + sfx +'_'+str(pwrstem)+ "_normality.png",
#                        bins=nbins,
#                        title_stem="Month 2 Employment Equal >0 Subset: Difference from Midpoint over " + labcol,
#                        figsize=(8,3))

for lnpr in printlater:
    print(lnpr)

df4_emp2_all[f'emp_emp3_over_absdiff_p_avgemp'] = df4_emp2_all['diff_3_2_emp']/np.sqrt(df4_emp2_all['abs_diff']+df4_emp2_all['avg_first_last_emp'])
df4_emp2_all[f'emp_emp3_over_absdiff_p_avgemp_p1'] = df4_emp2_all['diff_3_2_emp']/np.sqrt(1+df4_emp2_all['abs_diff']+df4_emp2_all['avg_first_last_emp'])
df4_emp2_all[f'emp_emp3_over_absdiff_p_avgemp_2'] = df4_emp2_all['diff_3_2_emp']/(df4_emp2_all['abs_diff']+df4_emp2_all['avg_first_last_emp'])
df4_emp2_all[f'emp_emp3_over_absdiff_p_avgemp_p1_2'] = df4_emp2_all['diff_3_2_emp']/(1+df4_emp2_all['abs_diff']+df4_emp2_all['avg_first_last_emp'])
df4_emp2_all[f'emp_emp3_over_absdiff_p1'] = df4_emp2_all['diff_3_2_emp']/np.sqrt(1+df4_emp2_all['abs_diff'])
df4_emp2_all[f'emp_emp3_over_absdiff_p1_2'] = df4_emp2_all['diff_3_2_emp']/(1+df4_emp2_all['abs_diff'])
df4_emp2_all[f'emp_emp3_over_avgemp'] = df4_emp2_all['diff_3_2_emp']/np.sqrt(df4_emp2_all['avg_first_last_emp'])
df4_emp2_all[f'emp_emp3_over_avgemp_2'] = df4_emp2_all['diff_3_2_emp']/(df4_emp2_all['avg_first_last_emp'])

plot_normality(df4_emp2_all.loc[df4_emp2_all['avg_first_last_emp']>0,'emp_emp3_over_absdiff_p_avgemp'],
               filename="DataDiag/param_model_select/emp2_noise_plots/all_emp_emp3_over_absdiff_p_avgemp_emp_normality.png", bins=nbins,
               title_stem="Month 2 and 3 Employment Difference over (Month 1 and 3 Abs. Difference + Average)^1/2",
               figsize=(7.5,3))

plot_hist_with_normal(df4_emp2_all.loc[df4_emp2_all['avg_first_last_emp']>0,'emp_emp3_over_absdiff_p1'],
                      filename="DataDiag/param_model_select/emp2_noise_plots/all_emp_emp3_over_absdiff_p1_emp_normhist.png",
                      bins=nbins,title_stem="Over (Month 1 and 3 Difference + 1)^1/2",
                      figsize=(5,3.5))
plot_normality(df4_emp2_all.loc[df4_emp2_all['avg_first_last_emp']>0,'emp_emp3_over_absdiff_p_avgemp_p1'],
               filename="DataDiag/param_model_select/emp2_noise_plots/all_emp_emp3_over_absdiff_p_avgemp_p1_emp_normality.png", bins=nbins,
               title_stem="Month 2 and 3 Employment Difference over (1+Month 1 and 3 Abs. Difference + Average)^1/2",
               figsize=(7.5,3))

plot_hist_with_normal(df4_emp2_all.loc[df4_emp2_all['avg_first_last_emp']>0,'emp_emp3_over_absdiff_p_avgemp'],
                      filename="DataDiag/param_model_select/emp2_noise_plots/all_emp_emp3_over_absdiff_p_avgemp_emp_normhist.png",
                      bins=nbins,title_stem="Over (Month 1 and 3 Difference + Average)^1/2",
                      figsize=(5,3.5))

plot_hist_with_normal(df4_emp2_all.loc[df4_emp2_all['avg_first_last_emp']>0,'emp_emp3_over_absdiff_p_avgemp_2'],
                      filename="DataDiag/param_model_select/emp2_noise_plots/all_emp_emp3_over_absdiff_p_avgemp_2_emp_normhist.png",
                      bins=nbins,title_stem="Over (Month 1 and 3 Difference + Average)",
                      figsize=(5,3.5))

plot_hist_with_normal(df4_emp2_all.loc[df4_emp2_all['avg_first_last_emp']>0,'emp_emp3_over_avgemp'],
                      filename="DataDiag/param_model_select/emp2_noise_plots/all_emp_emp3_over_avgemp_emp_normhist.png",
                      bins=nbins,title_stem="Over (Month 1 and 3 Average)^1/2",
                      figsize=(5,3.5))

plot_hist_with_normal(df4_emp2_all.loc[df4_emp2_all['avg_first_last_emp']>0,'emp_emp3_over_avgemp_2'],
                      filename="DataDiag/param_model_select/emp2_noise_plots/all_emp_emp3_over_avgemp_2_emp_normhist.png",
                      bins=nbins,title_stem="Over (Month 1 and 3 Average)",
                      figsize=(5,3.5))

plot_hist_with_normal(df4_emp2_all.loc[df4_emp2_all['avg_first_last_emp']>0,'emp_emp3_over_absdiff_p1_2'],
                      filename="DataDiag/param_model_select/emp2_noise_plots/all_emp_emp3_over_absdiff_p1_2_emp_normhist.png",
                      bins=nbins,title_stem="Over (Month 1 and 3 Difference + 1)",
                      figsize=(5,3.5))

plot_normality(df4_emp2_all.loc[df4_emp2_all['avg_first_last_emp']>0,'emp_emp3_over_absdiff_p_avgemp_2'],
               filename="DataDiag/param_model_select/emp2_noise_plots/all_emp_emp3_over_absdiff_p_avgemp_2_emp_normality.png", bins=nbins,
               title_stem="Month 2 and 3 Employment Difference over (Month 1 and 3 Abs. Difference + Average)",
               figsize=(7.5,3))

plot_normality(df4_emp2_all.loc[df4_emp2_all['avg_first_last_emp']>0,'emp_emp3_over_absdiff_p_avgemp_p1_2'],
               filename="DataDiag/param_model_select/emp2_noise_plots/all_emp_emp3_over_absdiff_p_avgemp_p1_2_emp_normality.png", bins=nbins,
               title_stem="Month 2 and 3 Employment Difference over (1+Month 1 and 3 Abs. Difference + Average)",
               figsize=(7.5,3))

for vname in ['emp_emp3_over_absdiff_p_avgemp', 'emp_emp3_over_absdiff_p_avgemp_p1', 'emp_emp3_over_absdiff_p_avgemp_2','emp_emp3_over_absdiff_p_avgemp_p1_2']:
    temp_stat, temp_p = stats.jarque_bera(df4_emp2_all.loc[df4_emp2_all['avg_first_last_emp']>0,vname].to_numpy())
    print(f"{vname}: stat={temp_stat}, p={temp_p}")


## Plots are less useful with the smaller array
# noise_coefs = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20, 25, 30, 35, 500.0]
#
# ## - For looking at general trends with the plots
# ## - most of the metrics (mean abs diff, mean sq diff, std sq/abs diff,...)grow linearly
# ## - Will take much longer to run
# # noise_coefs = np.arange(00, 500, 5).tolist()
# results = {}
#
# for coef in noise_coefs:
#     empMat = get_employmentCounts4(
#         df4,
#         m1emp_model=m1empfit,
#         m2emp_noisecoef=coef,
#         rseed=1,
#         include_m1emp_indicator=True
#     )
#
#     imputed_rows = empMat[empMat['emp1_source'] == "model"].copy()
#
#     imputed_rows['abs_diff_m1m2'] = (imputed_rows['emp2'] - imputed_rows['emp1']).abs()
#     imputed_rows['abs_diff_m1m3'] = (imputed_rows['emp3'] - imputed_rows['emp1']).abs()
#     imputed_rows['sq_diff_m1m2'] = (imputed_rows['emp2'] - imputed_rows['emp1']) ** 2
#
#     results[f"coef_{coef}"] = imputed_rows
#
# #comparison = pd.DataFrame({
# #    'noise_coef': noise_coefs,
# #    'mean_emp1': [results[f"coef_{c}"]['emp1'].mean() for c in noise_coefs],
# #    'mean_emp2': [results[f"coef_{c}"]['emp2'].mean() for c in noise_coefs],
# #    'std_emp2': [results[f"coef_{c}"]['emp2'].std() for c in noise_coefs],
# #    'mean_abs_diff_m1m2': [results[f"coef_{c}"]['abs_diff_m1m2'].mean() for c in noise_coefs],
# #    'ms_diff_m1m2': [results[f"coef_{c}"]['sq_diff_m1m2'].mean() for c in noise_coefs],
# #    'std_abs_diff_m1m2': [results[f"coef_{c}"]['abs_diff_m1m2'].std() for c in noise_coefs],
# #    'std_sq_diff_m1m2': [results[f"coef_{c}"]['sq_diff_m1m2'].std() for c in noise_coefs],
# #})
#
# #pd.set_option('display.float_format', lambda x: '%.2f' % x)
# #print("\nComparison across noise coefficients (only imputed rows):")
# #print(comparison)