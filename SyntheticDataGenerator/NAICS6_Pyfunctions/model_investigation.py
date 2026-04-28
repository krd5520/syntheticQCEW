import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import statsmodels.api as sm
import sys
import os
from stargazer.stargazer import Stargazer, LineLocation
import importlib
import random

sys.path.append(os.path.abspath("./NAICS6_Pyfunctions/"))
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


## In a poor judgement move, I started by bopying and pasting the following code for each set of models to investigate
## To save time, the variable names are not changed from the most recent application and are not as informative or general
## Thus function inputs are renames to the less logical version immediately in this function.
def compare_models(data,modelsdict,config,foldername=None,tablefname=None,csvfname=None,diagplotstem=None,wages=False):
    ##rename variable inputs to match the code
    employmentConfig = config
    df4=data

    model_dict_empdiff1=modelsdict

    ## Initialize stuff
    modoutdict = {}
    responsestr = []
    modelstr = []
    modfits = []
    adjrsquared = []
    outliertexts = []
    num_cooksd = []
    num_studres = []
    vifstr = []
    num_vifs = []
    ## For each in models dictionary
    for key, value in model_dict_empdiff1.items():
        employmentConfig['OLS_FORMULA'] = value[1] #change model
        employmentConfig[
            'DIAGNOSTIC_PLOTS'] = foldername+"/"+diagplotstem+ re.sub(
            "000", "_", re.sub(r'[^A-z0-9]+', "", re.sub(r"\s+", "000", key))) + ".png"

        #fit model
        if wages:
            modout=get_wages_model(df4, employmentConfig, return_text=True, modelname=value[0], quietly=True)
        else:
            modout = get_m1emp_model(df4, employmentConfig, return_text=True, modelname=value[0], quietly=True)
        if isinstance(modout, tuple): #if modout is a tuple, split into fit and outtext
            modfit, outtext = modout
            if len(outtext) == 4: #split outtext into componenets
                [cooksdlead, cooksd_idx, stud_restext, viftext] = outtext
                ncooksout = cooksd_idx.count(",")
            else:
                [cooksdtext, stud_restext, viftext] = outtext
                ncooksout = cooksdtext[13]
            nstudresout = stud_restext.split(":")[-1]
            linesplit_vif = viftext.splitlines()
            if len(linesplit_vif) > 1: #reformat VIF
                var_vif = linesplit_vif[3:]
                var_vif_noindex = []
                for line in var_vif:
                    if "C(" in line:
                        vifvarname = line.replace(")", "(").split("(")[1].strip()
                    else:
                        vifvarname = line.split()[1]
                    vifval = round(float(line.strip().split()[-1].strip()), 1)
                    var_vif_noindex = var_vif_noindex + [vifvarname + " (" + str(vifval) + ")"]
                try:
                    var_vifstr = "; ".join(var_vif_noindex)
                except:
                    print(var_vif_noindex)
                    var_vifstr = var_vif_noindex
            else:
                var_vifstr = "None with VIF>5"
            modoutdict[value[0]] = [key, value[1], modfit, modfit.rsquared_adj, outtext, ncooksout, nstudresout,
                                    var_vifstr]
            num_cooksd = num_cooksd + [ncooksout]
            num_studres = num_studres + [nstudresout]

            vifstr = vifstr + [var_vifstr]
            num_vifs = num_vifs + [vifstr.count('\n') + 1]
            adjrsquared = adjrsquared + [modfit.rsquared_adj]
            modfits = modfits + [modfit]
            modelstr = modelstr + [value[1]]
            responsestr = responsestr + [value[1].split("~")[0].strip()]
        else: #if not tuple
            modfit = modout
            modoutdict[value[0]] = [key, value[1], modfit, modfit.rsquared_adj]
            adjrsquared = adjrsquared + [modfit.rsquared_adj]
            modfits = modfits + [modfit]
            modelstr = modelstr + [value[1]]
            responsestr = responsestr + [value[1].split("~")[0].strip()]

    # #get list of predictors for ordering predictors later
    # listpredictors = []
    # for key, values in modoutdict.items():
    #     model = modoutdict[key][2]
    #     predictors = model.model.exog_names
    #     if listpredictors is None:
    #         listpredictors = [x for x in predictors]
    #     else:
    #         listpredictors = listpredictors + [x for x in predictors]
    # #get frequency of each predictor term across the models
    # freqs={}
    # for itm in listpredictors:
    #     freqs[itm]=freqs.get(itm,0)+1
    # orderedpreds = dict(sorted(freqs.items(),reverse=True)) #sort by frequency
    # listpredictors = list(dict.fromkeys(listpredictors))
    #
    # print(orderedpreds)
    # orderedbasepreds=[]
    # fullordered=[]
    # for key,value in orderedpreds.items():
    #     basekey=key
    #     if ")" in key:
    #         basekey = basekey.split(')', 1)[0].split("(")[-1]
    #         if "," in basekey:
    #             basekey = basekey.split(',')[0]
    #     basekey=basekey.strip()
    #     if basekey not in orderedbasepreds:
    #         orderedbasepreds=orderedbasepreds+[basekey]
    #         fullordered=fullordered+sorted([x for x in listpredictors if basekey in x])
    # fullordered=["Intercept"]+fullordered
    fullordered=predictor_order_freq(modoutdict)#list(dict.fromkeys(fullordered))
    print(fullordered)
    tabstar = Stargazer([modoutdict[key][2] for key in modoutdict.keys()])
    fname = foldername+"/"+tablefname
    tabstar.custom_columns([key for key in modoutdict], [1] * len(modoutdict.keys()))
    tabstar.significant_digits(2)
    tabstar.covariate_order(fullordered)
    tabstar.rename_covariates(stargazer_covariate_renamer(fullordered))
    tabstar.add_line('Studentized Residuals Outliers', num_studres, LineLocation.FOOTER_BOTTOM)
    tabstar.add_line("Cook's Distance Outliers", num_cooksd, LineLocation.FOOTER_BOTTOM)
    tabtex = tabstar.render_latex()
    with open(fname, 'w') as f:
        f.write(tabtex)

    compmods = pd.DataFrame({"model_id": model_dict_empdiff1.keys(),
                             "response": responsestr,
                             "formula": modelstr,
                             "adjRsquared": adjrsquared,
                             "num_influential": num_cooksd,
                             "num_studentized_residual_outliers": num_studres,
                             "num_high_vif": num_vifs,
                             "high_vif": vifstr})
    compmods.set_index("model_id")

    # print(modoutdict)
    compmods.to_csv(foldername+"/"+csvfname)
    print(compmods)

