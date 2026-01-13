import time

import pandas as pd
import sys
import os
import yaml
import random

sys.path.append(os.path.abspath("./NAICS6_Pyfunctions/"))
from EmploymentFunctions import *
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

## hardcoded variable (shouldn't change?)
naics_file_sep="," #sep for generalConfig["NAICS_FILE"]
naics_file_code_col=1 #column corresponding to naics codes at all levels in generalConfig["NAICS_FILE"]



def start(config):
    readinData=0
    createCombined=0
    NAICS6time=0
    generalConfig, microdataConfig, preprocessConfig, employmentConfig, wageConfig,supplementaryConfig = check_config(config)
    startime=time.time()
    print('---------- -------- -------- -------- ----------')
    print('---------- Retrieving or Loading Data ----------')
    print('---------- -------- -------- -------- ----------')
    if generalConfig["NAICS_FILE"] is not None:
        naicsdf = process_naics_file(generalConfig["NAICS_FILE"], code_col=naics_file_code_col, code_sep=naics_file_sep)
    else:
        naicsdf = None
    if generalConfig["SKIP_TO_MICRODATA"] is not None and generalConfig["SKIP_TO_MICRODATA"]:
        print('Reading in NAICS6 by County data from '+str(generalConfig["NAICS6_FILE"]))
        naics6df = pd.read_csv(str(generalConfig["NAICS6_FILE"]))  # , nrows=10000)
        readinData=time.time()-startime
    else:
        if os.path.exists(generalConfig['COMBINED_DATA']) and os.path.isfile(generalConfig['COMBINED_DATA']):
            print("Reading combined dataset from "+str(generalConfig['COMBINED_DATA']))
            df=pd.read_csv(generalConfig['COMBINED_DATA'],
                           usecols=['agglvl_code', 'lwbd_emp_qwi', 'avg_month_emp_wages', 'estnum',
                                    'emp3', 'emp2', 'emp1', 'wages',"year","qtr", 'emp1_source',"emp2_source","emp3_source",
                                    "wages_source","geoindkey","industry","state","cnty","naics2","naics3","naics4","naics5",
                                    "domain","supersector","wages_cbp_flag","emp3_cbp_flag","emp3_qwi_flag",'row_sources'],
                           low_memory=True
                           )
            readinData=time.time()-startime
        elif preprocessConfig is not None:
            print("Combining CBP and QWI datasets.")
            foldername = preprocessConfig['DATA_IN_FOLDER']
            if generalConfig['API_KEY'] is not None:
                print("Downloading Datasets")
                fipscodes_df=pd.read_csv(generalConfig["FIPS_STATE_FILE"])
                if generalConfig["STATES"] is None or generalConfig["STATES"]==["ALL"]:
                    states=[]
                else:
                    states=generalConfig["STATES"]
                check_dir(foldername + preprocessConfig['QWIDIR'])
                if os.listdir(foldername + preprocessConfig['QWIDIR'])==[]:
                    download_QWI(fipscodes_df=fipscodes_df, preprocessConfig=preprocessConfig,
                                           generalConfig=generalConfig, max_retries=3,
                                           savechunks=3)

                if not os.path.exists(foldername + preprocessConfig['CBPDATA']):
                    download_rawCBP(fipscodes_df=fipscodes_df, preprocessConfig=preprocessConfig,
                                              generalConfig=generalConfig, max_retries=3)
                if "QCEWDIR" in preprocessConfig and preprocessConfig["QCEWDIR"] is not None:
                    check_dir(foldername + preprocessConfig['QCEWDIR'])
                    if os.listdir(foldername + preprocessConfig['QCEWDIR']) == []:
                        qcew=download_QCEW(generalConfig=generalConfig, preprocessConfig=preprocessConfig, savechunks=1,
                                      forcombine=True)
                        #print("download qcew")
                    if os.listdir(foldername+preprocessConfig['QCEWDIR'])==[]:
                        #print("qcew didn't save properly???")
                        qcew.to_csv(foldername+preprocessConfig['QCEWDIR'])
                readinData = time.time() - startime
            else:
                readinData=""

            createcombinedstart=time.time()
            df=combine_qwi_cbp_qcew(rawfile=foldername + preprocessConfig['CBPDATA'],
                               imputedfile=foldername + preprocessConfig['IMPUTECBP'],
                               qwifolder=foldername + preprocessConfig['QWIDIR'],
                               outfilename=generalConfig['COMBINED_DATA'],
                                    diagnosticsfile=preprocessConfig["DIAGNOSTIC_FILE"],
                                    generalConfig=generalConfig,
                                    preprocessConfig=preprocessConfig,
                               outfilepath=preprocessConfig['OUTPATH'],
                               year=generalConfig['YEAR'],
                                    naicsdf=naicsdf)
            master_colsave=['agglvl_code','emp3_cbp','wages_cbp','estnum_cbp',
                 'emp1_qwi', 'emp3_qwi', 'lwbd_emp_qwi', 'avg_month_emp_wages', 'estnum',
                 'emp3', 'emp2', 'emp1', 'wages', 'year_qtr',"year","qtr","estnum",
                   'emp1_source',"emp2_source","emp3_source","wages_source",
                   "geoindkey","industry","state","cnty","naics2","naics3","naics4","naics5",
                   "emp1_qwi","emp1_qcew","wages_cbp_flag","emp3_cbp_flag","emp3_qwi_flag","row_sources","geo2naics","geo3naics","geo4naics","geo5naics"]
            if "supersector" in df.columns:
                colssave=master_colsave+["domain","supersector"]
            else:
                colssave=master_colsave

            df=df[colssave].copy()
            print("Combined file is saved: "+str(generalConfig['COMBINED_DATA']))
            createCombined=time.time()-createcombinedstart
            df = pd.read_csv(generalConfig['COMBINED_DATA'],
                             usecols=colssave,low_memory=True
                             )

        #df['year_qtr']=df['year'].astype(float)+(df["qtr"].astype(float)/4)

        tofloatcols = ['emp3_cbp', 'wages_cbp', 'estnum_cbp', 'year_qtr_cbp',
                 'emp1_qwi', 'emp3_qwi', 'lwbd_emp_qwi', 'avg_month_emp_wages', 'estnum',
                 'emp3', 'emp2', 'emp1', 'wages', 'year_qtr']
        for x in tofloatcols:
            if x in df.columns:
                df[x] = df[x].astype(float)

        #df.dropna(axis=0,how='any',subset='emp3',inplace=True) #remove ones missing emp3

        #print(f"NA count per column, out of {df.shape}")
        #print(df.isna().sum())
        #for cname in [x for x in df.columns if "_source" in x]:
        #    print(df[cname].value_counts(dropna=False))

        #print("after drop na")
        #print(pd.crosstab(df["emp1_source"],df["wages_source"],dropna=False))
        #print(pd.crosstab(df['emp1_source'],df['agglvl_code'],dropna=False))

        print(f'---------- Combined Data Done: Time {createCombined} ----------')
        print('---------- -------- -------- -------- ----------\n')

        print('\n---------- -------- -------- -------- ----------')
        print('---------- Getting Complete County by NAICS-6 Data ----------')
        print('---------- -------- -------- -------- ----------\n')
        startNAICS6=time.time()
        naics6df=generate_NAICS6_byCounty(generalConfig, employmentConfig, wageConfig,supplementaryConfig=supplementaryConfig, df=df,naicsdf=naicsdf)
        NAICS6time=time.time()-startNAICS6
        print(f'---------- Complete County by NAICS-6 Done: Time {NAICS6time} ----------')
        print('---------- -------- -------- -------- ----------\n')

    print('\n---------- -------- -------- -------- ----------')
    print('---------- Making Synthetic Microdata ----------')
    print('This may take a while, please be patient...')
    print('---------- -------- -------- -------- ----------\n')
    print('---------- Microdata Configuration ----------')
    startmicrodata=time.asctime()
    for key, value in microdataConfig.items():
        print(f"{key}: {value}")

    random.seed(microdataConfig['EST_SEED'])
    tempdf = make_syn_microdata(naics6df, numchunk=microdataConfig['NUMCHUNK'],
                                outfoldername=microdataConfig['OUTPATH'])
    microdatatime=time.time()-startmicrodata


    ################## Microdata Postprocessing #######################
    postprocessstart=time.time()
    print('---------- Generating Final Microdata ----------')


    combine_and_split_iterative(yr=generalConfig['YEAR'], qtr=generalConfig['QTR'])
    postprocessstime=time.time()-postprocessstart
    finaltime=time.time()-startime
    print('\n---------- -------- -------- -------- ----------')
    print('---------- -------- -------- -------- ----------\n')
    print("----------- Computation Time Summary ------------")
    print("reading in data (NAICS6 or combined data): "+str(readinData))
    if createCombined is not None:
        print("combining data (NAICS6 or combined data): " + str(createCombined))
    print("creating NAICS6 by County data: "+str(NAICS6time))
    print("creating microdata (in chunks): "+str(microdatatime))
    print("post processing time: "+str(postprocessstime))
    print("total: "+str(finaltime))
    print("----------------- DONE ---------------------------")
    print()

def main():
    """ Validate command line arguments, get config and then call start()"""
    if len(sys.argv) != 2:
        print("Config file not specified")
        print(f"Usage: python {sys.argv[0]} config_file_name")
        return
    with open(sys.argv[1], 'r') as configFile:
        config = yaml.safe_load(configFile)
    start(config)


if __name__ == "__main__":
    #main()
    with open("config_pre2017.yaml", 'r') as configFile:
        config = yaml.safe_load(configFile)

    start("config_pre2017.yaml")