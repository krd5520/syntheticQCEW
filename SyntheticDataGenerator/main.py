import time

import pandas as pd
import sys
import os
import yaml
import random

sys.path.append(os.path.abspath("./NAICS6_Pyfunctions/"))
from getAggLevelSummaries import *
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

redownload=False


def start(config):
    readinData=0
    createCombined=0
    NAICS6time=0
    generalConfig, microdataConfig, preprocessConfig, employmentConfig, wageConfig = check_config(config)
    startime=time.time()
    if generalConfig["SKIP_TO_MICRODATA"] is not None and generalConfig["SKIP_TO_MICRODATA"]:
        print('Reading in NAICS6 by County data from '+str(generalConfig["NAICS6_FILE"]))
        naics6df = pd.read_csv(str(generalConfig["NAICS6_FILE"]))  # , nrows=10000)
        readinData=time.time()-startime
    else:
        if os.path.exists(generalConfig['COMBINED_DATA']) and os.path.isfile(generalConfig['COMBINED_DATA']):
            print("Reading combined dataset from "+str(generalConfig['COMBINED_DATA']))
            df=pd.read_csv(generalConfig['COMBINED_DATA'],dtype=str)
            readinData=time.time()-startime
            print(df.head())
            print(df.columns)
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
                        print("download qcew")
                    if os.listdir(foldername+preprocessConfig['QCEWDIR'])==[]:
                        print("acew didn't save properly???")
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
                               year=generalConfig['YEAR'])
            print("Combined file is saved: "+str(generalConfig['COMBINED_DATA']))
            createCombined=time.time()-createcombinedstart
        print(df.head())
        print(df.columns)
        startNAICS6=time.time()
        naics6df=generate_NAICS6_byCounty(generalConfig, employmentConfig, wageConfig, df=df)
        NAICS6time=time.time()-startNAICS6
    print('---------- Microdata Configuration ----------')
    startmicrodata=time.asctime()
    for key, value in microdataConfig.items():
        print(f"{key}: {value}")
    print('---------- Making Synthetic Microdata ----------')
    print('This may take a while, please be patient...')
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