
import yaml
import os
import pandas as pd

def check_dir(dirname):
    if os.path.exists(dirname):
        if os.path.isdir(dirname):
            pass  # good
        else:
            raise Exception(f"Error: file {dirname} should be a directory.")
    else:
        os.makedirs(dirname)


def check_naics6_variables(naics6df,config,naics6file):
    print("Checking format of "+str(naics6file))
    ## Minimum variables
    needvars=['estnum','wages','emp1','emp3',"geoindkey","state","cnty"]

    naics6vars = naics6df.columns.tolist()
    print(naics6vars)
    saveindicator=False
    if set(['state','cnty']).issubset(set(naics6vars)) or set(['fipstate','fipscty']).issubset(set(naics6vars)):
        if set(['fipstate','fipscty']).issubset(set(naics6vars)):
            print("transforming fipstate and fipscty columns to state and cnty...")
            naics6df.rename(columns={"fipstate": "state",
                                      "fipscty": "cnty"}, inplace=True)

        if "geoindkey" not in naics6vars:
            print("transforming state and cnty columns to create geography column to make geoindkey...")
            naics6df[['state', 'cnty']] = naics6df[['state', 'cnty']].astype(str)
            naics6df['cnty'] = naics6df['cnty'].str.zfill(3)
            naics6df['geography']=naics6df['state'] + naics6df['cnty']
            saveindicator=True
    if "naics" in naics6vars and "geoindkey" not in naics6vars:
        print("transforming naics column to create geoindkey...")
        naics6df['industry']=str(naics6['naics'])
        naics6df['geoindkey'] = naics6df['geography'] + "_" + naics6df['industry']
        saveindicator=True
    elif "industry" in naics6vars and "geoindkey" not in naics6vars:
        print("transforming industry column to create geoindkey...")
        naics6df['geoindkey'] = naics6df['geography'] + "_" + str(naics6df['industry'])
        saveindicator=True
    naics6vars=naics6df.columns.tolist()
    if set(needvars).issubset(set(naics6vars)):
        if "emp2" in naics6vars or isinstance(config["EMP2_NOISECOEF"],float):
            check_microdataConfig(config,True)
            if saveindicator:
                naics6df.to_csv(str(naics6file), sep=',', index=False)
        else:
            raise Exception(f"Error: either microdataConfig missing a numeric EMP2_NOISE or file "+str(naics6file)+" missing a emp2 column.")
    else:
        raise Exception(f"Error: file "+str(naics6file)+" does not have required column names: estnum, wages, emp1, emp3, geoindkey, state, cnty.")


def check_microdataConfig(config,m2emp_indicator=False):
    numericinputs=["GAM_SHAPE","GAM_SCALE","WAGE_MIN"]
    for inputval in numericinputs:
        assert inputval in config and isinstance(config[inputval], (int,float)), f"mircodataConfig is missing numeric "+str(inputval)+"."
    if m2emp_indicator:
        pass
    else:
        assert "EMP2_NOISECOEF" in config and isinstance(config["EMP2_NOISECOEF"],
                                                 (int, float)), f"mircodataConfig is missing numeric EMP2_NOISECOEF."
    intornone=["EST_SEED","NUMCHUNK"]
    for inputval in intornone:
        assert inputval in config and isinstance(config[inputval], (int,None)), f"microdataConfig "+str(inputval)+" must be integer or None."
    assert "OUTPATH" in config, f"microdataConfig missing OUTPATH."
    check_dir(config["OUTPATH"])
    assert "SUBSET_OUTPATH" in config, f"microdataConfig missing SUBSET_OUTPATH."
    check_dir(config["SUBSET_OUTPATH"])
#    assert "CROSSWALK" in config and os.path.exists(config["CROSSWALK"]) and os.path.isfile(config["CROSSWALK"]), f"microdataConfig missing CROSSWALK file or file cannot be located."
    #assert "FIPS_STATE_FILE" in config and os.path.exists(config["FIPS_STATE_FILE"]) and os.path.isfile(
    #    config["FIPS_STATE_FILE"]), f"microdataConfig missing FIPS_STATE_FILE file or file cannot be located."



def check_config(config_file):
    with open(config_file, 'r') as configFile:
        config = yaml.safe_load(configFile)


    #getNAICS6Config=None
    wageConfig=None
    employmentConfig=None
    preprocessConfig=None
    supplementConfig=None
    quarterConfig=None

    assert "microdataConfig" in config and isinstance(config['microdataConfig'],dict), f"Config file is missing microdataConfig."
    assert "generalConfig" in config and isinstance(config['generalConfig'],dict), f"Config file is missing generalConfig."
    generalConfig = config['generalConfig']
    microdataConfig = config['microdataConfig']
    generalConfig['FIPS_STATE_FILE']="DataDiag/DataIn/FIPSstatecodename.txt"
    if 'otherdataConfig' in config:
        supplementConfig=config['otherdataConfig']
    if 'quarterConfig' in config:
        quarterConfig=config['quarterConfig']

    for val in ["YEAR","QTR","SEED"]:
        assert val in generalConfig and isinstance(generalConfig[val],int), f"generalConfig missing integer-valued "+str(val)+"."
    if "SKIP_TO_MICRODATA" in generalConfig and generalConfig["SKIP_TO_MICRODATA"]:
        assert "NAICS6_FILE" in generalConfig, f"generalConfig must have a NAICS6_FILE if SKIP_TO_MICRODATA is true."
        naics6file=generalConfig["NAICS6_FILE"]
        if os.path.exists(naics6file):
            if os.path.isfile(naics6file):
                naics6df = pd.read_csv(str(naics6file))  # , nrows=10000)
                check_naics6_variables(naics6df=naics6df,config=microdataConfig,naics6file=naics6file)
            else:
                raise Exception(f"Error: file {naics6file} should be a csv file.")
        else:
            raise Exception(f"Error:  cannot locate {naics6file}.")
    else:
        check_microdataConfig(microdataConfig, True)
        assert "employmentConfig" in config and isinstance(config['employmentConfig'],dict), f"Config file is missing employmentConfig."
        #assert "getNAICS6Config" in config and isinstance(config['getNAICS6Config'],dict), f"Config file is missing getNAICS6Config."

        #getNAICS6Config = config['getNAICS6Config']
        #getNAICS6Config['FUNCTIONSDIR']="./NAICS6_Pyfunctions/"
        employmentConfig = config['employmentConfig']
        if 'preprocessConfig' in config and isinstance(config['preprocessConfig'],dict):
            preprocessConfig=config['preprocessConfig']
            if generalConfig["API_KEY"] is None:
                neededin = ['DATA_IN_FOLDER', "QWIDIR"]
                assert 'DATA_IN_FOLDER' in preprocessConfig and os.path.exists(preprocessConfig["DATA_IN_FOLDER"]) and os.path.isdir(preprocessConfig["DATA_IN_FOLDER"]), f"preprocessConfig DATA_IN_FOLDER must be a directory."
                assert 'QWIDIR' in preprocessConfig and os.path.exists(preprocessConfig["DATA_IN_FOLDER"]+preprocessConfig["QWIDIR"]) and os.path.isdir(preprocessConfig["DATA_IN_FOLDER"]+preprocessConfig["QWIDIR"]), f"preprocessConfig DATA_IN_FOLDER/QWIDIR must be a directory."
                assert "CBPDATA" in preprocessConfig and os.path.exists(preprocessConfig["DATA_IN_FOLDER"]+preprocessConfig["CBPDATA"]) and os.path.isfile(
                    preprocessConfig["DATA_IN_FOLDER"]+preprocessConfig["CBPDATA"]), f"preprocessConfig DATA_IN_FOLDER/CBPDATA input must be a file."
            assert "OUTPATH" in preprocessConfig, f"OUTPATH must be in preprocessConfig."
            check_dir(preprocessConfig["OUTPATH"])
        else:
            assert "COMBINED_DATASET" in generalConfig and os.path.exists(generalConfig['COMBINED_DATASET']) and os.path.isfile(generalConfig['COMBINED_DATASET']), f"Need a csv file to exist at "+str(generalConfig['COMBINED_DATASET'])+" specified in generalConfig or need preprocessConfig inputs in config file."
        if generalConfig["YEAR"]<2017:
            assert "IMPUTECBP" in preprocessConfig and os.path.exists(preprocessConfig["DATA_IN_FOLDER"]+preprocessConfig["IMPUTECBP"]) and os.path.isfile(
                preprocessConfig["DATA_IN_FOLDER"]+preprocessConfig["IMPUTECBP"]), f"preprocessConfig DATA_IN_FOLDER/IMPUTECBP input must be a file."
            assert "wageConfig" in config and isinstance(config['wageConfig'],
                                                              dict), f"Config file is missing wageConfig."
            wageConfig=config['wageConfig']
        elif "wageConfig"in config and isinstance(config['wageConfig'],dict):
            wageConfig = config['wageConfig']


    return (generalConfig, microdataConfig, preprocessConfig, employmentConfig, wageConfig,supplementConfig, quarterConfig)

        #check NAICS6 by county file exists


# # #testing code
# # with open("../config_pre2017.yaml", 'r') as configFile:
# #     config = yaml.safe_load(configFile)
# # generalConfig = config['generalConfig']
# # getNAICS6Config = config['getNAICS6Config']
# # employmentConfig = config['employmentConfig']
# # microdataConfig = config['microdataConfig']
# # print(type(microdataConfig))
# # print(microdataConfig)
# # testdf = pd.DataFrame(data={#"geoindkey":['1001_202201','2003_202301'],
# #                             "state":[1,1],"cnty":[1,1],"industry":['202201','202301'],
# #                           "geo4naics":["1001_2022","2003_2023"],"estnum":[64,25],"m1emp":[150,20],"m3emp":[65,25],"wage":[400000,100000]})
# # tempgen,tempmirco,temppp,tempn6,tempemp,tempwage= check_config("../config_post2017.yaml")
# # print(tempwage)
# # # #print(temp.head())
