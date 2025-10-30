import pandas as pd
import os
import re
import yaml


import urllib.request
import urllib


pd.set_option("display.max_columns", None)

#subset fipscodes_df to only requested states and process savechunks into num_splits
def state_selector(fipscodes_df,states,savechunks):
    if savechunks is None:
        savechunks=1
    if states is not None and len(states)>0: #if states are inputted
        if isinstance(states,str) and not isinstance(states,list): #if STATES is a string
            states=list(states.upper()) #standardize format
            if states=="ALL": #if ALL, do not subset
                return fipscodes_df,savechunks
        try: #if iterable, standardize format
            states=[str(x).upper() for x in states]
        except:
            print('STATES must be iterable object. It was '+str(type(states)))
        if "ALL" in states: #if ALL do not subset
            return fipscodes_df, savechunks
        fipscode = fipscodes_df.iloc[:, 0].astype(str).str.upper() #standardize format col1
        stateselect=(fipscode.isin(states)) #initialize the logical to select states
        if len(fipscodes_df.columns)>1: #if there are more than 1 column
            for colname in fipscodes_df.columns: #iterate columns to create logical
                fipscode=fipscodes_df[colname].astype(str).str.upper()
                stateselect=(stateselect)|(fipscode.isin(states))
        if len(states)==1: #if only one state, then only 1 chunk to save
            savechunks=1
        fipscodes_df = fipscodes_df[stateselect] #subset
    return fipscodes_df, savechunks

### The qcewCreateDataRows and qcewGetAreaData functions are provided by the BLS
# *******************************************************************************
# qcewCreateDataRows : This function takes a raw csv string and splits it into
# a two-dimensional array containing the data and the header row of the csv file
# a try/except block is used to handle for both binary and char encoding
def qcewCreateDataRows(csv):
    dataRows = []
    try: dataLines = csv.decode().split('\r\n')
    except er: dataLines = csv.split('\r\n');
    for row in dataLines:
        dataRows.append(row.split(','))
    return dataRows
# *******************************************************************************




# *******************************************************************************
# qcewGetAreaData : This function takes a year, quarter, and area argument and
# returns an array containing the associated area data. Use 'a' for annual
# averages.
# For all area codes and titles see:
# http://www.bls.gov/cew/doc/titles/area/area_titles.htm
#
def qcewGetAreaData(year,qtr,area):
    urlPath = "http://data.bls.gov/cew/data/api/[YEAR]/[QTR]/area/[AREA].csv"
    urlPath = urlPath.replace("[YEAR]",year)
    urlPath = urlPath.replace("[QTR]",qtr.lower())
    urlPath = urlPath.replace("[AREA]",area.upper())
    httpStream = urllib.request.urlopen(urlPath)
    csv = httpStream.read()
    httpStream.close()
    return qcewCreateDataRows(csv)
# *******************************************************************************
##########


### My function:
## fipsfile is the file path to the full list of fips county codes
## year is the year you want to download, qtr is the quarter you want to download
## state is a list of the state numberic codes to specify which to download. If blank all states are retrieved.
## returns pandas dataframe with columns: area_fips, own_code, industry_code, agglvl_code, year, qtr, disclosure_code,
## qtrly_estabs, month1_emplvl, month2_emplvl, month3_emplvl, total_qtrly_wages
def download_QCEW(generalConfig, preprocessConfig,savechunks=2,forcombine=False):
    assert preprocessConfig['QCEWDIR'] is not None, f'No QCEW data requested in the config file.'
    assert generalConfig['FULL_FIPS_FILE'] is not None and os.path.exists(generalConfig["FULL_FIPS_FILE"]), f"{generalConfig['FULL_FIPS_FILE']} cannot be located or is not a csv file."
    if preprocessConfig["DIAGNOSTIC_FILE"] is not None:
        diagnosticfile=preprocessConfig["DIAGNOSTIC_FILE"]
    else:
        diagnosticfile=None
    fipsdf=pd.read_csv(generalConfig['FULL_FIPS_FILE'])
    state_abbr = {
        "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca", "08": "co", "09": "ct", "10": "de",
        "11": "dc", "12": "fl", "13": "ga", "15": "hi", "16": "id", "17": "il", "18": "in", "19": "ia", "20": "ks",
        "21": "ky", "22": "la", "23": "me", "24": "md", "25": "ma", "26": "mi", "27": "mn", "28": "ms", "29": "mo",
        "30": "mt", "31": "ne", "32": "nv", "33": "nh", "34": "nj", "35": "nm", "36": "ny", "37": "nc", "38": "nd",
        "39": "oh", "40": "ok", "41": "or", "42": "pa", "44": "ri", "45": "sc", "46": "sd", "47": "tn", "48": "tx",
        "49": "ut", "50": "vt", "51": "va", "53": "wa", "54": "wv", "55": "wi", "56": "wy"
    }
    states=list(state_abbr.keys())
    if "STATES" in generalConfig:
        if generalConfig['STATES'] is not None:
            if "ALL" not in generalConfig["STATES"]:
                states=generalConfig["STATES"]


    year=generalConfig["YEAR"]
    qtr=generalConfig['QTR']
    if not os.path.exists(preprocessConfig['QCEWDIR']):
        os.mkdir(preprocessConfig['QCEWDIR'])
    savefile=preprocessConfig['QCEWDIR']+"qcew_part"
    if states is not None and len(states)>0: #if specified certain states, subset fips to relevant ones
        stateselector=pd.Series([False]*len(fipsdf))
        for st in states:
            if st[0].isdigit():
                if st.startswith("0"):
                    stabbr=st[1]
                else:
                    stabbr=st
            else:
                stabbr=[key for key,value in state_abbr.items() if str(value).upper()==str(st).upper()]
                #if str(stabbr[0]).startswith("0"):
                #    stabbr=str(stabbr[0])[1]
            pattern=str(stabbr[0])
            stselector = fipsdf.area_fips.str.startswith(pattern)
            stateselector=stateselector|stselector
        fipsdf=fipsdf[stateselector]
        areadf = combine_states(fipsdf=fipsdf, year=year, qtr=qtr, stateselector=stateselector)
        if savefile is not None:
            if re.match(".csv",savefile):
                savefile=savefile
            else:
                savefile=savefile+".csv"
            areadf.to_csv(savefile, index=False)
    elif savefile is None or savechunks==1:
        stateselector=pd.Series([True]*len(fipsdf))
        areadf = combine_states(fipsdf=fipsdf, year=year, qtr=qtr, stateselector=stateselector)
        if savefile is not None:
            if re.match(".csv",savefile):
                savefile=savefile
            else:
                savefile=savefile+".csv"
            areadf.to_csv(savefile, index=False)

    if savefile is not None and savechunks>1:
        fullfull=None

        start = 0
        chunksize=round(50/savechunks)
        for i in range(1,savechunks+1):
            stop=start+chunksize
            if stop>50:
                stop=51
            stateselector = pd.Series([False] * len(fipsdf))
            states=list(state_abbr.keys())[start:stop]
            for st in states:
                #print(st)
                stselector = fipsdf.area_fips.str.match(str(st) + "[0-9]{3}")
                stateselector = stateselector | stselector
            subfipsdf = fipsdf[stateselector]
            areadf = combine_states(fipsdf=subfipsdf, year=year, qtr=qtr, stateselector=stateselector)
            if forcombine:
                areadf = areadf[areadf.agglvl_code > 70]
                areadf = areadf[areadf.own_code == 5]

            if savefile is not None and areadf is not None:
                print(areadf.shape)
                if fullfull is None:
                    fullfull=areadf
                else:
                    fullfull = pd.concat([fullfull,areadf[1:]],ignore_index=True)

                print("saving "+savefile+str(i)+".csv")
                areadf.to_csv(savefile+str(i)+".csv", index=False)
            start=stop+1
        fullfull.drop_duplicates(inplace=True)
        if diagnosticfile is not None:
            qcew_suppression_tab(fullfull, diagnosticsfile=preprocessConfig["OUTPATH"]+diagnosticfile)
        return fullfull
    if diagnosticfile is not None:
        qcew_suppression_tab(areadf, diagnosticsfile=preprocessConfig["OUTPATH"]+diagnosticfile)
    if forcombine:
        areadf = areadf[areadf.agglvl_code > 70]
        areadf = areadf[areadf.own_code == 5]
        areadf.drop_duplicates(inplace=True)
    return areadf



def combine_states(fipsdf,year,qtr,stateselector):
    #cycle the fips codes
    iter=0
    fulldata=None
    okindic=False
    for fips in fipsdf['area_fips']:
        okindic = True
        #if re.match("[0-9]*999",fips):
        #    pass
        #else:
            #print(fips)
        try:
            areadata=qcewGetAreaData(str(year),str(qtr),fips)
            if iter == 0:
                fulldata = areadata  # keep first row which is column headers
            else:
                fulldata.extend(areadata[1:])
            iter += 1
        except:
            okindic=False
            print("Bad FIPS: "+str(fips))

    if okindic and fulldata is not None:
        # create as pandas daatframe, select the desired columns, and keep only privately owned
        areadf = pd.DataFrame(fulldata[1:], columns=[s.strip('"') for s in fulldata[0]])
        areadf = areadf.iloc[:, :13]
        areadf = areadf.drop('size_code', axis=1)  # remove size_code column
        areadf = areadf[areadf['own_code'] == '"5"']

        # remove unnecessary quotation marks and convert some columns to numeric
        colstostr = ['area_fips', 'industry_code',"disclosure_code"]
        areadf[colstostr] = areadf[colstostr].replace({'"': ''}, regex=True)
        colstoint = ['own_code', 'agglvl_code', 'year', 'qtr']
        areadf[colstoint] = areadf[colstoint].replace({'"': ''}, regex=True).apply(pd.to_numeric, errors='coerce')
        areadf.drop_duplicates(inplace=True)



        #areadf = areadf[areadf['industry_code'].str.contains(
        #    '0|1|2|3|4|5|6|7|8')]  # remove industry codes that are all 9's (unknown/unspecified)

        return (areadf)
    else:
        return None


## Note disclosure code "-" is for cells with no establishments
def qcew_suppression_tab(data,diagnosticsfile=None):
    data=data[data["disclosure_code"].isin(["","N"])]
    data=data[data['agglvl_code']>70]
    supptab = pd.crosstab(data['agglvl_code'], data['disclosure_code']).join(
        pd.crosstab(data['agglvl_code'], data['disclosure_code'], normalize="index"),
        on="agglvl_code", how='outer', lsuffix='_count', rsuffix='_prop', sort=True)
    print(supptab.columns)
    supptabfull = pd.DataFrame(
        { 'n': supptab.sum(axis=1),
         'suppressed': supptab['N_count'], "pct_suppressed": supptab["N_prop"].multiply(100).round(0)})  # .join(supptabemp,on='agglvl_code',how='outer',lsuffix='_wage',rsuffix='_emp')
    if diagnosticsfile is not None and os.path.exists(diagnosticsfile):
        with open(diagnosticsfile, 'a') as f:
            print("--" * 20, file=f)
            print("----- QCEW Suppression Table -----", file=f)
            print("--" * 20, file=f)
        supptabfull.to_csv(diagnosticsfile, sep=",", mode="a")
    elif diagnosticsfile is not None:
        os.makedirs(os.path.dirname(diagnosticsfile),exist_ok=True)
        with open(diagnosticsfile, 'w') as f:
            print("--" * 20, file=f)
            print("----- QCEW Suppression Table -----", file=f)
            print("--" * 20, file=f)
        supptabfull.to_csv(diagnosticsfile, sep=",", mode="a")
    return(supptabfull)




state_abbr = {
            "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca", "08": "co", "09": "ct", "10": "de",
            "11": "dc", "12": "fl", "13": "ga", "15": "hi", "16": "id", "17": "il", "18": "in", "19": "ia", "20": "ks",
            "21": "ky", "22": "la", "23": "me", "24": "md", "25": "ma", "26": "mi", "27": "mn", "28": "ms", "29": "mo",
            "30": "mt", "31": "ne", "32": "nv", "33": "nh", "34": "nj", "35": "nm", "36": "ny", "37": "nc", "38": "nd",
            "39": "oh", "40": "ok", "41": "or", "42": "pa", "44": "ri", "45": "sc", "46": "sd", "47": "tn", "48": "tx",
            "49": "ut", "50": "vt", "51": "va", "53": "wa", "54": "wv", "55": "wi", "56": "wy"
        }
#print(list(state_abbr.keys())[:10])
#states=[str(x).lower() for x in ["AL","PA","NJ","RI","CA","IN","OH","NM"]]

#with open('config_pre2017.yaml','r') as configFile:
#    config = yaml.safe_load(configFile)
#    preprocessConfig = config['preprocessConfig']
#    generalConfig = config['generalConfig']
#temp=download_QCEW(generalConfig=generalConfig,preprocessConfig=preprocessConfig,savechunks=1,forcombine=True)
##temp=pd.read_csv("../DataDiag/DataIn/QCEW/qcew_part.csv")
#temp=temp[temp.agglvl_code>50]
#print(temp.head())
#print(temp.tail())
#print(temp[temp["agglvl_code"=="51"]])
#temp2=pd.read_csv("../DataDiag/DataIn/QCEW/qcew_part2.csv")
#temp=pd.concat([temp1,temp2],ignore_index=True)
#print(temp.head())


