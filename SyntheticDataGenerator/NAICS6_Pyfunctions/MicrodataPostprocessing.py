import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
import yaml
import sys

sys.path.append(os.path.abspath('./'))
from hierarchy_geoindkey import *
from GeneralFunctions import *
# with open('./config.yaml', 'r') as configFile:
#     config = yaml.safe_load(configFile)
#     postprocessingConfig = config['microdataConfig']
#     generalConfig=config['generalConfig']
#






def cut1(naicscode):
    # Cuts 1 character off end of string
    return str(naicscode)[:-1]

def postprocess_est_microdata_split(filename, generalConfig=None, naicsdata=None,xwalk=None):
    """
    Processes a single SynMicrodata file:
    
    Steps:
    1. Reads the CSV file into a DataFrame.
    2. Creates NAICS code:
         - 'naics5' is 'naics6' with the last digit removed.
         - 'naics4' is 'naics5' with the last digit removed.
         - ...
         - 'naics' is simply a copy of 'naics6'.
    3. Merges the DataFrame with the NAICS crosswalk DataFrame (xwalk) on 'naics_sector' 
       to retrieve the 'super_sector' information.
    4. Adds constant columns: 'year', 'qtr', 'own', 'can_agg', and 'rectype'.
    5. Stores 'm1emp', 'm2emp', 'm3emp' as integers.
    6. Reorders the DataFrame columns to the desired order.
    7. Extracts a file number from the filename using a regular expression and adds it as a new column 'filenumber'.
    
    Returns:
        The processed DataFrame for the microdata file.
    """
    print(f"Processing file: {filename}")
    estdf = pd.read_csv(filename)
    if xwalk is not None and ~set(["supersector","naics2"]).issubset(set(xwalk.columns)):
        xwalk['supersector']=xwalk['super_sector']
        xwalk['naics2']=xwalk['naics_sector']
    estdf=fill_from_geoindkey(estdf,numeric_ind_level=True,naics_xwalk=xwalk,naicsdata=naicsdata)


    # Add constant columns
    for varname in ["year","qtr"]:
        if varname not in estdf.columns:
            estdf[varname] = generalConfig[varname.upper()]
    estdf['own'] = 5
    estdf['can_agg'] = 'Y'
    estdf['rectype'] = 'C'
    
    # Employee columns are integers
    estdf['m1emp'] = estdf['emp1'].astype(int)
    estdf['m2emp'] = estdf['emp2'].astype(int)
    estdf['m3emp'] = estdf['emp3'].astype(int)
    estdf['wage'] = estdf['wages'].astype(int)


    if ~set(['naics_sector','super_sector']).issubset(set(estdf.columns)):
        estdf['naics_sector']=estdf['naics2']
        estdf['super_sector']=estdf['supersector']
    if 'naics' not in estdf.columns:
        estdf['naics']=estdf['naics6']
    # Reorder columns
    estdf = estdf[['year', 'qtr', 'state', 'cnty', 'own', 'naics', 
                   'naics3', 'naics4', 'naics5', 'naics_sector', 'super_sector', 
                   'm1emp', 'm2emp', 'm3emp', 'wage', 'can_agg', 'rectype']]
    
    # Gets file number and add as a column
    filenum = re.search(r'\d+', filename).group()
    estdf['filenumber'] = filenum
    
    return estdf

def combine_and_split_iterative(generalConfig, microdataConfig,filebasename="SynMicrodata",naicsdata=None):
    """
    Processes SynMicrodata files one by one, assigns primary keys,
    and writes each state's subset to its corresponding file.
    
    Detailed Steps:
    1. Gather all SynMicrodata files in the input folder.
    2. Retrieve the cleaned and expanded NAICS crosswalk DataFrame.
    3. Initialize a running counter (primary_key_counter) for assigning primary keys across files.
    4. For each file:
         a. Process the file using postprocess_est_microdata_split.
         b. Determine the number of rows and assign primary keys.
         c. Update the primary_key_counter.
         d. For each state (based on state_abbr mapping):
              i. Filter the processed DataFrame for rows matching that state.
              ii. Create the state's output directory (if not already existing).
              iii. Write the state's data to a CSV file:
                   - If the file already exists, append the data without a header.
                   - Otherwise, create a new file with the header.
    5. Print a message when all files have been processed.
    """
    folder = Path(microdataConfig['SUBSET_OUTPATH'])
    outdir = Path(microdataConfig['OUTPATH'])
    # Crosswalk file from
    ## https://www.bls.gov/cew/classifications/industry/industry-supersectors.htm
    crosswalk_file = generalConfig['BLS_NAICS_CROSSWALK']
    #state_file = postprocessingConfig['FIPS_STATE_FILE']

    if naicsdata is None and generalConfig.get["NAICS_FILE"] is not None:
        naicsdata=process_naics_file(generalConfig["NAICS_FILE"], code_col=1, code_sep=",")


    state_abbr = {
        "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca", "08": "co", "09": "ct", "10": "de",
        "11": "dc", "12": "fl", "13": "ga", "15": "hi", "16": "id", "17": "il", "18": "in", "19": "ia", "20": "ks",
        "21": "ky", "22": "la", "23": "me", "24": "md", "25": "ma", "26": "mi", "27": "mn", "28": "ms", "29": "mo",
        "30": "mt", "31": "ne", "32": "nv", "33": "nh", "34": "nj", "35": "nm", "36": "ny", "37": "nc", "38": "nd",
        "39": "oh", "40": "ok", "41": "or", "42": "pa", "44": "ri", "45": "sc", "46": "sd", "47": "tn", "48": "tx",
        "49": "ut", "50": "vt", "51": "va", "53": "wa", "54": "wv", "55": "wi", "56": "wy"
    }
    # Get numerically sorted list of files
    filenames = list(Path(folder).glob(f"{filebasename}*.csv"))
    filenames_sorted = sorted(filenames, key=lambda x: int(re.search(r'\d+', x.stem).group()))
    
    # Get the cleaned crosswalk
    crosswalk = get_xwalk_naics(crosswalk_file)
    #print(f'crosswalk in microdataPostprocessing:\n{crosswalk.head()}')
    
    # Ensure the overall output directory exists
    if os.path.exists(outdir):
        if os.path.isdir(outdir):
            pass  # good
        else:
            raise Exception(f"Error: file {outdir} should be a directory.")
    else:
        os.makedirs(outdir)
    
    # Running counter for primary key assignment
    primary_key_counter = 1
    
    #Holds all data for final csv
    final_data = []
    yr=generalConfig["YEAR"]
    qtr=generalConfig["QTR"]
    # Process each file individually
    for file in filenames_sorted:
        df = postprocess_est_microdata_split(str(file), generalConfig=generalConfig, naicsdata=naicsdata, xwalk=crosswalk)
        n_rows = len(df)
        # Assign primary keys for this file's rows
        df['primary_key'] = np.arange(primary_key_counter, primary_key_counter + n_rows)
        primary_key_counter += n_rows
        final_data.append(df)
        # For each state, write/appending the subset from this file
        if ~set(df['state'].astype(str).unique().tolist()).issubset(set(state_abbr.values())):
            df['state']=df['state'].astype(str).str.rjust(2,"0")
        for fips_code, abbr in state_abbr.items():
            subdata = df[df['state'].astype(str) == str(fips_code)]
            if not subdata.empty:
                subdir = outdir / f"{abbr}{fips_code}"
                subdir.mkdir(parents=True, exist_ok=True)
                file_name = f"{abbr}{fips_code}_qdb_{yr}_{qtr}.csv"
                file_path = subdir / file_name
                
                # Write header only if file doesn't exist; else append without header.
                if file_path.exists():
                    subdata.to_csv(file_path, mode='a', header=False, index=False)
                else:
                    subdata.to_csv(file_path, mode='w', header=True, index=False)
                print(f"Appended data for {abbr}{fips_code} from {file} to {file_path}")
        
    
    # Combine all the data from the final_data list into a single DataFrame
    final_df = pd.concat(final_data, ignore_index=True)
    
    # Write the final aggregated data to 'MicrodataFinal.csv'
    final_output_path = outdir / "MicrodataFinal.csv"
    final_df.to_csv(final_output_path, mode='w', header=True, index=False)
    print(f"Final microdata file written to {final_output_path}")

    print("All files processed and state files generated.")

# Run the iterative combine and split process
#combine_and_split_iterative(yr = 2016, qtr = 1)

