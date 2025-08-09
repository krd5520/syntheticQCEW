import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
import yaml
with open('./config.yaml', 'r') as configFile:
    config = yaml.safe_load(configFile)
    postprocessingConfig = config['postprocessingConfig']
    generalConfig=config['generalConfig']

inputpath = Path(postprocessingConfig['INPATH'])
outputpath = Path(postprocessingConfig['OUTPATH']) 

# Crosswalk file from 
## https://www.bls.gov/cew/classifications/industry/industry-supersectors.htm 
crosswalk_file = postprocessingConfig['CROSSWALK']
state_file = postprocessingConfig['FIPS_STATE_FILE']

state_abbr = {
    "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca", "08": "co", "09": "ct", "10": "de", 
    "11": "dc", "12": "fl", "13": "ga", "15": "hi", "16": "id", "17": "il", "18": "in", "19": "ia", "20": "ks", 
    "21": "ky", "22": "la", "23": "me", "24": "md", "25": "ma", "26": "mi", "27": "mn", "28": "ms", "29": "mo", 
    "30": "mt", "31": "ne", "32": "nv", "33": "nh", "34": "nj", "35": "nm", "36": "ny", "37": "nc", "38": "nd", 
    "39": "oh", "40": "ok", "41": "or", "42": "pa", "44": "ri", "45": "sc", "46": "sd", "47": "tn", "48": "tx", 
    "49": "ut", "50": "vt", "51": "va", "53": "wa", "54": "wv", "55": "wi", "56": "wy"
}

def get_xwalk_naics(crosswalk_file=crosswalk_file):
    """
    Reads and processes the NAICS crosswalk file.
    
    Steps:
    1. Reads the CSV file.
    2. Cleans the 'super_sector' column by removing all non-numeric characters.
    3. Cleans the 'naics_sector' column by removing all characters except digits and the '-' 
    4. For rows where 'naics_sector' contains a dash, expands the row into multiple rows 
       for each individual sector in the range, specifically 31-33, 44-45, and 48-49
    5. Removes any rows still containing a dash in 'naics_sector'.
    """
    xwalk = pd.read_csv(crosswalk_file)
    
    # Clean 'super_sector' by removing non-numeric characters
    xwalk['super_sector'] = xwalk['super_sector'].astype(str).str.replace(r'[^0-9]', '', regex=True)
    
    # Clean 'naics_sector' while preserving dashes
    xwalk['naics_sector'] = xwalk['naics_sector'].astype(str).str.replace(r'[^0-9-]', '', regex=True)
    
    # Identify rows with a dash in 'naics_sector'
    dash_rows = xwalk[xwalk['naics_sector'].str.contains("-")]
    
    # Define expanded ranges:
    expand_mapping = {
        "31-33": ["31", "32", "33"],
        "44-45": ["44", "45"],
        "48-49": ["48", "49"]
    }
    
    # Expand each row that contains a dash
    expanded_rows = []
    for idx, row in dash_rows.iterrows():
        key = row['naics_sector']
        if key in expand_mapping:
            for val in expand_mapping[key]:
                new_row = row.copy()
                new_row['naics_sector'] = val
                expanded_rows.append(new_row)
    
    if expanded_rows:
        df_expanded = pd.DataFrame(expanded_rows)
        xwalk = pd.concat([xwalk, df_expanded], ignore_index=True)
    
    # Remove rows still containing a dash in 'naics_sector'
    xwalk = xwalk[~xwalk['naics_sector'].str.contains("-")]
    
    return xwalk

def cut1(naicscode):
    # Cuts 1 character off end of string
    return str(naicscode)[:-1]

def postprocess_est_microdata_split(filename, yr, qtr, xwalk=None):
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
    
    # Generate NAICS 5,4,3,sector codes
    estdf['naics5'] = estdf['naics6'].apply(cut1)
    estdf['naics4'] = estdf['naics5'].apply(cut1)
    estdf['naics3'] = estdf['naics4'].apply(cut1)
    estdf['naics_sector'] = estdf['naics3'].apply(cut1)
    estdf['naics'] = estdf['naics6']
    
    # Merge in the super_sector info from the crosswalk
    estdf = estdf.merge(xwalk[['naics_sector', 'super_sector']], on='naics_sector', how='left')
    
    # Add constant columns
    estdf['year'] = yr
    estdf['qtr'] = qtr
    estdf['own'] = 5
    estdf['can_agg'] = 'Y'
    estdf['rectype'] = 'C'
    
    # Employee columns are integers
    estdf['m1emp'] = estdf['m1emp'].astype(int)
    estdf['m2emp'] = estdf['m2emp'].astype(int)
    estdf['m3emp'] = estdf['m3emp'].astype(int)
    estdf['wage'] = estdf['wage'].astype(int)
    
    # Reorder columns
    estdf = estdf[['year', 'qtr', 'state', 'cnty', 'own', 'naics', 
                   'naics3', 'naics4', 'naics5', 'naics_sector', 'super_sector', 
                   'm1emp', 'm2emp', 'm3emp', 'wage', 'can_agg', 'rectype']]
    
    # Gets file number and add as a column
    filenum = re.search(r'\d+', filename).group()
    estdf['filenumber'] = filenum
    
    return estdf

def combine_and_split_iterative(yr, qtr, filebasename="SynMicrodata", folder=inputpath, outdir=outputpath):
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
    # Get numerically sorted list of files
    filenames = list(Path(folder).glob(f"{filebasename}*.csv"))
    filenames_sorted = sorted(filenames, key=lambda x: int(re.search(r'\d+', x.stem).group()))
    
    # Get the cleaned crosswalk
    crosswalk = get_xwalk_naics()
    
    # Ensure the overall output directory exists
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Running counter for primary key assignment
    primary_key_counter = 1
    
    #Holds all data for final csv
    final_data = []
    
    # Process each file individually
    for file in filenames_sorted:
        df = postprocess_est_microdata_split(str(file), yr=yr, qtr=qtr, xwalk=crosswalk)
        n_rows = len(df)
        # Assign primary keys for this file's rows
        df['primary_key'] = np.arange(primary_key_counter, primary_key_counter + n_rows)
        primary_key_counter += n_rows
        final_data.append(df)
        # For each state, write/appending the subset from this file
        for fips_code, abbr in state_abbr.items():
            subdata = df[df['state'] == int(fips_code)]
            if not subdata.empty:
                subdir = outdir / f"{abbr}{fips_code}"
                subdir.mkdir(parents=True, exist_ok=True)
                file_name = f"{abbr}{fips_code}_qdb_{yr}_1.csv"
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

