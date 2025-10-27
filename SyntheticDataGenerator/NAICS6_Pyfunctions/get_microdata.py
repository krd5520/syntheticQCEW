import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
# with open('../config.yaml', 'r') as configFile:
#     config = yaml.safe_load(configFile)
#     microdataConfig = config['microdataConfig']
#     employmentConfig = config['employmentConfig']
def get_dirichlet_prior(estnum, g_shape, g_scale): 
    #generate gamma random variables to be shape parameters
    params = np.random.gamma(float(g_shape),float(g_scale),int(estnum))
    # shape and scale paramters for the gamma distribution were select by trial and error to get
    # some variety of proportions (not essentially equal across all establishments) while getting
    # similiar proportions per establishment across the generated dirichlet 
    # (i.e. not going from 15 to 20 to 16 for constant employment across the NAICS 6 aggregate)
    params_positive = np.abs(params) #shape parameters 
    if sum(params_positive)==0:
        params_positive = [1]*len(params_positive)
    return params_positive

# Get month 2 employment based on month 1 and month 3.
# The logic is to generate a normal random variable with the midpoint betwee m1emp and m3emp as the mean
# and the standard deviation proportional to difference between m1emp and m3emp over the mean of the two months
# noisecoef is a scalar for the standard deviation and the defaul was arbitrarly chosen
def get_m2emp_estlevel(m1emp,m3emp,noisecoef):
    nz_indicator = (m1emp>0)|(m3emp>0) #at least one month is position employment
    m2emp = [0]*len(nz_indicator) #initialize m2emp as zeros
    if sum(nz_indicator)>0: #if there is at least one establishment with positive employment
        #if m1emp+m2emp is 0 set numerator=0, denominator=1 (to avoid having to evaluate 0/0)
        noisevar_numerator = np.where(m1emp+m3emp==0,0,noisecoef*2*np.absolute(m1emp-m3emp)) 
        noisevar_denominator = np.where(m1emp+m3emp==0,1,m1emp+m3emp)
        #get variance values as a numpy array
        noisevar = np.array(np.divide(noisevar_numerator,noisevar_denominator),dtype=np.float64)
        noisesd = np.sqrt(noisevar.astype('float')) #get standard deviation
        
        #generate multivariate normal random variable with 
        # mean 0, variance corresponding to calulations above, and no correlation between elements
        # (i.e. independent normal random variables which all have mean 0, but each has it's own variance value)
        change_from_midpoint = np.random.multivariate_normal([0]*len(m3emp),np.diag(noisesd))
        m2emp = np.array(m1emp+(0.5*(m3emp-m1emp))+change_from_midpoint) #add midpoint to the normal rvs
        
        
        
        ######################### I messed this up but I already ran it all. I'll see if its a problem.
        # we adjust for month 2 employee counts that ended up negative
        #if m2emp is negative employees and month 1 or 3 has no employees, set month 2 to zero
        m2emp = np.where((m2emp<0) & ((m1emp==0)|(m3emp==0)),0,m2emp) 
        # if m2emp is negative and month 1 or 3 has employees, set month 2 to 1**********
        #m2emp = np.where((m2emp < 0) & ((m1emp!=0)|(m3emp!=0)), 1, m2emp)
        #*************the line above should be**************
        # I am assuming if month 1 and month 3 have employees then month 2 must have at least 1 employee.
        m2emp = np.where((m2emp <= 0) & ((m1emp!=0)&(m3emp!=0)), 1, m2emp) #month 1 & 3 have employees -> month 2 does too
    return np.rint(m2emp)

def find_emp_adj(m1emp,m3emp,addonidx,cntyN6):
    stopidx=False
    m1empg0 = np.argwhere(m1emp > 0)
    m3empg0 = np.argwhere(m3emp > 0)
    m1empeq1=np.argwhere(m1emp < 2)
    m3empeq1=np.argwhere(m3emp < 2)
    m1empeq0=np.argwhere(m1emp<1)
    m3empeq0=np.argwhere(m3emp<1)
    
    cantusem1emp= np.intersect1d(m1empg0,np.intersect1d(m1empeq1,m3empeq0))
    cantusem3emp=np.intersect1d(m3empg0,np.intersect1d(m1empeq0,m3empeq1))
    if m1empg0 is not None:
        if cantusem1emp is not None:
            canusem1emp=np.setdiff1d(m1empg0,cantusem1emp)
        else:
            canusem1emp=m1empg0
    else:
        canusem1emp=None
    if m3empg0 is not None:
        if cantusem3emp is not None:
            canusem3emp=np.setdiff1d(m3empg0,cantusem3emp)
        else:
            canusem3emp=m3empg0
    else:
        canusem3emp=None
    if canusem3emp is not None and canusem1emp is not None and len(canusem1emp)>0 and len(canusem3emp)>0:
        #print("len(canusem1emp)="+str(len(canusem1emp))+" ln(canusem3emp)="+str(len(canusem3emp)))
        selectwhich=np.random.randint(0,1)
        if selectwhich==1:
            m3emp[addonidx] = m3emp[addonidx] + 1
            m3empsubtract = np.random.choice(canusem3emp,1,False)
            m3emp[m3empsubtract] = m3emp[m3empsubtract] - 1
        else:
            m1emp[addonidx] = m1emp[addonidx] + 1
            m1empsubtract = np.random.choice(canusem1emp,1,False)
            m1emp[m1empsubtract] = m1emp[m1empsubtract] - 1
    elif canusem3emp is not None and len(canusem3emp)>0:
        m3emp[addonidx] = m3emp[addonidx] + 1
        m3empsubtract = np.random.choice(canusem3emp,1,False)
        m3emp[m3empsubtract] = m3emp[m3empsubtract] - 1
    elif canusem1emp is not None and len(canusem1emp)>0:
        m1emp[addonidx] = m1emp[addonidx] + 1
        m1empsubtract = np.random.choice(canusem1emp,1,False)
        m1emp[m1empsubtract] = m1emp[m1empsubtract] - 1
    else:
        stopidx=True
        #print("Warning: some establishments have all zero employment."+str(cntyN6))
    return m1emp, m3emp, stopidx
    

def adjust_emp_all_zeros(n6m1emp, n6m3emp, m1emp, m3emp, cntyN6):

    m1empzero = np.argwhere(m1emp<1)
    m3empzero=np.argwhere(m3emp<1)
    allzeros=np.intersect1d(m1empzero,m3empzero)
    if allzeros is None:
        numallzeroas=0
    else:
        numallzeros = len(allzeros)
    if numallzeros == 0:
        return (m1emp, m3emp)
    elif numallzeros == 1:
        if np.add(n6m1emp,n6m3emp) < 0:
            m1emp[allzeros] = m1emp[allzeros] + 1
        else:
            m1emp, m3emp, stopidx=find_emp_adj(m1emp,m3emp,allzeros,cntyN6+" m1:m3="+str(n6m1emp)+" : "+str(n6m3emp))
    else:  # len(m1empgeq1)+len(m3empgeq1)+len(bothgeq0)>=numallzeros:
        stopidx=False
        for zeroidx in allzeros:
            m1emp, m3emp, stopidx=find_emp_adj(m1emp,m3emp,zeroidx,cntyN6+" m1:m3="+str(n6m1emp)+" : "+str(n6m3emp))
            if ~stopidx:
                return m1emp, m3emp
    return m1emp, m3emp
                
     



    ## generate establishments for each county x 6-digit NAICS code. The number of establishments is determined by estnum
## the confidential values should sum to the aggregate values at the county x 6-digit NAICS code level.
def get_establishments_from_one_naics6(naics6row,gamma_shape,
                                       gamma_scale,
                                       noisecoef,
                                       wagemin=1):
    ## For conf values we get a random proportion for each establishment that sum to 1 (repeat for 4 conf values).
    ## Goals for Proportions:
    #### 1.  Aim to preserve the relationships between each employee count and wages within an establishment
    ######      Don't want establishment A to have [m1emp,m2emp,m3emp,wage]=[10,0,70,1000]
    ######      Thus 4 proportions within an establishment should be fairly similiar to each other.
    #### 2. Aim for some variety across establishments (Don't want NAICS6 aggregate values divided exactly evenly)
    #####################
    ## To do this we will
    #### 1. Generate shape parameters with a gamma random variable for each establishment (using gamma_shape & gamma_scale)
    #### 2. Generate a dirichlet rv for each establishment w/ shape parameters from step 1.
    #### 3. Repeat steps 1 and 2 three times (for m1emp, m3emp, and wage)
    #### 4. Generate m2emp based on m1emp and m3emp for each establishment (using noisecoef value)
    ## default gamma values were selected by trying several combinations, until I felt we had the balance of the two goals
    ## These proportions are multiplied by the confidential values at the naics6 level to get establishment level values
    

    n = naics6row['estnum'] #number of establishments
    minwage_reserve=float(wagemin)*float(n)
    naics6row["wage"]=float(naics6row["wage"])-float(minwage_reserve)
    shape_parameters = get_dirichlet_prior(n,g_shape=float(gamma_shape),g_scale=float(gamma_scale)) #shape params for dirichlet generation
    establishment_props = np.random.dirichlet(shape_parameters,3) #randomly generated proportions
    transpose_establishment_props = np.transpose(establishment_props) #transpose it for matrix multiplication


    conf_values = np.array([naics6row["m1emp"],naics6row["m3emp"],naics6row["wage"]]) #confidential values as an array

    establishment_values = np.multiply(conf_values,transpose_establishment_props) #establishment level values
    establishment_rows = np.transpose(establishment_values) #transpose for ease in creating data frame
    naics6val = naics6row["geoindkey"].split("_")[1] #get the naics 6 value
    #make dataframe with a row for each establishment and a
    #column for state, county,naics6, employment in month 1 and 3,and wages
    m1emp=np.rint(np.array(establishment_rows[0],dtype="float"))
    m3emp=np.rint(np.array(establishment_rows[1],dtype="float"))

    identifier=str(naics6row["state"])+str(naics6row["cnty"])+"_"+str(naics6val)+" n="+str(n)
    m1emp, m3emp=adjust_emp_all_zeros(naics6row["m1emp"],naics6row["m3emp"],m1emp,m3emp,identifier)
    m2emp = get_m2emp_estlevel(m1emp,m3emp,noisecoef=noisecoef) #get m2emp based on m1emp and m2emp

    #for state, cnty, and naics6 code,
    ##  the value from the naics6 row is repeated the same number of times as number of establishment
    establishments = pd.DataFrame({"state":[naics6row['state']]*int(n),
                                   "cnty":[naics6row['cnty']]*int(n),
                                   "naics6":[naics6val]*int(n),
                                   "m1emp":m1emp, #add m1emp
                                   "m2emp":m2emp, #add m2emp
                                   "m3emp":m3emp, #add m3mp
                                   "wage":np.rint(np.array(establishment_rows[2]+wagemin,dtype="float"))}) #add rounded wage

    return establishments

## get establishment level data from NAICS6 aggregates for each county by NAICS6 code
###INPUTS:
##### naics6df is naics6 aggregate data (needs columns: state,m1emp,m3emp,wage,cnty,estnum,geoindkey)
##### numchunk is the number of smaller datasets to split the full naics6 into. Each will produce a establishment-level
#####         dataset saved as SynMicrodata[iteration number].csv. This helps great checkpoints in case the code stops
#####         running or returns an error at any point. It could also allow the pd.apply command to be replaced with some
#####         parallelization commands like pool from multiprocessing or parallel_apply from pandarallel.
##### testsubset is boolean to test the make_syn_microdata on only the first few subsets of data
##### foldername is a string indicating where to save the SynMicrodata[iteration number].csv's
##### rseed is number to set the randomization seed.
### OUTPUTS: 
##### technically returns the etsablishment level data for the last subset of the naics6 data. However, it saves 
##### also saves each iterations' establishment level data in folder specified.
def make_syn_microdata(naics6df,microdataConfig,testsubset=False,outfoldername=os.getcwd(),rseed=None):
    if rseed is None and microdataConfig['EST_SEED'] is not None:
        rseed=microdataConfig['EST_SEED']
    if rseed is not None:
        np.random.seed(rseed)
    numchunk=microdataConfig['NUMCHUNK']
    counter=0 #to help name the files produced
    
    #split data into numchunk+1 subsets (the +1 is subset of size = the remainder of number of rows divided by numchunk)
    chunk_size=round(len(naics6df['state'])/numchunk) #size of each subset of naics6df
    print("starting make_syn_microdata: chunk size="+str(chunk_size)) 
    splitdf = [naics6df[i:i+chunk_size] for i in range(0,len(naics6df),chunk_size)] 
    print("split data frame checkpoint")
    
    if testsubset==True: #if testing in only a few dataframes, only keep 0:2 subsets of dataframe
        splitdf = splitdf[0:2] 
    for subdata in splitdf: #for each subset in the list of subsets from split naics6df
        start_time = time.time() #time of start of computation
        counter=counter+1 #iterate the counter up 1
        
        # for each row in subdata run 'get_establishments_from_one_naics6'
        # this command produces a series of dataframes (one for each naics6 by county code)
        df_per_naics6_list = subdata.apply(get_establishments_from_one_naics6, axis=1, args=(float(microdataConfig['GAM_SHAPE']),
                                       float(microdataConfig['GAM_SCALE']),
                                       float(microdataConfig['M2EMP_NOISECOEF']),
                                       float(microdataConfig["WAGE_MIN"])))
        
        # turn the series of dataframes into a list of dataframes and stack them
        microdata = pd.concat(df_per_naics6_list.to_list())
        
        #save the stacked dataframe 
        ##   (representing the establishments for each county by naics6 code in the subset of the naics6 data)
        ## These are saved to the outfolder name specified as .csv files named 
        ##    SynMicrodata1.csv to SynMicrodata[numchunk+1].csv 
        ##      (or SynMicrodata[numchunk].csv if the data frame is evenly split by numchunk w/out remainder)
        os.makedirs(outfoldername, exist_ok=True)
        microdata.to_csv(outfoldername+'SynMicrodata'+str(counter)+".csv",sep=',',index=False)
        # print a statement about how long the process took of making the 
        #     subset of establishment level data for each subset of naics6 data
        print("Data set "+str(counter)+" took",end=" ")
        print(time.time()-start_time,end=" ")
        print(" to process.")
    return microdata



##########  Testing functions and other scratch work ############
# # microdata = gm.make_syn_microdata(naics6)
# # print(microdata.head())
# # microdata.to_csv('SynMicrodata.csv',sep=',',index=False)

# # tempdf = pd.read_csv("~/Documents/DifferentialPrivacy/TestDownloadQWI/CopyPasteData/efsy_cbp_2016.csv")
# # print(tempdf.head())
# # print(dfrow)
# # print(dfrow['emp'])
# # print(get_dirichlet_prior(4))
# # print(np.random.dirichlet((1,2,3),2))
# # print(type(np.random.dirichlet((1,2,3),2)))
# # print(get_m1emp_estlevel(10,[0.2,0.3,0.35,0.15]))
# #testdf = pd.DataFrame(data={"geoindkey":['1001_202201','2003_202301'],"state":[1,1],"cnty":[1,1],
# #                         "geo4naics":["1001_2022","2003_2023"],"estnum":[64,25],"m1emp":[150,20],"m3emp":[65,25],"wage":[400000,100000]})
# #temp= make_syn_microdata(testdf, numchunk=1)
# #print(temp.head())
# # testestdf = scale_one_naics6(testrow)
# # print(testestdf)
# # print(type([1.35,4.2,9.001]))



# # print(np.random.dirichlet((1,1,1.1),4))
# # print(np.random.dirichlet((10,10,11),4))
# # print(np.random.dirichlet((1,35,70),4)) #this is best
# # print(np.random.dirichlet((135,130,150),4))




