import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
with open('../config.yaml', 'r') as configFile:
    config = yaml.safe_load(configFile)
    microdataConfig = config['microdataConfig']
    employmentConfig = config['employmentConfig']
def get_dirichlet_prior(estnum, g_shape, g_scale): 
    #generate gamma random variables to be shape parameters
    params = np.random.gamma(g_shape,g_scale,estnum)
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


def adjust_emp_all_zeros(n6m1emp, n6m3emp, m1emp, m3emp, iter=1):

    m1empgeq1 = np.argwhere(m1emp > 1)
    if m1empgeq1 is None:
        m1empgeq1 = np.array()
    else:
        m1empgeq1 = m1empgeq1.reshape(-1)
    m3empgeq1 = np.argwhere(m3emp > 1)
    if m3empgeq1 is None:
        m3empgeq1 = np.array()
    else:
        m3empgeq1 = m3empgeq1.reshape(-1)
    m1empgeq0 = np.argwhere(m1emp > 0)
    m3empgeq0 = np.argwhere(m3emp > 0)
    bothgeq0 = np.intersect1d(m1empgeq0, m3empgeq0)
    if bothgeq0 is None:
        bothgeq0 = np.array()
    # bothgeq0 = np.array([e in m1empgeq0.to_list() if e in m3empgeq0.to_list()])
    if len(m1empgeq1) + len(m3empgeq1) == 0:
        botheq1 = bothgeq0
    else:
        if len(m1empgeq1) < 1 or len(m3empgeq1) < 1:
            allgeq1 = np.append(m1empgeq1, m3empgeq1)
        else:
            allgeq1 = np.concatenate([m1empgeq1, m3empgeq1])
        botheq1 = np.setdiff1d(bothgeq0, allgeq1)  # essentially m1emp=1,m3emp=1
        if botheq1 is None:
            botheq1 = np.array()



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
            if len(m1empgeq1) > 0:
                m1emp[allzeros] = m1emp[allzeros] + 1
                m1empsubtract = m1empgeq1[0]
                #print("numallzeros="+str(numallzeros)+" m1empsubtract length=1")
                m1emp[m1empsubtract] = m1emp[m1empsubtract] - 1
            elif len(m3empgeq1) > 0:
                m3emp[allzeros] = m3emp[allzeros] + 1
                m3empsubtract = m3empgeq1[0]
                #print("numallzeros="+str(numallzeros)+" m3empsubtract length=1" )
                m3emp[m3empsubtract] = m3emp[m3empsubtract] - 1
            elif len(bothgeq0) > 0:
                m3emp[allzeros] = m3emp[allzeros] + 1
                m3empsubtract = bothgeq0[0]
                #print("numallzeros="+str(numallzeros)+" m3empsubtract length=1")
                m3emp[m3empsubtract] = m3emp[m3empsubtract] - 1
            else:
                #print("adding one to naics6 employment to correct establishments")
                m1emp[allzeros] = m1emp[allzeros] + 1
    else:  # len(m1empgeq1)+len(m3empgeq1)+len(bothgeq0)>=numallzeros:
        combolen=len(m1empgeq1)+len(m3empgeq1)
        #print("combolen is "+str(combolen)+" numallzeros is"+str(numallzeros))
        if combolen >= numallzeros:
            if m1empgeq1 is None or len(m1empgeq1)==0:
                #print("combolen>numallzeros but m1empgeq1=0")
                m3emp[allzeros] = m3emp[allzeros] + 1
                m3empsubtract = np.random.choice(m3empgeq1,len(allzeros),False)
                m3emp[m3empsubtract] = m3emp[m3empsubtract] - 1
                #print("numallzeros="+str(numallzeros)+" m3empsubtract length=" + str(len(m3empsubtract)))
            elif m3empgeq1 is None or len(m3empgeq1)==0:
                #print("combolen>=numallzeros but m3empgeq1=0")
                m1emp[allzeros] = m1emp[allzeros] + 1
                m1empsubtract = np.random.choice(m1empgeq1.reshape(-1), len(allzeros),False)
                #print("numallzeros="+str(numallzeros)+" m1empsubtract length=" + str(len(m1empsubtract)))
                m1emp[m1empsubtract] = m1emp[m1empsubtract] - 1
            else:
                #print("combolen>=numallzeros and m1empgeq1>0 and m3empgeq1>0")
                minsplit = max(0, len(m3empgeq1) - 1)
                maxsplit = numallzeros - len(m1empgeq1)
                if maxsplit<=0:
                    maxsplit=numallzeros-1
                if minsplit<maxsplit:
                    splitpoint = np.random.randint(minsplit, maxsplit)
                else:
                    splitpoint=maxsplit
                #print("minsplit is "+str(minsplit)+" maxsplit is "+str(maxsplit)+" splitpoint is"+str(splitpoint))
                if splitpoint == 0:
                    m3emp[allzeros] = m3emp[allzeros] + 1
                    m3empsubtract = m3emp[:len(allzeros)]
                    m3emp[m3empsubtract] = m3emp[m3empsubtract] - 1
                    #print("numallzeros="+str(numallzeros)+" m3empsubtract length=" + str(len(m3empsubtract)))
                elif splitpoint == numallzeros - 1:
                    m1emp[allzeros] = m1emp[allzeros] + 1
                    m1empsubtract = m1emp[:len(allzeros)]
                    m1emp[m1empsubtract] = m1emp[m1empsubtract] - 1
                    #print("numallzeros="+str(numallzeros)+" m1empsubtract length=" + str(len(m1empsubtract)))
                else:
                    m1empadd = allzeros[:splitpoint]
                    m3empadd = allzeros[splitpoint:]
                    m3empsubtract = np.random.choice(m3empgeq1, len(m3empadd),False)
                    m3emp[m3empsubtract] = m3emp[m3empsubtract] - 1
                    m3emp[m3empadd] = m3emp[m3empadd] + 1
                    m1emp[m1empadd] = m1emp[m1empadd] + 1
                    m1empsubtract = np.random.choice(m1empgeq1, len(m1empadd),False)
                    #print("numallzeros="+str(numallzeros)+" m1empsubtract+m3empsubtract length=" + str(len(m3empsubtract)+len(m1empsubtract)))
                    m1emp[m1empsubtract] = m1emp[m1empsubtract] - 1
        elif len(botheq1)>0 and combolen + len(botheq1) >= numallzeros:
            #print("combolen+len(botheq1)>=numallzeros")
            tempm1empsubtract = m1empgeq1
            m3empsubtract = m3empgeq1
            if len(botheq1) == 1:
                bonusm1empsubtract = botheq1
            else:
                bonusm1empsubtract = np.random.choice(botheq1, numallzeros - len(m1empgeq1) - len(m3empgeq1),False)
            m1empsubtract = np.concat([bonusm1empsubtract, tempm1empsubtract])
            #print("numallzeros="+str(numallzeros)+" m1empsubtract+m3empsubtract length=" + str(len(m3empsubtract)+len(m1empsubtract)))
            m1empadd = allzeros[:len(m1empsubtract)]
            m3empadd = allzeros[len(m1empsubtract):]
            m3emp[m3empsubtract] = m3emp[m3empsubtract] - 1
            m3emp[m3empadd] = m3emp[m3empadd] + 1
            m1emp[m1empadd] = m1emp[m1empadd] + 1
            m1emp[m1empsubtract] = m1emp[m1empsubtract] - 1
        else:
            #print("subtract from m1empgeq1 and botheq1 and m3empgeq1, then iterate")
            m1empsubtract = np.concat(m1empgeq1, botheq1)
            m3empsubtract = m3empgeq1
            m1empadd = allzeros[:len(m1empsubtract)]
            m3empadd = allzeros[len(m1empsubtract):]
            m3emp[m3empsubtract] = m3emp[m3empsubtract] - 1
            m3emp[m3empadd] = m3emp[m3empadd] + 1
            m1emp[m1empadd] = m1emp[m1empadd] + 1
            m1emp[m1empsubtract] = m1emp[m1empsubtract] - 1

    m1empzero = np.argwhere(m1emp < 1)
    m3empzero = np.argwhere(m3emp < 1)
    allzeros = np.intersect1d(m1empzero, m3empzero)
    if allzeros is None:
        numallzeros=0
    else:
        numallzeros = len(allzeros)
    if numallzeros == 0:
        return (m1emp, m3emp)
    elif numallzeros > 1 and iter < 50:
        #print(iter)
        return (adjust_emp_all_zeros(n6m1emp=n6m1emp, n6m3emp=n6m3emp, m1emp=m1emp, m3emp=m3emp, iter=iter + 1))
    else:
        #print("adding one to naics6 employment to correct establishments")
        m1emp[allzeros] = m1emp[allzeros] + 1
        return (m1emp, m3emp)



    ## generate establishments for each county x 6-digit NAICS code. The number of establishments is determined by estnum
## the confidential values should sum to the aggregate values at the county x 6-digit NAICS code level.
def get_establishments_from_one_naics6(naics6row,gamma_shape=microdataConfig['GAM_SHAPE'],
                                       gamma_scale=microdataConfig['GAM_SCALE'],
                                       noisecoef=employmentConfig['M2EMP_NOISECOEF'],
                                       wagemin=microdataConfig["WAGE_MIN"]):
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
    minwage_reserve=wagemin*n
    naics6row["wage"]=naics6row["wage"]-minwage_reserve
    shape_parameters = get_dirichlet_prior(n,g_shape=gamma_shape,g_scale=gamma_scale) #shape params for dirichlet generation
    establishment_props = np.random.dirichlet(shape_parameters,3) #randomly generated proportions
    transpose_establishment_props = np.transpose(establishment_props) #transpose it for matrix multiplication


    conf_values = naics6row[["m1emp","m3emp","wage"]].to_numpy() #confidential values as an array

    establishment_values = np.multiply(conf_values,transpose_establishment_props) #establishment level values
    establishment_rows = np.transpose(establishment_values) #transpose for ease in creating data frame
    naics6val = naics6row["geoindkey"].split("_")[1] #get the naics 6 value
    #make dataframe with a row for each establishment and a
    #column for state, county,naics6, employment in month 1 and 3,and wages
    m1emp=np.rint(np.array(establishment_rows[0],dtype="float"))
    m3emp=np.rint(np.array(establishment_rows[1],dtype="float"))

    m1emp, m3emp=adjust_emp_all_zeros(naics6row[["m1emp"]].to_numpy(),naics6row[["m3emp"]].to_numpy(),m1emp,m3emp,1)
    m2emp = get_m2emp_estlevel(m1emp,m3emp,noisecoef=noisecoef) #get m2emp based on m1emp and m2emp

    #for state, cnty, and naics6 code,
    ##  the value from the naics6 row is repeated the same number of times as number of establishment
    establishments = pd.DataFrame({"state":[naics6row['state']]*n,
                                   "cnty":[naics6row['cnty']]*n,
                                   "naics6":[naics6val]*n,
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
def make_syn_microdata(naics6df,numchunk,testsubset=False,outfoldername=os.getcwd(),rseed=microdataConfig["EST_SEED"]):
    np.random.seed(rseed)
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
        df_per_naics6_list = subdata.apply(get_establishments_from_one_naics6,axis=1)
        
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
# #                          "geo4naics":["1001_2022","2003_2023"],"estnum":[64,25],"m1emp":[150,20],"m3emp":[65,25],"wage":[400000,100000]})
# #temp= make_syn_microdata(testdf, numchunk=1)
# #print(temp.head())
# #print(temp[["m1emp","m2emp","m3emp","wage"]].describe())
# # testestdf = scale_one_naics6(testrow)
# # print(testestdf)
# # print(type([1.35,4.2,9.001]))
# # print(get_m2emp_estlevel(testestdf[['m1emp','m3emp']]))


# # print(np.random.dirichlet((1,1,1.1),4))
# # print(np.random.dirichlet((10,10,11),4))
# # print(np.random.dirichlet((1,35,70),4)) #this is best
# # print(np.random.dirichlet((135,130,150),4))




