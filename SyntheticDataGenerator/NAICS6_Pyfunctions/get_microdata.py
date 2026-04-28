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
    params = np.random.gamma(float(g_shape)/float(estnum),float(g_scale),int(estnum))
    # shape and scale paramters for the gamma distribution were select by trial and error to get
    # some variety of proportions (not essentially equal across all establishments) while getting
    # similiar proportions per establishment across the generated dirichlet 
    # (i.e. not going from 15 to 20 to 16 for constant employment across the NAICS 6 aggregate)
    params_positive = np.abs(params) #shape parameters 
    if sum(params_positive)==0:
        params_positive = [1]*len(params_positive)
    return params_positive

# Get month 2 employment based on month 1 and month 3.
# The logic is to generate a normal random variable with the midpoint betwee emp1 and emp3 as the mean
# and the standard deviation proportional to difference between emp1 and emp3 over the mean of the two months
# noisecoef is a scalar for the standard deviation and the defaul was arbitrarly chosen
def get_emp2_estlevel(emp1,emp3,noisecoef):
    nz_indicator = (emp1>0)|(emp3>0) #at least one month is position employment
    emp2 = [0]*len(nz_indicator) #initialize emp2 as zeros
    if sum(nz_indicator)>0: #if there is at least one establishment with positive employment
        #if emp1+emp2 is 0 set numerator=0, denominator=1 (to avoid having to evaluate 0/0)
        noisevar_numerator = np.where(emp1+emp3==0,0,noisecoef*2*np.absolute(emp1-emp3)) 
        noisevar_denominator = np.where(emp1+emp3==0,1,emp1+emp3)
        #get variance values as a numpy array
        noisevar = np.array(np.divide(noisevar_numerator,noisevar_denominator),dtype=np.float64)
        noisesd = np.sqrt(noisevar.astype('float')) #get standard deviation
        
        #generate multivariate normal random variable with 
        # mean 0, variance corresponding to calulations above, and no correlation between elements
        # (i.e. independent normal random variables which all have mean 0, but each has it's own variance value)
        change_from_midpoint = np.random.multivariate_normal([0]*len(emp3),np.diag(noisesd))
        emp2 = np.array(emp1+(0.5*(emp3-emp1))+change_from_midpoint) #add midpoint to the normal rvs
        
        
        
        ######################### I messed this up but I already ran it all. I'll see if its a problem.
        # we adjust for month 2 employee counts that ended up negative
        #if emp2 is negative employees and month 1 or 3 has no employees, set month 2 to zero
        emp2 = np.where((emp2<0) & ((emp1==0)|(emp3==0)),0,emp2) 
        # if emp2 is negative and month 1 or 3 has employees, set month 2 to 1**********
        #emp2 = np.where((emp2 < 0) & ((emp1!=0)|(emp3!=0)), 1, emp2)
        #*************the line above should be**************
        # I am assuming if month 1 and month 3 have employees then month 2 must have at least 1 employee.
        emp2 = np.where((emp2 <= 0) & ((emp1!=0)&(emp3!=0)), 1, emp2) #month 1 & 3 have employees -> month 2 does too
    return np.rint(emp2)

def find_empX_can_subtract_from(empX,empY,eq0_leq1_g0_empXY_dict=None):
    if eq0_leq1_g0_empXY_dict is None:
        eq0_leq1_g0_empXY_dict={
            'empX_g0' : np.argwhere(empX > 0),
            'empY_g0' : np.argwhere(empY > 0),
            'empX_leq1' : np.argwhere(empX < 2),
            'empY_leq1' : np.argwhere(empY < 2),
            'empX_eq0' : np.argwhere(empX < 1),
            'empY_eq0' : np.argwhere(empY < 1)}
    cant_use_empX = np.intersect1d(eq0_leq1_g0_empXY_dict['empX_g0'],
                                   np.intersect1d(eq0_leq1_g0_empXY_dict['empX_leq1'],eq0_leq1_g0_empXY_dict['empY_eq0']))  
    # essentially empX=1,empY=0, don't subtract 1 from emp1
    if eq0_leq1_g0_empXY_dict['empX_g0'] is not None:
        if cant_use_empX is not None:
            can_use_empX = np.setdiff1d(eq0_leq1_g0_empXY_dict['empX_g0'], cant_use_empX)  
            # emp1>1 or emp1==1, emp3>0, can subtract 1 from emp1
        else:
            can_use_empX = eq0_leq1_g0_empXY_dict['empX_g0']
    else:
        can_use_empX = None
    return can_use_empX
        
    
def find_emp_adj(emp1,emp3,onidx):
    stopidx=False
    can_use_emp1=find_empX_can_subtract_from(emp1,emp3)
    can_use_emp3=find_empX_can_subtract_from(emp3,emp1)

    if can_use_emp3 is not None and can_use_emp1 is not None and len(can_use_emp1)>0 and len(can_use_emp3)>0:
        #there are emp3 and emp1's where you can subtract one from, so randomly select which you take from
        selectwhich=np.random.randint(0,1)
        if selectwhich==1:
            emp3[onidx] = emp3[onidx] + 1
            emp3subtract = np.random.choice(can_use_emp3,1,False)
            emp3[emp3subtract] = emp3[emp3subtract] - 1
        else:
            emp1[onidx] = emp1[onidx] + 1
            emp1subtract = np.random.choice(can_use_emp1,1,False)
            emp1[emp1subtract] = emp1[emp1subtract] - 1
    elif can_use_emp3 is not None and len(can_use_emp3)>0: #you can ONLY subtract from emp3
        emp3[onidx] = emp3[onidx] + 1
        emp3subtract = np.random.choice(can_use_emp3,1,False)
        emp3[emp3subtract] = emp3[emp3subtract] - 1
    elif can_use_emp1 is not None and len(can_use_emp1)>0: #you can ONLY subtract from emp1
        emp1[onidx] = emp1[onidx] + 1
        emp1subtract = np.random.choice(can_use_emp1,1,False)
        emp1[emp1subtract] = emp1[emp1subtract] - 1
    else: #you can't subtract from emp1 or emp3
        stopidx=True
        #print("Warning: some establishments have all zero employment."+str(cntyN6))
    return emp1, emp3, stopidx
    

def adjust_emp_all_zeros(n6emp1, n6emp3, emp1, emp3):
## adjust establishments with 0 month 1 and month 3 employment.
## n6emp1, n6emp3 are the county X NAICS-6 employments for month 1 and 3 respectively
## emp1, and emp3 are vectors of month 1 and month3 employment with an element for each establishment
    if len(emp1)==1 and emp1==0 and emp3==0:
        emp1=emp1+1
        return(emp1,emp3)
    else:
        emp1zero = np.argwhere(emp1<1)
        emp3zero=np.argwhere(emp3<1)
        allzeros=np.intersect1d(emp1zero,emp3zero)
        if allzeros is None:
            numallzeros=0
        else:
            numallzeros = len(allzeros)
        if numallzeros == 0:
            return (emp1, emp3)
        elif numallzeros == 1:
            if np.add(n6emp1,n6emp3) < 0:
                emp1[allzeros] = emp1[allzeros] + 1
            else:
                emp1, emp3, stopidx=find_emp_adj(emp1,emp3,allzeros)#,cntyN6+" m1:m3="+str(n6emp1)+" : "+str(n6emp3))
        else:  # len(emp1geq1)+len(emp3geq1)+len(bothgeq0)>=numallzeros:
            for zeroidx in allzeros:
                emp1, emp3, stopidx=find_emp_adj(emp1,emp3,zeroidx)#,cntyN6+" m1:m3="+str(n6emp1)+" : "+str(n6emp3))
                if ~stopidx:
                    return emp1, emp3
        return emp1, emp3




def adjust_emp2_zero_sandwhich(emp1,emp2,emp3,depth_countdown=10):
    ### This could be improved if we had statistics for the number of open/closures of estbalishments
    emp1_nonzero = np.argwhere(emp1 > 0)
    emp2_zero = np.argwhere(emp2 < 1)
    emp3_nonzero = np.argwhere(emp3 > 0)
    zero_sandwhich = np.intersect1d(np.intersect1d(emp1_nonzero, emp3_nonzero), emp2_zero)
    numzero_sandwhiches = len(zero_sandwhich)
    if depth_countdown==0:
        emp2[zero_sandwhich]=emp2[zero_sandwhich]+1
        return emp1, emp2, emp3
    if zero_sandwhich is None:
        return emp1, emp2, emp3
    else:
        emp2_geq2=np.argwhere(emp2>=2)
        if emp2_geq2 is not None and len(emp2_geq2)>0:
            emp2[zero_sandwhich]=emp2[zero_sandwhich]+1
            subtractone_emp2 = np.random.choice(emp2_geq2.flatten(), min(numzero_sandwhiches,len(emp2_geq2)), False,emp2[emp2_geq2].flatten()/sum(emp2[emp2_geq2]) )
            emp2[subtractone_emp2] = emp2[subtractone_emp2] - 1
            return adjust_emp2_zero_sandwhich(emp1,emp2,emp3,depth_countdown=depth_countdown-1)
        else:
            opening = np.intersect1d(np.argwhere(emp3>0),np.intersect1d(np.argwhere(emp2 > 0), np.argwhere(emp1 < 1)))
            addone_idx=[]
            if opening is not None and len(opening)>0:
                shift_opening_month=np.random.randint(0,1,size=len(opening))
                if sum(shift_opening_month)==0:
                    shift_opening_month=[0]*len(opening)
                    shift_opening_month[0]=1
                if sum(shift_opening_month)>numzero_sandwhiches:
                    subtractone_emp2=np.random.choice(opening[shift_opening_month==1].flatten(),numzero_sandwhiches,False)
                    emp2[zero_sandwhich]=emp2[zero_sandwhich]+1
                    emp2[subtractone_emp2]=emp2[subtractone_emp2]-1
                    addone_idx=zero_sandwhich
                else:
                    addone_idx=np.random.choice(zero_sandwhich.flatten(),sum(shift_opening_month),False)
                    emp2[addone_idx]=emp2[addone_idx]+1
                    emp2[opening[shift_opening_month==1]]=emp2[opening[shift_opening_month==1]]-1
            zero_sandwhich=np.setdiff1d(zero_sandwhich,addone_idx)
            numzero_sandwhiches=len(zero_sandwhich)
            closing = np.intersect1d(np.argwhere(emp1>0),np.intersect1d(np.argwhere(emp2 > 0), np.argwhere(emp3 < 1)))
            if closing is not None and numzero_sandwhiches>0 and len(closing)>0:
                shift_closing_month = np.random.randint(0, 1, size=len(closing))
                if sum(shift_closing_month)==0:
                    shift_closing_month=[0]*len(closing)
                    shift_closing_month[0]=1
                if sum(shift_closing_month)>numzero_sandwhiches:
                    subtractone_emp2=np.random.choice(closing[shift_closing_month==1].flatten(),numzero_sandwhiches,False)
                    emp2[zero_sandwhich] = emp2[zero_sandwhich] + 1
                    emp2[subtractone_emp2] = emp2[subtractone_emp2] - 1
                else:
                    addone_idx=np.random.choice(zero_sandwhich.flatten(),sum(shift_closing_month),False)
                    emp2[addone_idx]=emp2[addone_idx]+1
                    emp2[closing[shift_closing_month == 1]] = emp2[closing[shift_closing_month == 1]] - 1
            return adjust_emp2_zero_sandwhich(emp1, emp2, emp3,depth_countdown=depth_countdown-1)


    ## generate establishments for each county x 6-digit NAICS code. The number of establishments is determined by estnum
## the confidential values should sum to the aggregate values at the county x 6-digit NAICS code level.
def get_establishments_from_one_naics6(naics6row,gamma_shape,
                                       gamma_scale,
                                       noisecoef,
                                       wagemin=1):
    ## For conf values we get a random proportion for each establishment that sum to 1 (repeat for 4 conf values).
    ## Goals for Proportions:
    #### 1.  Aim to preserve the relationships between each employee count and wages within an establishment
    ######      Don't want establishment A to have [emp1,emp2,emp3,wage]=[10,0,70,1000]
    ######      Thus 4 proportions within an establishment should be fairly similiar to each other.
    #### 2. Aim for some variety across establishments (Don't want NAICS6 aggregate values divided exactly evenly)
    #####################
    ## To do this we will
    #### 1. Generate shape parameters with a gamma random variable for each establishment (using gamma_shape & gamma_scale)
    #### 2. Generate a dirichlet rv for each establishment w/ shape parameters from step 1.
    #### 3. Repeat steps 1 and 2 three times (for emp1, emp3, and wage)
    #### 4. Generate emp2 based on emp1 and emp3 for each establishment (using noisecoef value)
    ## default gamma values were selected by trying several combinations, until I felt we had the balance of the two goals
    ## These proportions are multiplied by the confidential values at the naics6 level to get establishment level values
    
    ##Check for NA's
    isna_index=naics6row[naics6row.isna()].index.values
    if any(x in ['estnum','emp1','emp3','wages','geoindkey'] for x in isna_index):
        raise Exception("County by NAICS-6 data should not have any NA values in the estnum, emp1, emp3, wages, or geoindkey columns.")

    n = naics6row['estnum'] #number of establishments
    minwage_reserve=float(wagemin)*float(n)
    naics6row["wages"]=float(naics6row["wages"])-float(minwage_reserve)
    shape_parameters = get_dirichlet_prior(n,g_shape=float(gamma_shape),g_scale=float(gamma_scale)) #shape params for dirichlet generation
    establishment_props_emps_ends = np.random.dirichlet(shape_parameters,2) #randomly generated proportions
    establishment_props_wages = np.random.dirichlet(shape_parameters,1) #randomly generated proportions
    establishment_props_emp2 = np.random.dirichlet(shape_parameters,1) #randomly generated proportions


    transpose_establishment_props = np.transpose(establishment_props_emps_ends) #transpose it for matrix multiplication


    conf_values = np.array([naics6row["emp1"],naics6row['emp2'],naics6row["emp3"],naics6row["wages"]]) #confidential values as an array
    conf_values_emp_ends = np.array([naics6row["emp1"], naics6row["emp3"]])  # confidential values as an array

    #get emp1, emp3
    establishment_values_emps_ends = np.multiply(conf_values_emp_ends,transpose_establishment_props) #establishment level values
    print(establishment_values_emps_ends.head())
    establishment_rows = np.transpose(establishment_values) #transpose for ease in creating data frame
    if "industry" not in naics6row.index.values:
        naics6val = naics6row["geoindkey"].str.split("_").iloc[:,1] #get the naics 6 value
    else:
        naics6val=naics6row['industry']
    #make dataframe with a row for each establishment and a
    #column for state, county,naics6, employment in month 1 and 3,and wages
    emp1=np.rint(np.array(establishment_rows[0],dtype="float"))
    emp3=np.rint(np.array(establishment_rows[2],dtype="float"))
    #identifier = str(naics6row["state"]) + str(naics6row["cnty"]) + "_" + str(naics6val) + " n=" + str(n)
    emp1, emp3 = adjust_emp_all_zeros(naics6row["emp1"], naics6row["emp3"], emp1, emp3)

    if 'emp2' not in isna_index:
        emp2=np.rint(np.array(establishment_rows[1],dtype="float"))
    else:
        emp2 = get_emp2_estlevel(emp1,emp3,noisecoef=noisecoef) #get emp2 based on emp1 and emp2
    num_zero_sandwhich=len(emp1[(emp2==0)&(emp3>0)&(emp1>0)])
    if num_zero_sandwhich>0 and num_zero_sandwhich is not None:
        (emp1,emp2,emp3)=adjust_emp2_zero_sandwhich(emp1,emp2,emp3)

    #for state, cnty, and naics6 code,
    ##  the value from the naics6 row is repeated the same number of times as number of establishment
    establishments = pd.DataFrame({"state":[naics6row['state']]*int(n),
                                   "cnty":[naics6row['cnty']]*int(n),
                                   "naics6":[naics6val]*int(n),
                                   "emp1":emp1, #add emp1
                                   "emp2":emp2, #add emp2
                                   "emp3":emp3, #add m3mp
                                   "wages":np.rint(np.array(establishment_rows[2]+wagemin,dtype="float")),
                                  "geoindkey":[naics6row['geoindkey']]*int(n)}) #add rounded wage

    return establishments

## get establishment level data from NAICS6 aggregates for each county by NAICS6 code
###INPUTS:
##### naics6df is naics6 aggregate data (needs columns: state,emp1,emp3,wage,cnty,estnum,geoindkey)
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
def make_syn_microdata(naics6df,microdataConfig,naicsdata=None, testsubset=False,rseed=None):
    if rseed is None and microdataConfig['EST_SEED'] is not None:
        rseed=microdataConfig['EST_SEED']
    if rseed is not None:
        np.random.seed(rseed)
    numchunk=microdataConfig['NUMCHUNK']
    outfoldername=microdataConfig['SUBSET_OUTPATH']


    counter=0 #to help name the files produced
    
    #split data into numchunk+1 subsets (the +1 is subset of size = the remainder of number of rows divided by numchunk)
    chunk_size=round(len(naics6df['state'])/numchunk) #size of each subset of naics6df
    print("starting make_syn_microdata: chunk size="+str(chunk_size)) 
    splitdf = [naics6df.iloc[i:i+chunk_size,:] for i in range(0,len(naics6df),chunk_size)]
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
                                       float(microdataConfig['EMP2_NOISECOEF']),
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
# # print(get_emp1_estlevel(10,[0.2,0.3,0.35,0.15]))
# #testdf = pd.DataFrame(data={"geoindkey":['1001_202201','2003_202301'],"state":[1,1],"cnty":[1,1],
# #                         "geo4naics":["1001_2022","2003_2023"],"estnum":[64,25],"emp1":[150,20],"emp3":[65,25],"wages":[400000,100000]})
# #temp= make_syn_microdata(testdf, numchunk=1)
# #print(temp.head())
# # testestdf = scale_one_naics6(testrow)
# # print(testestdf)
# # print(type([1.35,4.2,9.001]))



# # print(np.random.dirichlet((1,1,1.1),4))
# # print(np.random.dirichlet((10,10,11),4))
# # print(np.random.dirichlet((1,35,70),4)) #this is best
# # print(np.random.dirichlet((135,130,150),4))




