import pandas as pd


def not_subdivided(geoindkey,missdf,investigatedf=None,fulldf=None):
    missdf.reset_index(drop=True,inplace=True)
    if geoindkey.endswith("------"):
        print("geoindkey indicates it is a county total, which is weird")
        print(missdf[missdf['geoindkey']==geoindkey])
        return missdf, investigatedf, fulldf
    else:
        geoindstem=geoindkey.replace("/","").replace("-","")
        lowerlvl_bool=missdf['geoindkey'].str.startswith(geoindstem,na=False)
        lowerindlvls=missdf[lowerlvl_bool]
        uniqest=lowerindlvls['estnum'].unique()
        #print(geoindkey + " has stem " + geoindstem+" and unique est is: ",uniqest)
        keepgeoindkey = missdf['geoindkey'] == geoindkey
        missdf = missdf[(keepgeoindkey) | (~lowerlvl_bool)].reset_index(drop=True)
        checkgeoindkeyin = missdf['geoindkey'] == geoindkey
        if checkgeoindkeyin.sum() == 0:
            print("something wrong geoindkey was not kept " + geoindkey)
        checklowernotin = missdf['geoindkey'].str.startswith(geoindstem, na=False)
        if checklowernotin.sum() > 1:
            print("something wrong sublevels of geoindkey were kept " + geoindkey)
        if len(uniqest)==1: #each lower cell is essentially a copy of geoindkey cell
            if fulldf is not None:
                skiplist=None
                abovegeostem=geoindstem[:-1]
                geoindrw=missdf.loc[keepgeoindkey,:]
                indlvlkey=geoindrw['ind_level'].reset_index(drop=True).iloc[0]
                if indlvlkey==2:
                    fulldf,skiplist=fix_missing_sector_cbp(fulldf,geoindkey,skiplist)

                else:
                    abovefull=fulldf[fulldf['geoindkey'].str.startswith(abovegeostem,na=False)]
                    oneabove=abovefull.loc[abovefull['ind_level']==indlvlkey-1,:]
                    atlvl=abovefull.loc[abovefull['ind_level']==indlvlkey,:]
                    geoindkey_atlvl=atlvl['geoindkey'].unique()
                    if len(geoindkey_atlvl)>1:
                        fulldf, skiplist, investigatedf= fix_missing_naics_cbp(fulldf,geoindkey,skiplist,investigatedf)
                        #print("More than 1 geoindkey at ind_level: ",geoindkey_atlvl)
                    else: #if geoindkey is the only child at this level of the parent geoindstem, use one above
                        if len(oneabove)!=1: #there isn't one above???
                            temp=geoindrw
                            temp['note']="no 1-level up geoindkey (not_subdivided)"
                            if investigatedf is None:
                                investigatedf=temp
                            else:
                                investigatedf=pd.concat([investigatedf,temp],ignore_index=True,axis=0)
                        else:
                            fulldf.loc[fulldf['geoindkey'].str.startswith(abovegeostem,na=False),"emp"]=oneabove['emp'].reset_index(drop=True).iloc[0]
            return missdf, investigatedf,fulldf
        else:
            lowerindlvls=lowerindlvls.sort_values(by='ind_level')
            lowerindlvls.reset_index(inplace=True)
            changeest=lowerindlvls['est'].diff()
            change_bool=changeest!=0
            idx_change=change_bool[change_bool].index.values.tolist()
            beforelvl=lowerindlvls.iloc[[x-1 for x in idx_change],:]
            beforelvl=beforelvl['ind_level'].to_numpy()
            changelvl=np.array([x+1 for x in np.unique(beforelvl)])
            investigate_lvls=np.unique(np.concat((beforelvl,changelvl)))
            temp=lowerindlvls[lowerindlvls['ind_level'].isin(investigate_lvls.tolist())].reset_index(drop=True)
            temp["note"]=["est changes (not subdivided)"]*len(temp)
            if investigatedf is None:
                investigatedf=temp
            else:
                investigatedf=pd.concat([investigatedf,temp],axis=0,ignore_index=True)

            #print("there are subdivided lower levels of "+geoindkey)
            return missdf, investigatedf, fulldf

def fix_missing_sector_cbp(data,geoindkey,skiplist=None):
    if skiplist is not None and geoindkey in skiplist:
        return data, skiplist
    geoindrw=data[data['geoindkey']==geoindkey]
    geo=geoindrw['geography'].reset_index(drop=True).iloc[0]
    geodf=data[data['geography']==geo]
    cntytotal_emp=geodf.loc[geodf['ind_level']==0,'emp'].values[0]
    cntytotal_wage=geodf.loc[geodf['ind_level']==0,'qp1'].values[0]
    geosectordf=geodf[geodf['ind_level']==2]
    geosector_emp=geosectordf['emp']
    geosectsum_emp=geosector_emp.sum(skipna=True)
    geosector_wage = geosectordf['qp1']
    geosectsum_wage=geosector_wage.sum(skipna=True)
    if geosectsum_emp==cntytotal_emp:
        data.loc[(data['geography']==geo)&(data['emp'].isna())&(data["ind_level"]==2),"emp"]=0
        data.loc[(data['geography']==geo)&(data['emp'].isna())&(data["ind_level"]==2),"_merge"]="ronly_fill0"
        if skiplist is None:
            skiplist=data.loc[(data['geography']==geo)&(data['emp'].isna())&(data["ind_level"]==2),'geoindkey']
        else:
            skiplist=pd.concat([skiplist,data.loc[(data['geography']==geo)&(data['emp'].isna())&(data["ind_level"]==2),'geoindkey']],ignore_index=True)

        return data, skiplist
    if geosectsum_wage==cntytotal_wage:
        data.loc[(data['geography']==geo)&(data['emp'].isna())&(data["ind_level"]==2), "qp1"] = 0
        data.loc[(data['geography']==geo)&(data['emp'].isna())&(data["ind_level"]==2),'qp1_nf']="Manual Fill"
        return(data)
    numna_emp=geosector_emp.isna().sum()
    numna_wage=geosector_wage.isna().sum()
    if numna_emp==1:
        data.loc[data["geoindkey"] == geoindkey, "emp"] = cntytotal_emp - geosector_emp.sum(skipna=True)
    else:
        empdiff=cntytotal_emp-geosectsum_emp
        imputeparams_emp=geosectordf.loc[geosectordf['emp'].isna(),'estnum']
        impute_emp=dirichlet_divider(imputeparams_emp,empdiff,size=1,rseed=int(geo))
        data.loc[(data['geography']==geo)&(data['emp'].isna())&(data["ind_level"]==2),"emp"]=impute_emp
        data.loc[(data['geography']==geo)&(data['emp'].isna())&(data["ind_level"]==2),"_merge"]="ronly_impute"
        data.loc[(data['geography']==geo)&(data['emp'].isna())&(data["ind_level"]==2),"emp_nf"]="Manual Fill"

        if skiplist is None:
            skiplist=data.loc[(data['geography']==geo)&(data['emp'].isna())&(data["ind_level"]==2),'geoindkey']
        else:
            skiplist=pd.concat([skiplist,data.loc[(data['geography']==geo)&(data['emp'].isna())&(data["ind_level"]==2),'geoindkey']],ignore_index=True)
    if geosector_wage.isna().sum()>1:
        pass
        #print("Too many missing sectors ("+str(numna_wage)+") fill in wages for "+str(geo))
    else:
        data.loc[data["geoindkey"]==geoindkey,"qp1"]=cntytotal_wage-geosector_wage.sum(skipna=True)
        data.loc[data["geoindkey"] == geoindkey, "qp1_nf"] = "Manual Fill"

    return data, skiplist

def fix_missing_naics_cbp(data,geoindkey,skiplist=None,investigatedf=None):
    if skiplist is not None and geoindkey in skiplist:
        return data, skiplist, investigatedf
    geoindrw=data[data['geoindkey']==geoindkey]
    indlvl=geoindrw['ind_level'].values[0]
    if indlvl==2:
        data, skiplist=fix_missing_sector_cbp(data,geoindkey,skiplist)
        return data, skiplist, investigatedf
    else:
        geoindstem=geoindkey[:-1]
        gistemI=(data['geoindkey'].str.startswith(geoindstem))
        twodf=data.loc[(gistemI)&(data['ind_level'].isin([indlvl,indlvl-1])),:]
        if len(twodf[twodf['ind_level']==indlvl-1])==0 or twodf.loc[twodf['ind_level']==indlvl-1,'emp'].isna().sum()==0:
            temp=data[data['geoindkey']==geoindkey].reset_index(drop=True)
            temp['note'] = ['1 level up missing or NA (fix_missing_naics_cbp)']*len(temp)
            if investigatedf is None:
                investigatedf = temp
            else:
                investigatedf = pd.concat([investigatedf,temp ], ignore_index=True, axis=0)
        else: #there are level above values
            uptotal_emp=twodf.loc[twodf['ind_level']==indlvl-1,'emp'].values[0]
            uptotal_wage=twodf.loc[twodf['ind_level']==indlvl-1,'qp1'].values[0]
            lowdf=twodf[twodf['ind_level']==indlvl]
            low_emp=lowdf['emp']
            low_wage = lowdf['qp1']
            lwsum_emp=low_emp.sum(skipna=True)
            lwsum_wage=low_wage.sum(skipna=True)
            if lwsum_emp==uptotal_emp:
                data.loc[(gistemI)&(data['emp'].isna())&(data["ind_level"]==indlvl),"emp"]=0
                data.loc[(gistemI) & (data['emp'].isna()) & (
                            data["ind_level"] == indlvl), "_merge"] = "ronly_fill0"
                if skiplist is None:
                    skiplist=data.loc[(gistemI)&(data['emp'].isna())&(data["ind_level"]==indlvl),'geoindkey']
                else:
                    skiplist=pd.concat([skiplist,data.loc[(gistemI)&(data['emp'].isna())&(data["ind_level"]==indlvl),'geoindkey']],ignore_index=True)
                return data, skiplist
            if lwsum_wage==uptotal_wage:
                data.loc[(gistemI)&(data['emp'].isna())&(data["ind_level"]==indlvl), "qp1"] = 0
                data.loc[(gistemI)&(data['emp'].isna())&(data["ind_level"]==indlvl),'qp1_nf']="Manual Fill"
                return(data)
            numna_emp=low_emp.isna().sum()
            numna_wage=low_wage.isna().sum()
            if numna_emp==1:
                data.loc[data["geoindkey"] == geoindkey, "emp"] = uptotal_emp - lwsum_emp
            else:
                empdiff=uptotal_emp-lwsum_emp
                imputeparams_emp=lowdf.loc[lowdf['emp'].isna(),'est']
                impute_emp=dirichlet_divider(imputeparams_emp,empdiff,size=1,rseed=int(lowdf[lowdf['emp'].isna()].index.values[0]))
                data.loc[(gistemI)&(data['emp'].isna())&(data["ind_level"]==indlvl),"emp"]=impute_emp
                data.loc[(gistemI)&(data['emp'].isna())&(data["ind_level"]==indlvl),"emp_nf"]="Manual Fill"
                data.loc[(gistemI)&(data['emp'].isna())&(data["ind_level"]==indlvl),"_merge"]="ronly_impute"
                if skiplist is None:
                    skiplist=data.loc[(gistemI)&(data['emp'].isna())&(data["ind_level"]==indlvl),'geoindkey']
                else:
                    skiplist=pd.concat([skiplist,data.loc[(gistemI)&(data['emp'].isna())&(data["ind_level"]==indlvl),'geoindkey']],ignore_index=True)
            if numna_wage>1:
                pass
                #print("Too many missing sectors ("+str(numna_wage)+") fill in wages for "+str(geo))
            else:
                data.loc[data["geoindkey"]==geoindkey,"qp1"]=uptotal_wage-lwsum_wage
                data.loc[data["geoindkey"] == geoindkey, "qp1_nf"] = "Manual Fill"
    return data, skiplist, investigatedf


## Functions below are only used if for some reason there are missing imputed employments. They investigate and attempt to fill the situation.
def cbp_mismatch(data):
    if "emp" not in data.columns:
        changecolname=True
        data['emp']=data['emp3_cbp']
        data['qp1']=data['wages_cbp']
    else:
        changecolname=False
    if "estnum" not in data.columns:
        data['estnum']=data['est']
    missdf=data[data['emp'].isna()]
    if len(missdf)>0:
        #print(missdf.head())
        missfips=missdf['geography'].unique()
        missindlvl=sorted(missdf['ind_level'].unique())
        print("In merged CBP: There are "+str(len(missdf))+" rows without emp. # FIPS codes missing="+str(len(missfips))+". Industry levels with missing values: "+", ".join([str(x) for x in missindlvl]))
        weirddf=None
        for ilvl in missindlvl:
            print("Starting to adjust industry level: "+str(ilvl))
            if ilvl==6:
                pass
            else:
                missilvl=missdf[missdf['ind_level']==ilvl]
                for gikey in missilvl['geoindkey']:
                    missdf,weirddf, data=not_subdivided(gikey,missdf,weirddf,data)
        print(missdf[missdf['emp_nf']=="Manual Fill"].head())
        print("weirddf has "+str(len(weirddf))+" rows.")
        weirddf.drop(columns=['industry','geography','qp1','qp1_nf','_merge','emp'],inplace=True)
        weirddf.reset_index(inplace=True,drop=True)
        print(pd.crosstab(weirddf['ind_level'],weirddf['note']))
        print("ind_level")
        print(weirddf['ind_level'].value_counts())
        print("emp_nf")
        print(weirddf['emp_nf'].value_counts())
        print("estnum")
        print(weirddf['estnum'].value_counts())
        print("state")
        print(weirddf['state'].value_counts())
        print("head weirddf")
        print(weirddf.head())
        print("99----")
        print(data[data['geoindkey'].str.endswith('99----')])
        #weirdidx=data['geoindkey'].isin(weirddf['geoindkey'].unique())
        print("Manual Fill on Employment")
        print(pd.crosstab(data['ind_level'],data['emp_nf']))
        print("Manual Fill on Wages")
        print(pd.crosstab(data['ind_level'], data['qp1_nf']))
        #if "ub" in data.columns:
        #    data[data['geoindkey'].isin(weirdidx),"emp"]=data.loc[data['geoindkey'].isin(weirdidx),:].apply(random_midpoint,axis=1)
    if changecolname:
        data['emp3_cbp']=data['emp']
        data['wages_cbp']=data['qp1']
        data.drop(columns=["emp",'qp1'],inplace=True)
    return data


# get random midpoint between lower bound (lb) and upper bound (up)
def random_midpoint(datarw):
    if datarw['lb']==datarw['ub']:
        return(datarw['lb'])
    elif datarw['lb']<datarw['ub']:
        return(round(np.random.uniform(float(datarw['lb']),float(datarw['ub'])),0))
    else:
        return datarw['lb']




