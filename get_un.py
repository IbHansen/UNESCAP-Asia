# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 13:41:40 2021

@author: bruger
"""

import pandas as pd
import re
from pathlib import Path
from dataclasses import dataclass
import functools
from tqdm import tqdm 
import os



from modelclass import model 
import modelmf 
import modelmanipulation as mp 

import modelnormalize as nz

assert 1==1


from modelclass import model


#%% functions 
@dataclass
class GrapUNModel():
    '''This class takes a UN specification, variable data and variable description
    and transform it to ModelFlow business language'''
    
    
    frml      : str = 'model/mod_text.txt'            # path to model 
    data      : str = 'data/model_data.xlsx'          # path to data 
    modelname : str = 'Asia'           # modelname
    start     : int = 2017
    end       : int = 2050 
    country_trans   : any = lambda x:x[:].replace('$','_DOLLAR')    # function which transform model specification
    country_df_trans : any = lambda x:x     # function which transforms initial dataframe 
    
    
    def __post_init__(self):
        # breakpoint()
        
        print(f'\nProcessing the model:{self.modelname}',flush=True)
        self.rawmodel_org_text = open(self.frml).read().upper()
        self.eview_names = set(re.findall('@[A-Z0-1_]*',self.rawmodel_org_text))  
        raw_list =    [l for l in self.rawmodel_org_text.split('\n') ]   
        select = lambda l:  len(l.strip()) and  not '@INNOV'  in l and '@ADD' not in l 
        raw_eq     =    [l for l in raw_list if select(l)     ]   
        noise_list =     [l for l in raw_list if not select(l) ]     
        add_list   =     [l.split() for l in raw_list if '@ADD' in l]
        add_vars   =   {endo for add,endo,add_factor in add_list}
        self.rawmodel_org = '\n'.join(raw_eq)
        self.rawmodel = self.country_trans(self.rawmodel_org)
        # rawmodel6 = self.trans_eviews(self.rawmodel)
        rawmodel6 = '\n'.join(self.trans_eviews(r) for r in self.rawmodel.split('\n'))
        line_type = []
        line =[] 
        # breakpoint()
        bars = '{desc}: {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}'
        with tqdm(total=len(rawmodel6.split('\n')),desc='Reading original model',bar_format=bars) as pbar:
            for l in rawmodel6.split('\n'):
                # the logic says this
                #line_type.append('iden'if l.startswith('@IDENTITY ') else 'stoc' )
                #but equations like USA_INT says this 
                line_type.append('stoc' )
                line.append(l.replace('@IDENTITY ',''))
                    # print(f' {sec} {l[:30]} ....')
                pbar.update(1)    
                    
        self.all_frml = [nz.normal(l,add_adjust=(typ=='stoc')) for l,typ in tqdm(zip(line,line_type),desc='Normalizing model',total=len(line),bar_format=bars)]
        lfname = ["<Z,EXO> " if typ == 'stoc' else '' for typ in line_type ]
        lfname = ["" if typ == 'stoc' else '' for typ in line_type ]
        self.rorg = [fname + f.normalized for f,fname in zip(self.all_frml,lfname) ]
                
        self.rres = [f.calc_adjustment for f in self.all_frml if len(f.calc_adjustment)]
        # breakpoint()
        self.fmodel = mp.exounroll(mp.tofrml ('\n'.join(self.rorg)))
        self.fres =   ('\n'.join(self.rres))
        self.mmodel = model(self.fmodel,modelname = self.modelname)
        self.mmodel.set_var_description(self.asia_des)
        self.mres = model(self.fres,modelname = f'Adjustment factors for {self.modelname}')
        # breakpoint()
        self.base_input = self.mres.res(self.dfmodel,self.start,self.end)
 
    @functools.cached_property
    def dfmodel(self):
                '''The original input data enriched with during variablees, variables containing 
                values for specific historic years and model specific transformation '''
                # Now the data 
                df = (pd.read_excel(self.data).
                      pipe( lambda df : df.rename(columns={c:c.upper() for c in df.columns}))
                      .pipe( lambda df : df[[c for c in df.columns if not c+'_0' in df.columns]])
                      .pipe( lambda df : df.rename(columns = {c : c[:-2] if c.endswith('_0') else c for c in df.columns}))
                      .pipe( lambda df : df.rename(columns={'UNNAMED: 0':'DATEID'}))
                      .pipe( lambda df : df.set_index('DATEID'))
                      .pipe( lambda df : df.rename(columns = {c : c.replace('$','_DOLLAR')          for c in df.columns}))
                      )
                
                
                try:
                    sca = pd.read_excel(self.scalars ,index_col=0,header=None).T.pipe(
                        lambda _df : _df.loc[_df.index.repeat(len(df.index)),:]).\
                        set_index(df.index)
                    df= pd.concat([df,sca],axis=1)    
                except: 
                    print(f'{self.modelname} no Scalars prowided ')
                    
                #% Now set the vars with fixedvalues 
                value_vars = self.mmodel.vlist('*_value_*')
                for var,val,year in (v.rsplit('_',2) for v in value_vars) : 
                    df.loc[:,f'{var}_{val}_{year}'] = df.loc[int(year),var]
                self.showvaluevars = df[value_vars] 
                
                #% now set the values of the dummies   
                during_vars = self.mmodel.vlist('*during_*')
                for var,(dur,per) in ((v,v.split('_',1)) for v in during_vars):
                    df.loc[:,var]=0
                    # print(var,dur,per)
                    pers = per.split('_')
                    if len(pers) == 1:
                        df.loc[int(pers[0]),var] = 1
                    else:
                        df.loc[int(pers[0]):int(pers[1]),var]=1.
                self.showduringvars = df[during_vars] 
                # breakpoint()
                df_out = self.mmodel.insertModelVar(df).pipe(self.country_df_trans)
                return df_out.loc[:2050,:].copy()
            
            
    def __call__(self):

        return self.mmodel,self.base_input
    
    def test_model(self,start=None,end=None,maxvar=1_000_000, maxerr=100,tol=0.0001,showall=False):
        '''
        Compares a straight calculation with the input dataframe. 
        
        shows which variables dont have the same value 

        Args:
            df (TYPE): dataframe to run.
            start (TYPE, optional): start period. Defaults to None.
            end (TYPE, optional): end period. Defaults to None.
            maxvar (TYPE, optional): how many variables are to be chekked. Defaults to 1_000_000.
            maxerr (TYPE, optional): how many errors to check Defaults to 100.
            tol (TYPE, optional): check for absolute value of difference. Defaults to 0.0001.
            showall (TYPE, optional): show more . Defaults to False.

        Returns:
            None.

        '''
        _start = start if start else self.start
        _end    = end if end else self.end
        # breakpoint()
    
        resresult = self.mmodel(self.base_input,_start,_end,reset_options=True,silent=0,solver='base_res')
        self.mmodel.basedf = self.dfmodel
        pd.options.display.float_format = '{:.10f}'.format
        err=0
        print(f'\nChekking residuals for {self.mmodel.name} {_start} to {_end}')
        for i,v in enumerate(self.mmodel.solveorder):
            if i > maxvar : break
            if err > maxerr : break
            check = self.mmodel.get_values(v,pct=True).T
            check.columns = ['Before check','After calculation','Difference','Pct']
            # breakpoint()
            if (check.Difference.abs() >= tol).any():                
                err=err+1
                maxdiff = check.Difference.abs().max()
                maxpct  = check.Pct.abs().max()
                # breakpoint()
                print('\nVariable with residuals above threshold')
                print(f"{v}, Max difference:{maxdiff:15.8f} Max Pct {maxpct:15.10f}% It is number {i} in the solveorder and error number {err}")
                if showall:
                    print(f'\n{self.mmodel.allvar[v]["frml"]}')
                    print(f'\nResult of equation \n {check}')
                    print(f'\nEquation values before calculations: \n {self.mmodel.get_eq_values(v,last=False,showvar=1)} \n')
        self.mmodel.oldkwargs = {}
        
     
       
    @property 
    def var_description(self):
        '''
        '''
        
        if isinstance(self.des,dict):
            return self.des
        
        try:
            trans0 = pd.read_excel(self.des).loc[:,['mnem','Excel']].set_index('mnem').to_dict(orient = 'dict')['Excel']
            var_description = {str(k) : str(v).strip() for k,v in trans0.items() if 'nan' != str(v)}
        except:
            print('*** No variable description',flush=True)
            var_description = {}
        return var_description    
           
        
    @staticmethod
    def trans_eviews(rawmodel):
        # breakpoint()
        rawmodel0 = '\n'.join(l for l in rawmodel.upper().split('\n') if len(l.strip()) >=2)
        # trailing and leading "
        rawmodel1 = '\n'.join(l[1:-1] if l.startswith('"') else l for l in rawmodel0.split('\n'))
        # powers
        rawmodel2 = rawmodel1.replace('^','**').replace('""',' ').replace('"',' ').\
            replace('@EXP','exp').replace('@RECODE','recode').replace('@MOVAV','movavg').replace('@LOGIT(','LOGIT(') \
            .replace('@MEAN(@PC(','@AVERAGE_GROWTH((').replace('@PC','PCT_GROWTH').replace('@QGAMMA','QGAMMA').replace('@CLOGNORM','CLOGNORM')\
            .replace('@TREND','TREND')
        # @ELEM and @DURING 
        # @ELEM and @DURING 
        rawmodel3 = nz.elem_trans(rawmodel2) 
        rawmodel4 = re.sub(r'@DURING\( *([0-9]+) *\)', r'during_\1',rawmodel3) 
        rawmodel5 = re.sub(r'@DURING\( *([0-9]+) *([0-9]+) *\)', r'during_\1_\2',rawmodel4) 
        
        # during check 
        ldur = '\n'.join(l for l in rawmodel5.split('\n') if '@DURING' in l)
        ldur2 = '\n'.join(l for l in rawmodel5.split('\n') if 'during' in l)
        
        # check D( 
        ld  = '\n'.join(l for l in rawmodel5.split('\n') if re.search(r'([^A-Z]|^)D\(',l) )
        ld1  = '\n'.join(l for l in rawmodel5.split('\n') if re.search(r'([^A-Z0-9_]|^)D\(',l) )
        # breakpoint()
        rawmodel6 = nz.funk_replace('D','DIFF',rawmodel5) 
        # did we get all the lines 
        ldif = '\n'.join(l for l in rawmodel6.split('\n') if 'DIFF(' in l )
        return rawmodel6

    @functools.cached_property
    def asia_des(self):
        des = (pd.read_excel(self.data,sheet_name='Variable definitions',header=None).
              rename(columns={0:'var_template',1:'des1',2:'des2'}))
        # breakpoint()
        try:
            des['des'] = des.des1.str.cat(des.des2.astype(str).str.replace('nan',''))
        except:
            des['des'] = des.des1
            
        des = des[['var_template','des']]
        des_dict = {self.country_trans(var_template.strip()) : des.replace('CO2','$CO^2$') for var_template,des in [row.tolist() for i,row in des.iterrows()]}
        
        iso = (pd.read_excel(self.data,sheet_name='Country prefixes').
               iloc[1:,:])
        iso_dict = {iso:country for country,iso in [row.tolist() for i,row in iso.iterrows()]}
        #%%
        # for iso,country in iso_dict.items():
        #     print(iso,country)
            
        #     for var_template,des in des_dict.items():
        #         print(var_template.replace('{ISO}',iso).replace('$','_DOLLAR'),f'{des} ,{country}')
                
        dict2=   {var_template.replace('{ISO}',iso):f'{des} ,{country}'
        for var_template,des in des_dict.items()
        for iso,country in iso_dict.items() 
        }   
        return dict2
    
# os.environ['PYTHONBREAKPOINT']=None
asia = GrapUNModel(data='data/model_data_7Sept.xlsx',
                   frml='model/modtext_7sept.txt',
                   modelname='Asia_7sept')
asia.test_model(2020,2050,maxerr=200,tol=0.000001,showall=1)

#%%
def var_description_addon(self):
    '''Adds vardescriptions for add factors, exogenizing dummies and exoggenizing values '''
    add_d =   { newname : 'Add factor:'+ self.var_description.get(v,v)      for v in self.endogene if  (newname := v+'_A') in self.exogene }
    dummy_d = { newname : 'Exo dummy:'+ self.var_description.get(v,v)  for v in self.endogene if  (newname := v+'_D')  in self.exogene }
    exo_d =   { newname : 'Exo value:'+ self.var_description.get(v,v)      for v in self.endogene if  (newname := v+'_X')  in self.exogene }
    self.set_var_description({**self.var_description,**add_d,**dummy_d,**exo_d})
model.var_description_addon = var_description_addon
masia = asia.mmodel
masia.var_description_addon()
baseline = asia.base_input.copy()
share_countries  =[v.split('_')[0] for v in masia['*_sharex'].names]
sharenames = [f'{country}_{sharename}_A'.upper() for country in share_countries for sharename in ['sharee','sharex','shareh','sharesp']]
baseline[sharenames] = 0.25 
masia.basedf = baseline
_ = masia(baseline,2021,2050,silent=0)
masia['*_GDI'].dif.plot(sharey=0,colrow=1)

masia.modeldump('asia/Asia_sep7.pcim')
#%% experiment
altdf = baseline.copy()
altdf.loc[2021:2021,'KHM_GCARBR_A'] = altdf.loc[2021:2021,'KHM_GCARBR_A'] + 20  

altres = masia(altdf,2021,2050,silent=0,first_test = 5,ljit=0)
masia['*_GDI'].dif.plot(colrow=1)
masia.KHM_GCARBR
masia.KHM_PREM
#%%
    