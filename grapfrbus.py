# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 15:15:33 2021

@author: ibhan
"""
import xml
import pandas as pd
import re
from dataclasses import dataclass
import functools
from tqdm import tqdm 


from modelclass import model 

import modelnormalize as nz


@dataclass
class GrapFRBModel():
    '''This class takes a world bank model specification, variable data and variable description
    and transform it to ModelFlow business language'''
    
    
    frml      : str = r'..\frbus_package\frbus_package\mods\model.xml'            # path to model 
    data      : str = ''          # path to data 
    des       : any = ''            # path to descriptions
    scalars   : str = ''           # path to scalars 
    modelname : str = 'Frb/US'           # modelname
    start     : int = 2017
    end       : int = 2040 
    country_trans   : any = lambda x:x[:]    # function which transform model specification
    country_df_trans : any = lambda x:x     # function which transforms initial dataframe 
    
    
    def __post_init__(self):
        # breakpoint()
        frbustrans = lambda x : x.replace('@record','record')
        print(f'\nProcessing the model:{self.modelname}',flush=True)
        def getinf(y): 
            ''' Extrct informations on a variables and equations 
            in the FRB/US model'''
            name = y.find('name').text
            definition = y.find('definition').text
            type = y.find('equation_type').text
            
        #    print('name:',name)
            if y.find('standard_equation'):
                neq= y.find('standard_equation/eviews_equation').text.replace('\n','').replace(' ','').replace('@recode','record')
                cdic = {coeff.find('cf_name').text : coeff.find('cf_value').text for coeff in y.findall('standard_equation/coeff')}
                # if 'y_dmptlur(1)' in cdic:
                #     print(cdic)
                #     print(neq)
                #     cdic['y_dmptlur(1)']='-25'  # don't remember why, so ditched this year
                
                for c,v in cdic.items():
                    neq=neq.replace(c,v)
        #        print(neq)
            else:
                neq=''
                cdic={}
                
            if y.find('mce_equation'):
                meq = y.find('mce_equation/eviews_equation').text.replace('\n','').replace(' ','').replace('@recode','record')
                cmdic = {coeff.find('cf_name').text : coeff.find('cf_value').text for coeff in y.findall('mce_equation/coeff')}
                for c,v in cmdic.items():
                    meq=meq.replace(c,v)
            else:
                meq=''
                cmdic={}
        #        print(meq)
            return name,{'definition':definition, 'type':type , 'neq':neq, 'cdic':cdic, 'meq':meq}
            
               
        with open(self.frml,'rt') as t:
            tfrbus = t.read()
            
        rfrbus = xml.etree.ElementTree.fromstring(tfrbus)
        
        frbusdic = {name.upper():inf for name,inf in (getinf(y) for y in rfrbus.iterfind('variable'))}
        # breakpoint()
        var_typedic = {v : d['type'] for v,d in frbusdic.items()}
        self.var_description = {v.upper():inf['definition'] for v,inf in frbusdic.items()}
        
        var_ffrbus = {v.upper(): d['neq'].upper().replace('\n',' ')  for v,d in frbusdic.items() if d['neq']}  # first extract of definition
        mce_ffrbus = {v.upper(): d['meq'].upper().replace('\n',' ') if d['meq'] else d['neq'].upper().replace('\n',' ')  for v,d in frbusdic.items() if d['neq']}  # first extract of definition
            
        bars = '{desc}: {percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}'
         
        var_normalized = {v: nz.normal(l,the_endo=v, add_adjust=var_typedic[v] == 'Behavioral') for v,l in tqdm(var_ffrbus.items() ,
                                                          desc=f'Normalizing {self.modelname} VAR version',total=len(var_ffrbus),bar_format=bars)}
        mce_normalized  = {v: nz.normal(l,the_endo=v, add_adjust=var_typedic[v] == 'Behavioral') for v,l in tqdm(mce_ffrbus.items() ,
                                                          desc=f'Normalizing {self.modelname} MCE version',total=len(mce_ffrbus),bar_format=bars)} 

        var_frml = '\n'.join( [("<Z> " if var_typedic[v] == 'Behavioral' else '')  + f.normalized  for v,f in var_normalized.items()  ])
        mce_frml =  '\n'.join([("<Z> " if var_typedic[v] == 'Behavioral' else '')  + f.normalized  for v,f in mce_normalized.items()  ])
                
        var_res_frml = '\n'.join([f.calc_adjustment for v,f in var_normalized.items() if len(f.calc_adjustment)])
        mce_res_frml = '\n'.join([f.calc_adjustment for v,f in mce_normalized.items() if len(f.calc_adjustment)])
        
        self.mmodel_var = model(var_frml,modelname = f'{self.modelname} VAR version')
        self.mmodel_mce = model(mce_frml,modelname = f'{self.modelname} MCE version')
        self.mmodel_var_res = model(var_res_frml,modelname = f'{self.modelname} VAR version calculation of residuals')
        self.mmodel_mce_res  = model(mce_res_frml,modelname =  f'{self.modelname} MCE version calculation of residuals')
        
        for mmodel in [self.mmodel_var,self.mmodel_mce,  self.mmodel_var_res,  self.mmodel_mce_res ] :
            mmodel.set_var_description(self.var_description)
        breakpoint()
        return 
    
       
    @functools.cached_property
    def dfmodel(self):
        '''The original input data enriched with during variablees, variables containing 
        values for specific historic years and model specific transformation '''
        # Now the data 
        df = (pd.read_excel(self.data).
              pipe( lambda df : df.rename(columns={c:c.upper() for c in df.columns})).
              pipe( lambda df : df.rename(columns={'_DATE_':'DATEID'})).
              pipe( lambda df : df.set_index('DATEID'))
              )
        df.index = [int(i.year) for i in df.index]
        
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
        return df_out
    
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
        
if __name__ == '__main__':
    frbus = GrapFRBModel()
    
