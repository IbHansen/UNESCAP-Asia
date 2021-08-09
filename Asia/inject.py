# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 14:46:11 2021

To inject methods and properties into a asia model instance, so we dont have to recreate it

@author: bruger
"""

import pandas as pd

assert 1==1
#%%
data      : str = '../data/model_data.xlsx'          # path to data 

def inject(self):
    des = (pd.read_excel(data,sheet_name='Variable definitions',header=None).
          rename(columns={0:'var_template',1:'des1',2:'des2'}))
    
    des['des'] = des.des1.str.cat(des.des2.astype(str).str.replace('nan',''))
    des = des[['var_template','des']]
    self.des_dict = {var_template.strip() : des for var_template,des in [row.tolist() for i,row in des.iterrows()]}
    
    iso = (pd.read_excel(data  ,sheet_name='Country prefixes').
           iloc[1:,:])
    self.iso_dict = {iso:country for country,iso in [row.tolist() for i,row in iso.iterrows()]}
    self.country_dict_GCARBR_A = {country:v for country,v in self.iso_dict.items() if f'{country}_GCARBR_A' in self.exogene}
    
    self.countries_GCARBR_A  = [c for c in self.country_dict_GCARBR_A.keys()]
    
if __name__ == '__main__':
    from modelclass import model 
    if 'masia' not in locals(): 
        masia,baseline = model.modelload('Asia.pcim',run=1,silent=1)    
    inject(masia)

    masia.country_dict_GCARBR_A
