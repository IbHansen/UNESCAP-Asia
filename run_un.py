# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 13:41:40 2021

@author: bruger
"""




from modelclass import model 


masia,baseline = model.modelload('asia/Asia.pcim',run=1)
#%% experiment
altdf = baseline.copy()
altdf.loc[2021:2021,'KHM_GCARBR_A'] = altdf.loc[2021:2021,'KHM_GCARBR_A'] + 20  
for exovar in 'KHM_PREM KHM_EXPE KHM_EXPH KHM_EXPSP KHM_OGC KHM_OGI'.split():
        altdf.loc[2021:2050,exovar+'_D'] = 1
        altdf.loc[2021:2050,exovar+'_X'] = altdf.loc[2021:2050,exovar]
        

altres = masia(altdf,2021,2050,silent=0,first_test = 4,ljit=0,solver='newton',forcenum=1)
masia['KHM_GDI KHM_GCARBR'].dif.plot(colrow=1,sharey=0)
masia.KHM_GCARBR
masia.KHM_PREM
#%%
masia.FSM_YFT
