# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 14:46:11 2021

To inject methods and properties into a asia model instance, so we dont have to recreate it

@author: bruger
"""

import pandas as pd
import  ipywidgets as widgets  
from IPython.display import display, clear_output,Latex, Markdown


from modelclass import insertModelVar
from modeljupyter import get_alt_dic, jupviz

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
    country_dict_GCARBR_A = {country:v for country,v in self.iso_dict.items() if f'{country}_GCARBR_A' in self.exogene}
    
    self.countries_GCARBR_A  = [c for c in country_dict_GCARBR_A.keys()]
    self.countries_ASEAN = 'BRN KHM IDN LAO MYS MMR PHL SGP THA VNM'.split()

    #breakpoint()
    setattr(self.__class__,'inputwidget_asia', inputwidget_asia)
    
    self.nametrans = nametrans
    
def nametrans(varnames,thismodel):
    out = [thismodel.iso_dict.get(cname := c.split('_')[0],cname) for c in varnames]
    return out
                              
def inputwidget_asia(self, start='', slut='', basedf=None, **kwargs):
    ''' calls modeljupyter input widget, and keeps the period scope '''
    if type(basedf) == type(None):
        with self.set_smpl(start=start, slut=slut):
            return inputwidget(self, self.basedf, **kwargs)
    else:
        tempdf = insertModelVar(basedf, self)
        self.basedf = tempdf
        with self.set_smpl(start=start, slut=slut):
            return inputwidget(self, self.basedf, **kwargs)


def inputwidget(model,basedf,slidedef={},radiodef=[],checkdef=[],modelopt={},varpat='RFF XGDPN RFFMIN GFSRPN DMPTRSH XXIBDUMMY'
                 ,showout=1,trans=None,base1name='',alt1name='',go_now=True,showvar=False):
    '''Creates an input widgets for updating variables 
    
    :df: Baseline dataframe 
    :slidedef: dict with definition of variables to be updated by slider
    :radiodef: dict of lists. each at first level defines a collection of radiobuttoms
               second level defines the text for each leved and the variable to set or reset to 0
    :varpat: the variables to show in the output widget
    :showout: 1 if the output widget is to be called '''
    
    if type(trans) == type(None):
        thistrans = model.var_description
    else:
        thistrans = trans 
    
    lradiodef= len(radiodef)
    lslidedef = len(slidedef)
    lcheckdef = len(checkdef)
    basename = base1name if base1name else 'Baseline' 
    altname = alt1name if alt1name else 'Alternative' 
    
    if lradiodef: 
        wradiolist = [widgets.RadioButtons(options=[i for i,j in cont],description=des,layout={'width':'70%'},
                                           style={'description_width':'37%'}) for des,cont in radiodef.items()]
        if len(wradiolist) <=2:
            wradio = widgets.HBox(wradiolist)
        else: 
            wradio = widgets.VBox(wradiolist)

            

# define slidesets 
    if lslidedef:     
        wexp  = widgets.Label(value="Carbon tax rate, US$ per tonn ",layout={'width':'54%'})
        walt  = widgets.Label(value=f'{altname}',layout={'width':'10%'})
        wbas  = widgets.Label(value=f'{basename}',layout={'width':'20%'})
        whead = widgets.HBox([wexp,walt,wbas])
        
        wset  = [widgets.FloatSlider(description=des,
                                    min=cont['min'],max=cont['max'],value=cont['value'],step=cont.get('step',0.01),
                                    layout={'width':'60%'},style={'description_width':'40%'},readout_format = f":<,.{cont.get('dec',2)}f")
                 for des,cont in slidedef.items()]
        
        
        waltval= [widgets.Label(value=f"{cont['value']:<,.{cont.get('dec',2)}f}",layout={'width':'10%'})
                  for des,cont  in slidedef.items()]
        wslide = [widgets.HBox([s,v]) for s,v in zip(wset,waltval)]
       
# cheklist  
    if lcheckdef:         
        wchecklist = [widgets.Checkbox(description=des,value=val)   for des,var,val in checkdef]
        wcheck  = widgets.HBox(wchecklist)   

# some buttons and text     
    wname = widgets.Text(value=basename,placeholder='Type something',description='Scenario name:',
                        layout={'width':'30%'},style={'description_width':'50%'})
    wpat = widgets.Text(value=varpat,placeholder='Type something',description='Output variables:',
                        layout={'width':'65%'},style={'description_width':'30%'})
    if showvar:
        wpat.layout.visibility = 'visible'
    else:
        wpat.layout.visibility = 'hidden'
        

    winputstring = widgets.HBox([wname,wpat])
    
    wgo   = widgets.Button(description="Run scenario")
    wreset   = widgets.Button(description="Reset to start")
    wzero   = widgets.Button(description="Set all to 0")
    wsetbas   = widgets.Button(description="Use as baseline")
    wbut  = widgets.HBox([wgo,wreset,wzero,wsetbas])
    
    wvar = [whead]+wslide if lslidedef else []
    if lradiodef: wvar = wvar + [wradio]
    if lcheckdef: wvar = wvar + [wcheck]
        
    w     = widgets.VBox(wvar+[winputstring] +[wbut])

    # This function is run when the button is clecked 
    firstrun = True
    model.inputwidget_alternativerun = False

    model.rundic = {}
    
    def run(b):
        nonlocal firstrun
        nonlocal altname
        # mulstart       = model.basedf.copy()
        mulstart       = insertModelVar(basedf.copy(deep=True),model)
        # model.smpl(df=mulstart)
        
        # First update from the sliders 
        if lslidedef:
            for i,(des,cont) in enumerate(slidedef.items()):
                op = cont.get('op','=')
                var = cont['var']
                for var in cont['var'].split():
                    if  op == '+':
                        mulstart.loc[model.current_per,var]    =  mulstart.loc[model.current_per,var] + wset[i].value
                    elif op == '+impulse':    
                        mulstart.loc[model.current_per[0],var] =  mulstart.loc[model.current_per[0],var] + wset[i].value
                    elif op == '=start-':   
                        startindex = mulstart.index.get_loc(model.current_per[0])
                        varloc = mulstart.columns.get_loc(var)
                        mulstart.iloc[:startindex,varloc] =  wset[i].value
                    elif op == '=':    
                        mulstart.loc[model.current_per,var]    =   wset[i].value
                    elif op == '=impulse':    
                        mulstart.loc[model.current_per[0],var] =   wset[i].value
                    else:
                        print(f'Wrong operator in {cont}.\nNot updated')
                        assert 1==3,'wRONG OPERATOR'
                
        # now  update from the radio buttons 
        if lradiodef:
            for wradio,(des,cont) in zip(wradiolist,radiodef.items()):
                # print(des,wradio.value,wradio.index,cont[wradio.index])
                for v in cont:
                    mulstart.loc[model.current_per,v[1]] = 0.0
                mulstart.loc[model.current_per,cont[wradio.index][1]] = 1.0  
 
        if lcheckdef:           
            for box,(des,var,_) in zip(wchecklist,checkdef):
                mulstart.loc[model.current_per,var] = 1.0 * box.value

        #with out:
        clear_output()
        display(w)
        display(widgets.Label(value="Solving model ",layout={'width':'54%'}))
        mul = model(mulstart,**modelopt)
        # model.mulstart=mulstart


        clear_output()
        display(w)
        #_ = mfrbus['XGDPN RFF RFFMIN GFSRPN'].dif.rename(trans).plot(colrow=1,sharey=0)
        
        
        if firstrun:
            model.experiment_results = {}
            model.experiment_results[basename] = {'results':model.lastdf.copy()}
            firstrun = False
            model.inputwidget_firstrun = False
            wname.value = f'{altname}'
        else:
            altname = wname.value
            walt.value = f'{altname}'
            model.experiment_results[altname] = {'results':model.lastdf.copy()}
            model.inputwidget_alternativerun = True

        model.dekomp.cache_clear()
        if showout:
            varpat_this =  wpat.value
            resdic = get_alt_dic(model,varpat_this,model.experiment_results)
            a = jupviz(resdic,trans=thistrans)()
        else:  
            a = vis_alt4(get_alt_dic(model,wpat.value,model.experiment_results),model,trans=thistrans)

    def reset(b):

        if lslidedef:
            for i,(des,cont) in enumerate(slidedef.items()):
                wset[i].value  =   cont['value']
            
        if lradiodef:
            for wradio in wradiolist:
                wradio.index = 0
            
        if lcheckdef:           
            for box,(des,var,defvalue) in zip(wchecklist,checkdef):
                box.value = defvalue
                
    def zeroset(b):
        basename = base1name if base1name else 'Baseline' 
        walt.value = f'{altname}'
        wbas.value = f'{basename}'

        if lslidedef:
            for i,(des,cont) in enumerate(slidedef.items()):
                wset[i].value  =   type(cont['value'])(0.0)
            
        if lradiodef:
            for wradio in wradiolist:
                wradio.index = 0
            
        if lcheckdef:           
            for box,(des,var,defvalue) in zip(wchecklist,checkdef):
                box.value = defvalue


                

    def setbas(b):
        nonlocal basename
        nonlocal firstrun 
        model.basedf = model.lastdf.copy(deep=True)
        basename = wname.value
        walt.value = f'{altname}'
        wbas.value = f'{basename}'
        wname.value = f'{basename}'
        model.rundic = {}
        firstrun = True 
        
        if lslidedef:
            for i,(des,cont) in enumerate(slidedef.items()):
                waltval[i].value=  f"{cont['value']:<,.{cont.get('dec',2)}f}"
                
        if lradiodef:
            for wradio in wradiolist:
                wradio.index = 0
                
        if lcheckdef:           
            for box,(des,var,defvalue) in zip(wchecklist,checkdef):
                box.value = defvalue


    # Assign the function to the button  
    wgo.on_click(run)
    wreset.on_click(reset)
    wzero.on_click(zeroset)
    wsetbas.on_click(setbas)
    # out = widgets.Output()
    
    
    if go_now:
        run(None)
    return w


    
if __name__ == '__main__':
    from modelclass import model 
    if 'masia' not in locals(): 
        masia,baseline = model.modelload('Asia.pcim',run=1,silent=1)    
    inject(masia)

    masia.country_dict_GCARBR_A
