#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import win32com.client as win32
from y_n import y_n
from upsilon_n import upsilon_n
import datetime
import time
import numpy as np


# if __name__ == "__main__":

# In[2]:


parafins = [ 'methane', 'ethane', 'propane', 
             'butane', 'pentane', 'hexane', 
               'heptane', 'octane', 
             'nonane', 'decane', 'undecane', 'dodecane',
               'tridecane', 'tetradecane', 'pentadecane', 'hexadecane', 'heptadecane',
               'octadecane', 'nonadecane', 'eicosane', 'heneicosane', 'docosane',
               'tricosane', 'tetracosane', 'pentacosane',
               'hexacosane', 'heptacosane',
               'octacosane', 'nonacosane', 'triacontane', 
             'hentriacontane', 'dotriacontane',
               'tritriacontane', 'tetratriacontane', 'pentatriacontane', 'hexatriacontane'
            ]

olefins = [ 'ethylene', 
            'propene',
            '1-butene', '1-pentene', '1-hexene',
            '1-heptene', '1-octene', 
            '1-nonene', '1-decene',
            '1-undecene', '1-dodecene',
            '1-tridecene', '1-tetradecene', '1-pentadecene',
            '1-hexadecene', '1-heptadecene', '1-octadecene',
          ]

others = ['carbon monoxide', 'hydrogen', 'water']


# In[3]:


hysys = win32.Dispatch("HYSYS.Application")


# In[4]:


hy_case = hysys.Application.ActiveDocument
hy_case.Visible = 1


# In[5]:


hy_solver = hy_case.Solver

hy_f = hy_case.Flowsheet        

hy_ms = hy_f.MaterialStreams
hy_es = hy_f.EnergyStreams

reactor = hy_case.Flowsheet.Operations.Item('PFR-100')


# In[6]:


n_parafins = {}
for i, parafin in enumerate(parafins):
    n = i + 1
    if n not in [4, 5, 6, 7, 8, 31, 32, 33, 34, 35, 36]:
        n_parafins[parafin] = n

n_olefins = {}
for i, olefin in enumerate(olefins):
    n = i + 2
    if n not in [3, 4, 5, 6, 7, 8]:
        n_olefins[olefin] = n

N = list(set(list(n_parafins.values()) + list(n_olefins.values())))


# In[7]:


rxn_set = hy_case.BasisManager.ReactionPackageManager.ReactionSets.Item('FTS')

F2 = hy_ms.Item('F2')

T = F2.TemperatureValue + 273.15

alpha = float(np.load('param_alpha.npy'))

ALPHA_SHEET = hy_case.Flowsheet.Operations.Item('ALPHA')

ALPHA_SHEET.Cell(0,0).CellValue = alpha

# must multiply reaction rate by correction factor below
# because hysys bases reaction rate only on the gas phase, not the catalyst volume
vol_cat_to_vol_gas =  (1 / reactor.VoidFraction) - 1 # m³ cat / m³ gas
overall_freq_fact = 685.4303638787276 # mol / s / kg cat / Pa²
packing = 1157 # kg / m³
kmol_to_mol = 1 / 1000 # kmol / mol
overall_freq_fact = overall_freq_fact * packing * kmol_to_mol # kmol / s / m³ cat / Pa²

for component, n in n_parafins.items():
    freq_fact = n * y_n(n) * upsilon_n(n, T) * overall_freq_fact * vol_cat_to_vol_gas # kmol / s / m³ gas / Pa²
    rxn_set.ReactionPackage.Reactions.Item(component).ForwardFrequencyFactor = freq_fact
    print(f'updating rate for {component}')

for component, n in n_olefins.items():
    freq_fact = n * y_n(n) * ( 1 - upsilon_n(n, T) ) * overall_freq_fact * vol_cat_to_vol_gas # kmol / s / m³ / Pa²
    rxn_set.ReactionPackage.Reactions.Item(component).ForwardFrequencyFactor = freq_fact
    print(f'updating rate for {component}')

