#!/usr/bin/env python
# coding: utf-8

# In[1]:


import clr  # From pythonnet
import os
import random
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Paths
dwsim_path = r"C:\Users\rafael\AppData\Local\DWSIM\\"
sim_path = r"Z:\uem\Doutorado\Ensaios\SAF\SAF_8_08_08_2025\SAF_kinetics_paper_code_with_WGS\dwsim_bench_model.dwxmz"


# In[3]:


# 1. Setup paths to DWSIM installation
clr.AddReference(os.path.join(dwsim_path, "DWSIM.Automation.dll"))
clr.AddReference(os.path.join(dwsim_path, "DWSIM.Interfaces.dll"))

from DWSIM.Automation import Automation3
from System import String

# 2. Initialize the Automation Manager
interf = Automation3()

# 3. Load an existing simulation (.dwxmz)
Flowsheet = interf.LoadFlowsheet(sim_path)


# In[4]:


# get flowsheet objects
H2 = Flowsheet.GetFlowsheetSimulationObject('H2').GetAsObject()
CO = Flowsheet.GetFlowsheetSimulationObject('CO').GetAsObject()
PFR_1 = Flowsheet.GetFlowsheetSimulationObject('PFR-1').GetAsObject()
syncrude = Flowsheet.GetFlowsheetSimulationObject('syncrude').GetAsObject()
compressor = Flowsheet.GetFlowsheetSimulationObject('C-1').GetAsObject()
cooler = Flowsheet.GetFlowsheetSimulationObject('CL-1').GetAsObject()
C1_C8 = Flowsheet.GetFlowsheetSimulationObject('C1-C8').GetAsObject()
C9_C15 = Flowsheet.GetFlowsheetSimulationObject('C9-C15').GetAsObject()
C16p = Flowsheet.GetFlowsheetSimulationObject('C16p').GetAsObject()
unreacted_syngas = Flowsheet.GetFlowsheetSimulationObject('unreacted syngas').GetAsObject()
water = Flowsheet.GetFlowsheetSimulationObject('H2O').GetAsObject()
CO2 = Flowsheet.GetFlowsheetSimulationObject('CO2').GetAsObject()
syncrude = Flowsheet.GetFlowsheetSimulationObject('syncrude').GetAsObject()
syncrude_phase = syncrude.GetPhase('Overall')
E_reactor = Flowsheet.GetFlowsheetSimulationObject('E1').GetAsObject()

PFR_1 = Flowsheet.GetFlowsheetSimulationObject('PFR-1').GetAsObject()
PFR_1.set_dV(0.5)


# In[5]:


v0 = 180 # mL / min
v0 = v0 / ( 1000 * 1000 * 60 ) # m3 / s

F0 = v0 * 1e5 / ( 8.314 * 273 ) # mol / s

H2_CO_in = 2
F_H2_in = F0 * (H2_CO_in / ( 1 + H2_CO_in )) # mol / s
F_CO_in = F0 - F_H2_in

H2.SetMolarFlow(F_H2_in)
CO.SetMolarFlow(F_CO_in)
compressor.POut = 2e6
cooler.OutletTemperature = 240 + 273

# Ligar o solver
interf.CalculateFlowsheet2(Flowsheet)

# check if solved
if not Flowsheet.Solved:
    interf.SaveFlowsheet(Flowsheet, sim_path, True)
    raise ValueError(f'Something went wrong at index {i}, check your simulation.')

interf.SaveFlowsheet(Flowsheet, sim_path, True)


# In[6]:


E_reactor.GetEnergyFlow()

