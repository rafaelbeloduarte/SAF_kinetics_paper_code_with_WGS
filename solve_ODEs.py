#!/usr/bin/env python
# coding: utf-8

from math import e
from PBR import PBR
from Reaction import Reaction
from KineticModel import KineticModel
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from IPython.display import clear_output
import multiprocessing

def solve_ODEs(kinetic_data, model_name, param_dict):
    for i in kinetic_data.index:
        print(f'Running simulation on index {i} of {kinetic_data.index.max()}')
        clear_output(wait = True)
        F0 = kinetic_data.loc[i, 'F_H2_e_mol_s'] + kinetic_data.loc[i, 'F_CO_e_mol_s']
    
        H2_CO = kinetic_data.loc[i, '$H_2/CO$']
        x_CO = 1 / ( 1 + H2_CO )
        x_H2 = 1 - x_CO
    
        # create reactor object
        T0 = (kinetic_data.loc[i, 'T_R_K'], 'K')
        x0 = {
                          'carbon monoxide': x_CO,
                          'hydrogen' : x_H2,
                      }
        reactor = PBR(T0 = T0, 
                      P0 = (kinetic_data.loc[i, 'P_abs_Pa'], 'Pa'), 
                      x0 = x0, 
                      F0 = (F0, 'mol/s'),
                      isothermal = True, 
                      ergun = False,
                      WT = (kinetic_data.loc[i, 'm_cat_g'], 'g'), 
                      W_steps = 50,
                      U = (300, 'W/m2/K'),
                      D = (7, 'mm'),
                      Dp = (0.1725, 'mm'),
                      pellet_density = (2530, 'kg/m3'),
                      Ta = T0,
                      liquid_phase = False,
                      units = 'SI',
                      solver_method = 'BDF',
                      rtol = 1e-5, atol = 1e-7,
                      calculate_equilibrium = False,
                      equilibrium_obj_fun_tolerance = 1e-12,
                      log = True,
                     )
    
        match model_name:
            case 'PowerLaw':
                from models_FTS.PowerLaw         import PowerLaw
                PowerLaw = PowerLaw()
                PowerLaw.kin_param_dict = param_dict
                reactor.add_model(PowerLaw)
            case 'Yates':
                from models_FTS.Yates            import Yates
                Yates = Yates()
                Yates.kin_param_dict = param_dict
                reactor.add_model(Yates)
            case 'Botes':
                from models_FTS.Botes            import Botes
                Botes = Botes()
                Botes.kin_param_dict = param_dict
                reactor.add_model(Botes)
            case 'Ojeda':
                from models_FTS.Ojeda            import Ojeda
                Ojeda = Ojeda()
                Ojeda.kin_param_dict = param_dict
                reactor.add_model(Ojeda)
            case 'Mousavi':
                from models_FTS.Mousavi          import Mousavi
                Mousavi = Mousavi()
                Mousavi.kin_param_dict = param_dict
                reactor.add_model(Mousavi)
            case 'PowerLaw2':
                from models_FTS.PowerLaw2        import PowerLaw2
                PowerLaw2 = PowerLaw2()
                PowerLaw2.kin_param_dict = param_dict
                reactor.add_model(PowerLaw2)
        reactor.solve()
        print(f'Params {model_name}: {reactor.model.kin_param_dict}')
    
        sol_df = reactor.sol_dfs[model_name]
    
        F_H2_out_model = sol_df.loc[(sol_df.W == sol_df.W.max()) & (sol_df.variable == 'hydrogen')].value.values[-1]
        F_CO_out_model = sol_df.loc[(sol_df.W == sol_df.W.max()) & (sol_df.variable == 'carbon monoxide')].value.values[-1]
    
        kinetic_data.loc[i, f'F_H2_out_model_{model_name}'] = F_H2_out_model
        kinetic_data.loc[i, f'F_CO_out_model_{model_name}'] = F_CO_out_model
    
    
    kinetic_data[f'X_H2_model_{model_name}'] = (kinetic_data.F_H2_e_mol_s 
                                                - kinetic_data[f'F_H2_out_model_{model_name}']) / kinetic_data.F_H2_e_mol_s 
    
    kinetic_data[f'X_CO_model_{model_name}'] = (kinetic_data.F_CO_e_mol_s 
                                                - kinetic_data[f'F_CO_out_model_{model_name}']) / kinetic_data.F_CO_e_mol_s 
    return kinetic_data