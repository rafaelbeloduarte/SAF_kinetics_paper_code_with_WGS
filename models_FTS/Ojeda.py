import sys
from math import e
import numpy as np

# setting path
sys.path.append('..')
from KineticModel import KineticModel
from Reaction import Reaction

from models_FTS.components import parafins, olefins, others
from y_n import y_n
from upsilon_n import upsilon_n

def Ojeda():
    model_name = 'Ojeda'
    param_dict = {'k_ads': np.float64(2.838395313874119e-06),
  'H_ads': np.float64(-0.01643909412067914),
  'A_HCs_1': np.float64(403.67257240527465),
  'E_HCs_1': np.float64(195058.5873251934),
  'A_HCs_2': np.float64(1828.264747018951),
  'E_HCs_2': np.float64(114567.42827486034)}
    
    bnds = {
        'k_ads': (0, None),
        'H_ads': (0, None),
        'A_HCs_1': (0, None),
        'E_HCs_1': (0, None),
        'A_HCs_2': (0, None),
        'E_HCs_2': (0, None),
    }

    model = KineticModel(model_name, params = param_dict, bnds = bnds)

    n_parafins = {}
    for i, parafin in enumerate(parafins):
        n = i + 1
        n_parafins[parafin] = n
    
    n_olefins = {}
    for i, olefin in enumerate(olefins):
        n = i + 1
        n_olefins[olefin] = n
    
    for parafin in n_parafins:
        n = n_parafins[parafin]
        def rate_n_para(param_dict, T, P, x, F, K, eq_distance, R, data, n = n):
            P_H2 = x['hydrogen']*P
            P_CO = x['carbon monoxide']*P
            
            k_ads = param_dict['k_ads']
            H_ads = param_dict['H_ads']
            A_HCs_1 = param_dict['A_HCs_1']
            E_HCs_1 = param_dict['E_HCs_1']
            A_HCs_2 = param_dict['A_HCs_2']
            E_HCs_2 = param_dict['E_HCs_2']
    
            F_H2 = x['hydrogen']*F
            F_CO = x['carbon monoxide']*F
                        
            K_ads = k_ads   * e**(-H_ads   / (R * T))
            k1    = A_HCs_1 * e**(-E_HCs_1 / (R * T))
            k2    = A_HCs_2 * e**(-E_HCs_2 / (R * T))
            r     = y_n(n) * upsilon_n(n, T) * (k1 * P_H2 + k2) * P_CO / (1 + K_ads * P_CO)**2
            return r
    
        reaction = Reaction(
                     name = parafin, 
                     stoic = {
                             'carbon monoxide': -n,
                             'hydrogen': -(2*n + 1),
                              parafin: 1,
                             'water': n,
                         },
                     base_component = parafin, 
                     rate_function = rate_n_para,
                     rate_unit = 'mol/s/kg',
        )
        
        model.add_reaction(reaction)
    
    for olefin in n_olefins:
        n = n_olefins[olefin]
        def rate_n_ole(param_dict, T, P, x, F, K, eq_distance, R, data, n = n):
            P_H2 = x['hydrogen']*P
            P_CO = x['carbon monoxide']*P
            
            k_ads = param_dict['k_ads']
            H_ads = param_dict['H_ads']
            A_HCs_1 = param_dict['A_HCs_1']
            E_HCs_1 = param_dict['E_HCs_1']
            A_HCs_2 = param_dict['A_HCs_2']
            E_HCs_2 = param_dict['E_HCs_2']
    
            F_H2 = x['hydrogen']*F
            F_CO = x['carbon monoxide']*F
                        
            K_ads = k_ads   * e**(-H_ads   / (R * T))
            k1    = A_HCs_1 * e**(-E_HCs_1 / (R * T))
            k2    = A_HCs_2 * e**(-E_HCs_2 / (R * T))
            r     = y_n(n) * (1 - upsilon_n(n, T)) * (k1 * P_H2 + k2) * P_CO / (1 + K_ads * P_CO)**2
            return r
    
        reaction = Reaction(
                     name = olefin, 
                     stoic = {
                             'carbon monoxide': -n,
                             'hydrogen': -(2*n),
                              olefin: 1,
                             'water': n,
                         },
                     base_component = olefin, 
                     rate_function = rate_n_ole,
                     rate_unit = 'mol/s/kg',
        )
        
        model.add_reaction(reaction)
    return model