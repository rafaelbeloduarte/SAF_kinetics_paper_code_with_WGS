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
from models_FTS.chain_length import n_parafins, n_olefins

def Ojeda():
    model_name = 'Ojeda'
    param_dict = {'k_ads': np.float64(3.730093391449159e-06),
                     'H_ads': np.float64(72.95034563075708),
                     'A_HCs_1': np.float64(4393.1276634506985),
                     'E_HCs_1': np.float64(25724783.134262726),
                     'A_HCs_2': np.float64(0.02526364345428176),
                     'E_HCs_2': np.float64(64040.357844284124),
                     'A_WGS': np.float64(6368129686.779257),
                     'E_WGS': np.float64(105466.71937308327),
                     'a_WGS': np.float64(0.007226556322800434),
                     'b_WGS': np.float64(-0.5384067120372443)}

    bnds = {
        'k_ads': (0, None),
        'H_ads': (0, None),
        'A_HCs_1': (0, None),
        'E_HCs_1': (0, None),
        'A_HCs_2': (0, None),
        'E_HCs_2': (0, None),
        'A_WGS': (0, None),
        'E_WGS': (0, None),
        'a_WGS': (None, None),
        'b_WGS': (None, None),
    }

    model = KineticModel(model_name, params = param_dict, bnds = bnds)

    def rate_WGS(param_dict, T, P, x, F, K, eq_distance, R, data):
        P_CO = x['carbon monoxide']*P
        P_H2O = x['water']*P
        P_CO2 = x['carbon dioxide']*P
        P_H2 = x['hydrogen']*P

        A_WGS = param_dict['A_WGS']
        E_WGS = param_dict['E_WGS']
        a = param_dict['a_WGS']
        b = param_dict['b_WGS']

        K_WGS = 1.45e-2 * np.exp( 4.62e3 / T )
        k_WGS = A_WGS * np.exp ( - E_WGS / ( 8.314 * T ) )

        eq_dist = 1 - P_CO2 * P_H2 / ( K_WGS * max(10, P_CO) * max(10, P_H2O) )
        
        rate_WGS = ( k_WGS * P_CO**a * P_H2O**b ) * eq_dist
        return rate_WGS

    WGS = Reaction(
                     name = 'WGS',
                     stoic = {
                             'carbon monoxide': -1,
                             'water': -1,
                             'carbon dioxide': 1,
                             'hydrogen': 1,
                         },
                     base_component = 'carbon monoxide',
                     rate_function = rate_WGS,
                     rate_unit = 'mol/s/kg',
        )

    model.add_reaction(WGS)

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
            r     = y_n(n, T) * upsilon_n(n, T) * (k1 * P_H2 + k2) * P_CO / (1 + K_ads * P_CO)**2
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
            r     = y_n(n, T) * (1 - upsilon_n(n, T)) * (k1 * P_H2 + k2) * P_CO / (1 + K_ads * P_CO)**2
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