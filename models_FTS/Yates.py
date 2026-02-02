import sys
from math import e

# setting path
sys.path.append('..')
from KineticModel import KineticModel
from Reaction import Reaction

from models_FTS.components import parafins, olefins, others

def Yates():
    model_name = 'Yates'
    param_dict = {
        'k_ads': 3.6710267535099613e-06,
        'H_ads': 0, # J/mol
        'A_HCs': 2.2727936268821096e-09,
        'E_HCs': 53620.12463581562, # J/mol
    }
    
    bnds = {
        'k_ads': (0, None), # 1/Pa
        'H_ads': (0, None), # J/mol
        'A_HCs': (0, None), # mol/s/kg/Pa²
        'E_HCs': (0, None), # J/mol
    }
    
    beta_0 = 48.45
    beta_1 = -0.09191
    beta_2 = -0.7489
    
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
            A_HCs = param_dict['A_HCs']
            E_HCs = param_dict['E_HCs']
    
            F_H2 = x['hydrogen']*F
            F_CO = x['carbon monoxide']*F
            
            alfa = 1/(1 + e**-(beta_0 + beta_1*T + beta_2*F_H2/F_CO))
            
            upsilon_exp = -3.303 + 0.4139*n + 3.391*(n == 2) + 21.01*(n == 3)
            upsilon_n = 1/(1 + (1 - (n == 1)) * e**-upsilon_exp)
    
            Kc2 = e ** (29.61 - 0.05564 * T) - 1
            Kc3 = e ** (39.37 - 0.07788 * T) - 1
    
            y_n = (1-alfa)*alfa**(n - 1 + Kc2*(n == 2) + Kc3*(n == 3))
            
            K_ads = k_ads * e**(-H_ads / (R * T))
            k_HCs = A_HCs * e**(-E_HCs / (R * T))
            r     = y_n * upsilon_n * k_HCs * P_H2 * P_CO / (1 + K_ads * P_CO)**2
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
            A_HCs = param_dict['A_HCs']
            E_HCs = param_dict['E_HCs']
    
            F_H2 = x['hydrogen']*F
            F_CO = x['carbon monoxide']*F
            
            alfa = 1/(1 + e**-(beta_0 + beta_1*T + beta_2*F_H2/F_CO))
            
            upsilon_exp = -3.303 + 0.4139*n + 3.391*(n == 2) + 21.01*(n == 3)
            upsilon_n = 1/(1 + (1 - (n == 1)) * e**-upsilon_exp)
    
            Kc2 = e ** (29.61 - 0.05564 * T) - 1
            Kc3 = e ** (39.37 - 0.07788 * T) - 1
    
            y_n = (1-alfa)*alfa**(n - 1 + Kc2*(n == 2) + Kc3*(n == 3))
            
            K_ads = k_ads * e**(-H_ads / (R * T))
            k_HCs = A_HCs * e**(-E_HCs / (R * T))
            r     = y_n * (1 - upsilon_n) * k_HCs * P_H2 * P_CO / (1 + K_ads * P_CO)**2
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