import sys
from math import e

# setting path
sys.path.append('..')
from KineticModel import KineticModel
from Reaction import Reaction

from models_FTS.components import parafins, olefins, others

def WangSimple():
    model_name = 'WangSimple'

    param_dict = {
        'k_ads_CO': 1.64373179e+01,
        'H_ads_CO': 6.72627803e+04, # J/mol
        'k_ads_H2O': 1.72400888e+01,
        'H_ads_H2O': 4.38520022e+04, # J/mol
        'A_HCs': 2.17288962e-10,
        'E_HCs': 4.93659514e+04, # J/mol
        'a'    : 1.18899989e+00,
        'b'    : 1.20289487e+00,
    }
    
    bnds = {
        'k_ads_CO': (0, None), # 1/Pa
        'H_ads_CO': (0, None), # J/mol
        'k_ads_H2O': (0, None), # 1/Pa
        'H_ads_H2O': (0, None), # J/mol
        'A_HCs': (0, None), # mol/s/kg/Pa²
        'E_HCs': (0, None), # J/mol
        'a': (0, None),
        'b': (0, None),
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
            P_H2O = max(1, x['water']*P) # using max to avoid division by zero
            
            k_ads_CO = param_dict['k_ads_CO']
            H_ads_CO = param_dict['H_ads_CO']
            k_ads_H2O = param_dict['k_ads_H2O']
            H_ads_H2O = param_dict['H_ads_H2O']
            A_HCs = param_dict['A_HCs']
            E_HCs = param_dict['E_HCs']
            a = param_dict['a']
            b = param_dict['b']
            c = 1
            d = 1
    
            F_H2 = x['hydrogen']*F
            F_CO = x['carbon monoxide']*F
            
            alfa = 1/(1 + e**-(beta_0 + beta_1*T + beta_2*F_H2/F_CO))
            
            upsilon_exp = -3.303 + 0.4139*n + 3.391*(n == 2) + 21.01*(n == 3)
            upsilon_n = 1/(1 + (1 - (n == 1)) * e**-upsilon_exp)
    
            Kc2 = 204.08 - 0.3960*T
            Kc3 = 88.47  - 0.1737*T
    
            y_n = (1-alfa)*alfa**(n - 1 + Kc2*(n == 2) + Kc3*(n == 3))
            
            K_ads_CO  = k_ads_CO  * e**(-H_ads_CO  / (R * T))
            K_ads_H2O = k_ads_H2O * e**(-H_ads_H2O / (R * T))
            k_HCs     = A_HCs     * e**(-E_HCs     / (R * T))
            r         = y_n * upsilon_n * k_HCs * P_H2**a * P_CO**b / (1 + K_ads_CO * P_CO**c + K_ads_H2O * P_H2O**d)**2
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
            P_H2O = max(1, x['water']*P) # using max to avoid division by zero
            
            k_ads_CO = param_dict['k_ads_CO']
            H_ads_CO = param_dict['H_ads_CO']
            k_ads_H2O = param_dict['k_ads_H2O']
            H_ads_H2O = param_dict['H_ads_H2O']
            A_HCs = param_dict['A_HCs']
            E_HCs = param_dict['E_HCs']
            a = param_dict['a']
            b = param_dict['b']
            c = 1
            d = 1
    
            F_H2 = x['hydrogen']*F
            F_CO = x['carbon monoxide']*F
            
            alfa = 1/(1 + e**-(beta_0 + beta_1*T + beta_2*F_H2/F_CO))
            
            upsilon_exp = -3.303 + 0.4139*n + 3.391*(n == 2) + 21.01*(n == 3)
            upsilon_n = 1/(1 + (1 - (n == 1)) * e**-upsilon_exp)
    
            Kc2 = 204.08 - 0.3960*T
            Kc3 = 88.47  - 0.1737*T
    
            y_n = (1-alfa)*alfa**(n - 1 + Kc2*(n == 2) + Kc3*(n == 3))
            
            K_ads_CO  = k_ads_CO  * e**(-H_ads_CO  / (R * T))
            K_ads_H2O = k_ads_H2O * e**(-H_ads_H2O / (R * T))
            k_HCs     = A_HCs     * e**(-E_HCs     / (R * T))
            r         = y_n * (1 - upsilon_n) * k_HCs * P_H2**a * P_CO**b / (1 + K_ads_CO * P_CO**c + K_ads_H2O * P_H2O**d)**2
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