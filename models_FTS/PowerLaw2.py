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

def PowerLaw2():
    model_name = 'PowerLaw2'
    param_dict =  {'k_ads': np.float64(0.39629148891247384),
  'H_ads': np.float64(0.0029094956008316854),
  'A_HCs': np.float64(18.487821317782657),
  'E_HCs': np.float64(25783.654676718477),
  'a': np.float64(-0.6780673232168071),
  'b': np.float64(2.7605728154297893),
  'c': np.float64(1.276309321861667)}

    bnds = {
        'k_ads': (0, None), # 1/Pa
        'H_ads': (0, None), # J/mol
        'A_HCs': (0, None), # mol/s/kg/Pa²
        'E_HCs': (0, None), # J/mol
        'a': (0, None),
        'b': (0, None),
        'c': (0, None),
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
            A_HCs = param_dict['A_HCs']
            E_HCs = param_dict['E_HCs']
            a = param_dict['a']
            b = param_dict['b']
            c = param_dict['c']

            F_H2 = x['hydrogen']*F
            F_CO = x['carbon monoxide']*F

            K_ads = k_ads * e**(-H_ads / (R * T))
            k_HCs = A_HCs * e**(-E_HCs / (R * T))
            r     = y_n(n) * upsilon_n(n, T) * k_HCs * P_H2**a * P_CO**b / (1 + K_ads * P_CO**c)**2
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
            a = param_dict['a']
            b = param_dict['b']
            c = param_dict['c']

            F_H2 = x['hydrogen']*F
            F_CO = x['carbon monoxide']*F

            K_ads = k_ads * e**(-H_ads / (R * T))
            k_HCs = A_HCs * e**(-E_HCs / (R * T))
            r     = y_n(n) * (1 - upsilon_n(n, T)) * k_HCs * P_H2**a * P_CO**b / (1 + K_ads * P_CO**c)**2
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