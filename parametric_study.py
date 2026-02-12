# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: env0
#     language: python
#     name: env0
# ---

# %% editable=true slideshow={"slide_type": ""}
from math import e
from PBR import PBR
from Reaction import Reaction
from KineticModel import KineticModel
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# %%
pd.options.display.max_seq_items = 2000

# %%
# loading data for plotting
# Remenber: the fit was also performed on reconciled data
with open('reconciled_data_validation.pkl', 'rb') as f:
    data = pickle.load(f)
kinetic_data = data.loc[(data['CINÉTICA'] == 1.0) | (data['VALIDATION'] == 1.0)].reset_index()
kinetic_data['m_cat_kg'] = kinetic_data['m_cat_g']/1000

# %%
kinetic_data.head()

# %%
F0 = kinetic_data['F_CO_e_mol_s'].mean() + kinetic_data['F_H2_e_mol_s'].mean()
F0

# %%
H2_CO = kinetic_data['$H_2/CO$'].mean()
x_CO = 1 / ( 1 + H2_CO )
x_H2 = 1 - x_CO

# %%
# create reactor object
T0 = (220, 'C')
x0 = {
                  'carbon monoxide': x_CO,
                  'hydrogen' : x_H2,
              }
reactor = PBR(T0 = T0, 
              P0 = (20, 'bar'), 
              x0 = x0, 
              F0 = (F0, 'mol/s'),
              isothermal = True, 
              ergun = True,
              WT = (5.12, 'g'), 
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

# %%
from models_FTS.Yates            import Yates
from models_FTS.PowerLaw         import PowerLaw
from models_FTS.Botes            import Botes
from models_FTS.vanSteen         import vanSteen
from models_FTS.Ojeda            import Ojeda
from models_FTS.Mousavi          import Mousavi
from models_FTS.MousaviPLaw      import MousaviPLaw
from models_FTS.Wang             import Wang
from models_FTS.WangSimple       import WangSimple
from models_FTS.Elementary       import Elementary
from models_FTS.PowerLaw2        import PowerLaw2

# %%
Yates       =      Yates()
PowerLaw    =      PowerLaw()
Botes       =      Botes()
vanSteen    =      vanSteen()
Ojeda       =      Ojeda()
Mousavi     =      Mousavi()
MousaviPLaw =      MousaviPLaw()
Wang        =      Wang()
WangSimple  =      WangSimple()
Elementary  =      Elementary()
PowerLaw2 = PowerLaw2()

# %%
reactor.add_model(Yates)

# %%
model = 'Yates'

# %%
param_dict_T0 = {
              'T0'             : (np.linspace(180, 260, num = 20), 'C'),
              # 'P0'             : ([1e5], 'Pa'),
              # 'F0'             : ([0.0003286777777777], 'mol/s'),
              # 'x0'             : [{'methane': 1 - x_H2O - ( 1 - x_H2O ) / ( 1 + CH4_CO2 ), 
              #                      'water': x_H2O,
              #                      'carbon dioxide': ( 1 - x_H2O ) / ( 1 + CH4_CO2 ),
              #                      'carbon monoxide': 0.0,
              #                      'hydrogen': 0.0} for x_H2O in np.linspace(0.1, 0.9, num = 25, endpoint = True)],
              # 'isothermal'     : [False],
              # 'ergun'          : [False],
              # 'U'              : ([121], 'W/m2/K'),
              # 'D'              : ([1], 'in'),
              # 'Dp'             : ([4], 'mm'),
              # 'Ta'             : ([800], 'C'),
            }

# %%
reactor.W_steps = 2
dfs = reactor.parametric_study(param_dict_T0,
                               'parametric_results_FTS/parametric_study',
                               conversions = ['carbon monoxide', 'hydrogen'],
                              )
parametric_T0 = dfs[model]

# %%
plots_leg = {
             'carbon monoxide'      : 'CO',
             'X_CO'                 : 'CO',
             'F_CO_s_mol_s'         : 'CO',
             'hydrogen'             : r'$\mathrm{H_2}$',
             'X_H2'                 : r'$\mathrm{H_2}$',
             'F_H2_s_mol_s'         : r'$\mathrm{H_2}$',
            }


# %% editable=true slideshow={"slide_type": ""}
kinetic_data_filtered = kinetic_data[['T_R_C', 'X_CO', 'X_H2']]
kinetic_data_filtered = kinetic_data_filtered.melt(id_vars = 'T_R_C')

fig, ax = plt.subplots(figsize=(3,2.5))
sns.lineplot(x = 'T0 (C)', y = 'X',
             data = parametric_T0.loc[(parametric_T0['X'].notna()) & (parametric_T0['W'] == parametric_T0['W'].max())],
             hue = 'variable',
             style = 'variable',
             palette = 'bright'
            )
sns.scatterplot(x = 'T_R_C', y = 'value',
             data = kinetic_data_filtered,
             hue = 'variable',
             style = 'variable',
             palette = 'bright'
            )
handles, labels = ax.get_legend_handles_labels()
labels = [plots_leg[label] for label in labels if label in plots_leg]
plt.legend(handles, labels)
plt.ylim(-0.05, 1.05)
plt.xlabel(r'$T$ (ºC)')
plt.ylabel(r'$X$')
plt.grid('both')

# %%
kinetic_data_filtered

# %% editable=true slideshow={"slide_type": ""}
kinetic_data_filtered = kinetic_data[['T_R_C', 'F_CO_s_mol_s', 'F_H2_s_mol_s']]
kinetic_data_filtered = kinetic_data_filtered.melt(id_vars = 'T_R_C')

fig, ax = plt.subplots(figsize=(3,2.5))
sns.lineplot(x = 'T0 (C)', y = 'value',
             data = parametric_T0.loc[(parametric_T0['X'].notna()) & (parametric_T0['W'] == parametric_T0['W'].max())],
             hue = 'variable',
             style = 'variable',
             palette = 'bright'
            )
sns.scatterplot(x = 'T_R_C', y = 'value',
             data = kinetic_data_filtered,
             hue = 'variable',
             style = 'variable',
             palette = 'bright'
            )
ax.ticklabel_format(style='sci', scilimits=(0,0), axis = 'y')
handles, labels = ax.get_legend_handles_labels()
labels = [plots_leg[label] for label in labels if label in plots_leg]
plt.legend(handles, labels, bbox_to_anchor=(1.05, 1.05))
plt.xlabel(r'$T$ (ºC)')
plt.ylabel(r'$F$ (mol/s)')
plt.grid('both')

# %%
