#!/usr/bin/env python
# coding: utf-8

# In[112]:


import pandas as pd
from math import exp
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
import multiprocessing
from scipy.optimize import curve_fit
from sklearn.cluster import MeanShift
import random
from scipy.stats import sobol_indices, uniform


# In[2]:


from models_FTS.components import parafins, olefins, others


# In[3]:


def calc_alpha(beta, T):
    b0 = beta[0][:, np.newaxis]
    b1 = beta[1][:, np.newaxis]
    alpha = 1 / ( 1 + np.exp( - ( b0 + b1 * T.values ) ) )
    return alpha

def K_C2(beta, T):
    b0 = beta[0][:, np.newaxis]
    b1 = beta[1][:, np.newaxis]
    b2 = beta[2][:, np.newaxis]
    K_C2 = b0/(1 + np.exp( - ( b1 + b2 * T.values ) ) )
    return K_C2

def calc_y_n(beta_alpha, beta_KC, n, T):
    alpha = calc_alpha(beta_alpha, T)
    y_n = ( 1 - alpha ) * alpha ** ( n - 1  + K_C2(beta_KC, T) * ( n == 2 ) )
    return y_n

def y_n_distribution(beta_alpha, beta_KC, T, N):
    y_n = {}

    for n in N:
        y_n[n] = calc_y_n(beta_alpha, beta_KC, n, T)

    # the chategorical variables change the distribution sum to be always less than 1
    # so we will normalize it before returning
    y_n = {key: value/sum(y_n.values()) for key, value in y_n.items()}
    return y_n


# In[4]:


def upsilon_n(beta, n, T):
    exponential = np.exp ( - ( beta[0] + beta[1] * T + beta[2] * n + beta[3] * ( n == 2 ) ) )
    upsilon_n = 1 / ( 1 + ( 1 - ( n == 1 ) - ( n == 3 ) ) * exponential )
    return upsilon_n


# In[5]:


def upsilons(beta, N, T):
    return {n: upsilon_n(beta, n, T) for n in N}


# In[6]:


np.random.seed(84)
random.seed(84)


# In[7]:


# pd.set_option('display.float_format', '{:,.4E}'.format)


# In[8]:


pd.options.display.max_seq_items = 2000


# In[9]:


reconciled_data = pd.read_pickle('reconciled_data.pkl')


# In[10]:


kinetic_data = reconciled_data.loc[reconciled_data.CINÉTICA == True]


# In[11]:


X = kinetic_data[['T_R_C']]
clustering = MeanShift(bandwidth=2).fit(X)
clustering.labels_


# In[12]:


kinetic_data['cluster'] = clustering.labels_


# In[13]:


cluster_set = list(set(kinetic_data.cluster))


# In[14]:


val_clusters = random.sample(cluster_set, 2)
val_clusters


# In[15]:


kinetic_data.groupby(['cluster'])['T_R_C'].mean().sort_values()


# In[16]:


kinetic_data.groupby(['cluster'])['T_R_C'].std()


# In[17]:


kinetic_data['cluster'].isin(val_clusters).sum() / len(kinetic_data)


# In[18]:


Fj_out = kinetic_data[[
    'F_H2_s_mol_s', 'F_CH4_s_mol_s', 'F_CO_s_mol_s',
       'F_CO2_s_mol_s', 'F_C2H4_s_mol_s', 'F_C2H6_s_mol_s', 'F_C3H8_s_mol_s',
       'F_H2O_s_mol_s',
    'F_C9', 'F_C10', 'F_C11',
       'F_C12', 'F_C13', 'F_C14', 'F_C15', 'F_C16',
    'F_C17', 'F_C18', 'F_C19',
       'F_C20', 'F_C21', 'F_C22', 'F_C23', 'F_C24', 'F_C25', 
        'F_C26', 'F_C27',
       'F_C28', 'F_C29', 'F_C30', 'F_C31', 'F_C32', 'F_C33', 'F_C34', 'F_C35',
       'F_C36', 
    'F_Olef_C9', 'F_Olef_C10',
       'F_Olef_C11', 'F_Olef_C12', 'F_Olef_C13', 'F_Olef_C14', 'F_Olef_C15',
       'F_Olef_C16', 'F_Olef_C17', 'F_Olef_C18'
]]


# In[19]:


Fj_out_HCs = Fj_out.drop(columns=['F_H2_s_mol_s', 'F_CO_s_mol_s', 'F_CO2_s_mol_s', 'F_H2O_s_mol_s'])


# In[20]:


Ns_parafins = {
    'F_CH4_s_mol_s': 1, 'F_C2H6_s_mol_s': 2, 'F_C3H8_s_mol_s': 3,
       'F_C9': 9, 'F_C10': 10, 'F_C11': 11, 'F_C12': 12, 'F_C13': 13, 'F_C14': 14, 'F_C15': 15, 'F_C16': 16,
       'F_C17': 17, 'F_C18': 18, 'F_C19': 19, 'F_C20': 20, 'F_C21': 21, 'F_C22': 22, 'F_C23': 23, 'F_C24': 24,
       'F_C25': 25, 
       'F_C26': 26, 'F_C27': 27, 'F_C28': 28, 'F_C29': 29, 'F_C30': 30, 'F_C31': 31, 'F_C32': 32,
       'F_C33': 33, 'F_C34': 34, 'F_C35': 35, 'F_C36': 36,
}
Ns_olefins = {
    'F_C2H4_s_mol_s': 2,
       'F_Olef_C9': 9, 'F_Olef_C10': 10,
       'F_Olef_C11': 11, 'F_Olef_C12': 12, 'F_Olef_C13': 13, 'F_Olef_C14': 14, 'F_Olef_C15': 15,
       'F_Olef_C16': 16, 'F_Olef_C17': 17, 'F_Olef_C18': 18, 
}


# In[21]:


all_Ns = list(set(list(Ns_parafins.values()) + list(Ns_olefins.values())))


# In[22]:


rates_Ns_parafins = {
    'r_CH4_s_mol_s': 1, 'r_C2H6_s_mol_s': 2, 'r_C3H8_s_mol_s': 3,
       'r_C9': 9, 'r_C10': 10, 'r_C11': 11, 'r_C12': 12, 'r_C13': 13, 'r_C14': 14, 'r_C15': 15, 'r_C16': 16,
       'r_C17': 17, 'r_C18': 18, 'r_C19': 19, 'r_C20': 20, 'r_C21': 21, 'r_C22': 22, 'r_C23': 23, 'r_C24': 24,
       'r_C25': 25,
       'r_C26': 26, 'r_C27': 27, 'r_C28': 28, 'r_C29': 29, 'r_C30': 30, 'r_C31': 31, 'r_C32': 32,
       'r_C33': 33, 'r_C34': 34, 'r_C35': 35, 'r_C36': 36,
}
rates_Ns_olefins = {
    'r_C2H4_s_mol_s': 2,
       'r_Olef_C9': 9, 'r_Olef_C10': 10,
       'r_Olef_C11': 11, 'r_Olef_C12': 12, 'r_Olef_C13': 13, 'r_Olef_C14': 14, 'r_Olef_C15': 15,
       'r_Olef_C16': 16, 'r_Olef_C17': 17, 'r_Olef_C18': 18, 
}


# In[23]:


MM_parafins = {key: (n*12 + 2*n + 2) for key, n in Ns_parafins.items()}
MM_olefins = {key: (n*12 + 2*n) for key, n in Ns_olefins.items()}


# In[24]:


Fj_out_HCs_parafins = Fj_out_HCs[Ns_parafins.keys()]
Fj_out_HCs_olefins = Fj_out_HCs[Ns_olefins.keys()]


# In[25]:


m_out_parafins = Fj_out_HCs_parafins.mul(MM_parafins, axis = 'columns')
m_out_olefins = Fj_out_HCs_olefins.mul(MM_olefins, axis = 'columns')


# In[26]:


rates_exp_parafins = Fj_out_HCs_parafins.div(kinetic_data.m_cat_g, axis='rows')
rates_exp_olefins = Fj_out_HCs_olefins.div(kinetic_data.m_cat_g, axis='rows')
rates_exp_parafins.columns = rates_exp_parafins.columns.str.replace('^F_', 'r_', regex=True)
rates_exp_olefins.columns = rates_exp_olefins.columns.str.replace('^F_', 'r_', regex=True)

rates_exp_H2 = (kinetic_data.F_H2_s_mol_s - kinetic_data.F_H2_e_mol_s) / kinetic_data.m_cat_g
rates_exp_CO = (kinetic_data.F_CO_s_mol_s - kinetic_data.F_CO_e_mol_s) / kinetic_data.m_cat_g


# In[27]:


loc_validation = kinetic_data['cluster'].isin(val_clusters)
loc_training = ~loc_validation
kinetic_data['training'] = loc_training


# In[28]:


len(kinetic_data) == len(kinetic_data.loc[loc_validation]) + len(kinetic_data.loc[loc_training])


# In[29]:


F_CO_in = kinetic_data.F_CO_e_mol_s
F_CO_out = kinetic_data.F_CO_s_mol_s

F_H2_in = kinetic_data.F_H2_e_mol_s
F_H2_out = kinetic_data.F_H2_s_mol_s

y_H2_in = F_H2_in / (F_H2_in + F_CO_in)
y_CO_in = F_CO_in / (F_H2_in + F_CO_in)

y_H2_out = Fj_out['F_H2_s_mol_s'] / Fj_out.sum(axis=1)
y_CO_out = Fj_out['F_CO_s_mol_s'] / Fj_out.sum(axis=1)

y_H2_mean = ( y_H2_in + y_H2_out ) / 2
y_CO_mean = ( y_CO_in + y_CO_out ) / 2

kinetic_data['P_H2'] = (kinetic_data.P_abs_Pa * y_H2_out).astype('float')
kinetic_data['P_CO'] = (kinetic_data.P_abs_Pa * y_CO_out).astype('float')


# In[30]:


def yn_upsilon_n(y_n_dist, upsilons, index):
    return pd.DataFrame({n: y_n_dist[n] * upsilons[n] for n in rates_Ns_parafins.values()},
                       index = index)


# In[31]:


def yn_1_minus_upsilon_n(y_n_dist, upsilons, index):
    return pd.DataFrame({n: y_n_dist[n] * (1 - upsilons[n]) for n in rates_Ns_olefins.values()},
                       index = index)


# In[32]:


def model_rates_parafins(data, rate_fcn, params, p_loc):
    beta_alpha = [params[loc] for loc in p_loc['alpha']]
    beta_KC = [params[loc] for loc in p_loc['KC']]
    beta_ups = [params[loc] for loc in p_loc['upsilon']]
    params_rate = [params[loc] for loc in p_loc['rate']]

    Ns = rates_Ns_parafins.values()

    y_n_dist = y_n_distribution(beta_alpha, beta_KC, data.T_R_K, Ns)

    rates = yn_upsilon_n(y_n_dist, 
                         upsilons(beta_ups, Ns, data.T_R_K), 
                         data.index).mul(
                                        rate_fcn(*params_rate, 
                                        data.T_R_K, 
                                        data.P_CO, 
                                        data.P_H2), axis='index')

    rates = rates.rename(columns = {value: key for key, value in rates_Ns_parafins.items()})

    return rates


# In[33]:


def model_rates_olefins(data, rate_fcn, params, p_loc):
    beta_alpha = [params[loc] for loc in p_loc['alpha']]
    beta_KC = [params[loc] for loc in p_loc['KC']]
    beta_ups = [params[loc] for loc in p_loc['upsilon']]
    params_rate = [params[loc] for loc in p_loc['rate']]

    Ns = rates_Ns_olefins.values()

    y_n_dist = y_n_distribution(beta_alpha, beta_KC, data.T_R_K, Ns)

    rates = yn_1_minus_upsilon_n(y_n_dist, 
                                 upsilons(beta_ups, Ns, data.T_R_K), 
                                 data.index).mul(
                                                rate_fcn(*params_rate, 
                                                data.T_R_K,
                                                data.P_CO,
                                                data.P_H2
                                                        ), axis='index')

    rates = rates.rename(columns = {value: key for key, value in rates_Ns_olefins.items()})

    return rates


# In[34]:


def model_Fout(rates, W):
    Fout_model = rates.mul(W, axis = 'index')
    Fout_model.columns = Fout_model.columns.str.replace('^r_', 'F_', regex=True)
    return Fout_model


# In[35]:


def residuals(params, data, rate_fcn, p_loc):
    rates_model_parafins = model_rates_parafins(data, 
                                                rate_fcn, 
                                                params, p_loc)
    rates_model_olefins  = model_rates_olefins( data, 
                                                rate_fcn, 
                                                params, p_loc)

    rates_model_CO  = rates_model_parafins.mul(rates_Ns_parafins, axis = 'columns').sum(axis = 'columns')
    rates_model_CO += rates_model_olefins.mul(rates_Ns_olefins, axis = 'columns').sum(axis = 'columns')
    rates_model_CO = - rates_model_CO

    stoic_H2_parafins = {key: (value * 2 + 1) for key, value in rates_Ns_parafins.items()}
    stoic_H2_olefins = {key: (value * 2) for key, value in rates_Ns_olefins.items()}

    rates_model_H2  = rates_model_parafins.mul(stoic_H2_parafins, axis = 'columns').sum(axis = 'columns')
    rates_model_H2 += rates_model_olefins.mul(stoic_H2_olefins, axis = 'columns').sum(axis = 'columns')
    rates_model_H2 = - rates_model_H2

    # Fout_model_parafins = model_Fout(rates_model_parafins, data.m_cat_g)
    # Fout_model_olefins = model_Fout(rates_model_olefins, data.m_cat_g)

    # m_out_parafins_model = Fout_model_parafins.mul(MM_parafins, axis = 'columns')
    # m_out_olefins_model = Fout_model_olefins.mul(MM_olefins, axis = 'columns')

    res_parafins = (rates_exp_parafins - rates_model_parafins)
    res_olefins  = (rates_exp_olefins - rates_model_olefins)

    res_CO = (rates_exp_CO - rates_model_CO)
    res_H2 = (rates_exp_H2 - rates_model_H2)

    # res_m_parafins = m_out_parafins - m_out_parafins_model
    # res_m_olefins = m_out_olefins - m_out_olefins_model

    res_parafins = res_parafins.dropna()
    res_olefins = res_olefins.dropna()
    res_CO = res_CO.dropna()
    res_H2 = res_H2.dropna()
    # res_m_parafins = res_m_parafins.dropna()
    # res_m_olefins = res_m_olefins.dropna()

    return res_parafins, res_olefins, res_CO, res_H2


# In[36]:


def objective_fcn(params, data, rate_fcn, p_loc):
    (res_parafins, res_olefins,
     res_CO, res_H2) = residuals(params, data, rate_fcn, p_loc)

    res_parafins = res_parafins.mul(MM_parafins, axis = 'columns').sum(axis = 'columns')
    res_olefins = res_olefins.mul(MM_olefins, axis = 'columns').sum(axis = 'columns')
    res_CO = 28 * res_CO
    res_H2 = 2 * res_H2

    sum_diff_sqr = (
        ( res_parafins ** 2 ).sum().sum() 
        + ( res_olefins ** 2 ).sum().sum() 
        + (res_CO**2).sum() + (res_H2**2).sum()
    )
    # print(params)
    # print(sum_diff_sqr)
    # clear_output(wait = True)
    return sum_diff_sqr


# In[37]:


def fit_function(train_data, rate_fcn, p0, bnds, model_name, max_fev, p_loc, method):
    sol = minimize(objective_fcn, p0,
             method=method,
             bounds=bnds, 
             args = (train_data, rate_fcn, p_loc),
             options = {'maxfev': max_fev*len(p0)},
             )
    return sol


# In[38]:


max_fev = 20


# In[39]:


def rate_HCs_Yates(k_ads, H_ads, A_HCs, E_HCs, T, P_CO, P_H2):
    k_HCs = A_HCs * np.exp ( - E_HCs / ( 8.314 * T ) )
    K_CO =  k_ads * np.exp ( - H_ads / ( 8.314 * T ) )
    rate = k_HCs * P_CO * P_H2 / ( ( 1 + K_CO * P_CO ) ** 2 )
    return rate

beta_alpha = [19.46471056, -0.03635193]
beta_KC = [58.02629507, 36.02113642, -0.07562323]
beta_upsilon = [-12.37799969,   0.02011026,   0.29720755,   2.58255525]

p0_Yates = [1, 5e4, 1, 1e5]

p0_Yates = p0_Yates + beta_alpha + beta_KC + beta_upsilon

Yates_p_loc = {
    'rate': [0,1,2,3],
    'alpha': [4,5],
    'KC': [6,7,8],
    'upsilon': [9,10,11,12],
}

Yates_bnds = ( (0, None), (0, None), (0, None), (0, None) ,
               (None, None), (None, None),
               (None, None), (None, None), (None, None),
               (None, None), (None, None), (None, None), (None, None),
             )

sol_Yates = fit_function(kinetic_data.loc[loc_training], rate_HCs_Yates, p0_Yates, 
                         Yates_bnds, 'Yates', max_fev, Yates_p_loc,
                         'Nelder-Mead')
Yates_param_dict =    {'k_ads': sol_Yates.x[0],
                       'H_ads': sol_Yates.x[1],
                       'A_HCs': sol_Yates.x[2]*1000,
                       'E_HCs': sol_Yates.x[3],
                       }


# In[116]:


Yates_bnds_sobol = []
for p in sol_Yates.x:
    if p < 0:
        Yates_bnds_sobol.append(uniform(loc=p*1.1, scale=p*0.9))
    else:
        Yates_bnds_sobol.append(uniform(loc=p*0.9, scale=p*1.1))


# In[118]:


from tqdm import tqdm


# In[121]:


def obj_Yates(p):
    return objective_fcn(p, kinetic_data.loc[loc_training], rate_HCs_Yates, Yates_p_loc)

indices = sobol_indices(
    func=obj_Yates, n=128,
    dists=Yates_bnds_sobol,
)
indices.first_order
indices.total_order


# In[45]:


sol_Yates.x


# In[46]:


Yates_param_dict


# In[47]:


params_dict = {
    # 'PowerLaw'  : PowerLaw_param_dict,
    'Yates'     : Yates_param_dict,
    # 'Botes'     : Botes_param_dict,
    # 'Ojeda'     : Ojeda_param_dict,
    # 'Mousavi'   : Mousavi_param_dict,
    # 'PowerLaw2' : PowerLaw2_param_dict,
              }


# In[48]:


rates_Yates_parafins = model_rates_parafins(kinetic_data, rate_HCs_Yates, sol_Yates.x, Yates_p_loc)
rates_Yates_olefins = model_rates_olefins(kinetic_data, rate_HCs_Yates, sol_Yates.x, Yates_p_loc)


# In[49]:


rates_Yates_CO = rates_Yates_parafins.mul(rates_Ns_parafins, axis = 'columns').sum(axis = 'columns')
rates_Yates_CO += rates_Yates_olefins.mul(rates_Ns_olefins, axis = 'columns').sum(axis = 'columns')
rates_Yates_CO = - rates_Yates_CO


# In[50]:


stoic_H2_parafins = {key: (value * 2 + 1) for key, value in rates_Ns_parafins.items()}
stoic_H2_olefins = {key: (value * 2) for key, value in rates_Ns_olefins.items()}


# In[51]:


rates_Yates_H2 = rates_Yates_parafins.mul(stoic_H2_parafins, axis = 'columns').sum(axis = 'columns')
rates_Yates_H2 += rates_Yates_olefins.mul(stoic_H2_olefins, axis = 'columns').sum(axis = 'columns')
rates_Yates_H2 = - rates_Yates_H2


# In[52]:


rates_plot_olefins = pd.concat((rates_Yates_olefins.melt(value_name = 'rates_olefins_Yates'), 
           rates_exp_olefins.melt(value_name = 'rates_exp_olefins').drop(columns='variable')), axis = 'columns')


# In[53]:


rates_plot_parafins = pd.concat((rates_Yates_parafins.melt(value_name = 'rates_parafins_Yates'), 
           rates_exp_parafins.melt(value_name = 'rates_exp_parafins').drop(columns='variable')), axis = 'columns')


# In[54]:


rates_H2_CO = pd.DataFrame()
rates_H2_CO['rates_Yates_H2'] = rates_Yates_H2
rates_H2_CO['rates_Yates_CO'] = rates_Yates_CO
rates_H2_CO['rates_exp_H2'] = rates_exp_H2
rates_H2_CO['rates_exp_CO'] = rates_exp_CO


# In[55]:


rates_H2_CO.columns


# In[56]:


rates_plot_parafins.rates_exp_parafins.max()


# In[57]:


rates_plot_parafins['n'] = rates_plot_parafins['variable'].map(rates_Ns_parafins)


# In[58]:


fig, axs = plt.subplots(figsize = (3, 2.5))

max_xy = -rates_H2_CO.min().min()
sns.scatterplot(data = rates_plot_parafins, x = 'rates_exp_parafins', y = 'rates_parafins_Yates',
               ax = axs, 
                # hue = 'n',
               label = 'parafins',
               )
sns.scatterplot(data = rates_plot_olefins, x = 'rates_exp_olefins', y = 'rates_olefins_Yates',
               ax = axs, label = 'olefins')
sns.scatterplot(data = -rates_H2_CO, x = 'rates_exp_H2', y = 'rates_Yates_H2',
               ax = axs, label = 'H2')
sns.scatterplot(data = -rates_H2_CO, x = 'rates_exp_CO', y = 'rates_Yates_CO',
               ax = axs, label = 'CO')
plt.yscale('log')
plt.xscale('log')
plt.plot((0, max_xy), (0, max_xy))
plt.grid('both')
plt.legend(bbox_to_anchor=(1.02, 1))


# from solve_ODEs import solve_ODEs
# 
# model_names = [
#     # 'PowerLaw', 
#     'Yates', 
#     # 'Botes', 'Ojeda', 'Mousavi', 'PowerLaw2'
# ]
# 
# for model in model_names:
#     kinetic_data = solve_ODEs(kinetic_data, model, params_dict[model])
# 
# kinetic_data.to_pickle('kinetic_data.pkl')

# kinetic_data = pd.read_pickle('kinetic_data.pkl')

# F_out_Yates_parafins = kinetic_data[[f'F_{component}_out_model_Yates' for component in parafins]]
# F_out_Yates_olefins = kinetic_data[[f'F_{component}_out_model_Yates' for component in olefins]]

# F_out_Yates_parafins.columns = Fj_out_HCs_parafins.columns
# F_out_Yates_olefins.columns = Fj_out_HCs_olefins.columns

# kinetic_data.columns

# fig, ax = plt.subplots(figsize = (3,2.5))
# max_xy = kinetic_data['F_H2_s_mol_s'].max()
# sns.scatterplot(data = kinetic_data, x = 'F_H2_s_mol_s', y = 'F_H2_out_model_Yates', ax = ax, label = r'$\mathrm{H_2}$')
# sns.scatterplot(data = kinetic_data, x = 'F_CO_s_mol_s', y = 'F_CO_out_model_Yates', ax = ax, label = 'CO')
# plt.plot((0, max_xy), (0, max_xy))
# plt.grid('both')

# fig, ax = plt.subplots(figsize = (3,2.5))
# max_xy = kinetic_data['$m_{liq,out}$ (g/s)'].max()
# sns.scatterplot(data = kinetic_data, x = '$m_{liq,out}$ (g/s)', y = 'm_liq_out_model_Yates')
# plt.plot((0, max_xy), (0, max_xy))
# plt.grid('both')

# F_out_Yates_parafins.melt()

# F_plot_parafins = pd.concat((F_out_Yates_parafins.melt(value_name = 'F_out_Yates_parafins'), 
#            Fj_out_HCs_parafins.melt(value_name = 'Fj_out_HCs_parafins').drop(columns='variable')), axis = 'columns')

# F_plot_olefins = pd.concat((F_out_Yates_olefins.melt(value_name = 'F_out_Yates_olefins'), 
#            Fj_out_HCs_olefins.melt(value_name = 'Fj_out_HCs_olefins').drop(columns='variable')), axis = 'columns')

# fig, axs = plt.subplots(figsize = (3, 2.5))
# max_xy = F_plot_parafins.Fj_out_HCs_parafins.max()
# sns.scatterplot(data = F_plot_parafins, x = 'Fj_out_HCs_parafins', y = 'F_out_Yates_parafins',
#                ax = axs, label = 'parafins')
# sns.scatterplot(data = F_plot_olefins, x = 'F_out_Yates_olefins', y = 'Fj_out_HCs_olefins',
#                ax = axs, label = 'olefins')
# # plt.xscale('log')
# # plt.yscale('log')
# plt.plot((0, max_xy), (0, max_xy))
# plt.grid('both')
