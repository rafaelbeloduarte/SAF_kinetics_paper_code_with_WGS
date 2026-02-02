from math import e as exp
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from IPython.display import clear_output
from DVar import DVar
from Unit import Unit
from TEMPLATE_DATA import TEMPLATE_DATA
from thermo import ChemicalConstantsPackage, HeatCapacityGas, HeatCapacityLiquid
from thermo import ViscosityGas, ViscosityGasMixture
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
import logging
import warnings
import os
logger = logging.getLogger(__name__)

def solve(self, model_name = ""):
    # set the model to solve, if model_name not provided select the last model
    if model_name != "":
        self.model = self.models[model_name]
    else:
        self.model = list(self.models.values())[-1]

    if self.liquid_phase:
        self.solve_liq()
    else:
        self.solve_gas()

def solve_liq(self):
    raise ValueError("Liquid phase calculation not yet implemented.")

def solve_gas(self):
    # before doing anything populate the thermodynamic constants and objects
    self.get_thermo_constants()

    def eq_balances(W, Yi):
        # defining the variables from the overall vector
        Q = Yi[0]
        # do not remove the DVarType call in the line below
        # functions expect DVar type temperature
        P = DVarType(max(1e-3, Yi[1]), 'P', self.units)
        T = DVarType(max(1e-3, Yi[2]), 'T', self.units) # T must be >= 0
        if P.val < 1:
            raise ValueError(f"Pressure of {P} is too low. Check your simulation.")
        if T.val < 1:
            raise ValueError(f"Temperature of {T} is too low. Check your simulation.")
        Fi = np.array(Yi[3:len(Yi)]).clip(min = self.minimal_F)
        F = sum(Fi)
        x = {component: Fi[i]/F for i, component in enumerate(self.component_list)}
        Fi = {component: Fi[i] for i, component in enumerate(self.component_list)}

        # energy balance and equilibrium calculation
        # must remember to update the Cps and reaction delta H
        # updata_reactions_HSG must be outside the ifs,
        # otherwise K_eq is not updated when performing equilibrium calculation
        self.update_reactions_HSG(T)
        # calculate reaction rates
        self.calc_rx_rates(T.val, P.val, x, F)
        dT_dW = 0
        dQ_dW = 0
        dP_dW = 0

        # material balances
        dF_dW_i = {}
        for component in self.component_list:
            # initialize the dF_dW_i dict
            dF_dW_i[component] = 0
            for reaction in self.model.reactions:
                # only calculate dF_dW if the component exists in the reaction
                if component in self.model.reactions[reaction].component_names:
                    # when any Fi in the reaction goes below zero, zero the reaction rate
                    if Fi[component] < 0:
                        self.model.reactions[reaction].reaction_rate = DVarType(0, 'r', self.units)
                    nu_ij = self.model.reactions[reaction].stoic[component]
                    nu_j_base = self.model.reactions[reaction].stoic[self.model.reactions[reaction].base_component]
                    eq_dist = self.model.reactions[reaction].eq_distance
                    rate = self.model.reactions[reaction].reaction_rate.val
                    dF_dW_i[component] += eq_dist * rate * nu_ij / abs(nu_j_base)

        dF_dW = [dF_dW_i[component] for component in dF_dW_i]

        # to add more items do: dY_dW = dY_dW + new_list
        dY_dW = [dQ_dW] + [dP_dW] + [dT_dW] + dF_dW
        return dY_dW

    def balances(W, Yi):
        # defining the variables from the overall vector
        Q = Yi[0]
        # do not remove the DVarType call in the line below
        # functions expect DVar type temperature
        P = DVarType(max(1e-3, Yi[1]), 'P', self.units)
        T = DVarType(max(1e-3, Yi[2]), 'T', self.units) # T must be >= 0
        if P.val < 1:
            raise ValueError(f"Pressure of {P} is too low. Check your simulation.")
        if T.val < 1:
            raise ValueError(f"Temperature of {T} is too low. Check your simulation.")
        Fi = np.array(Yi[3:len(Yi)]).clip(min = self.minimal_F)
        F = sum(Fi)
        x = {component: Fi[i]/F for i, component in enumerate(self.component_list)}
        Fi = {component: Fi[i] for i, component in enumerate(self.component_list)}
        # need to calculate mass fraction to pass it to the viscosity calculation
        mass_frac = {component: x[component]*self.thermodynamic_constants[component].MWs[0]
                         for component in self.component_list
                         }
        mass_frac = {component: value/sum(mass_frac.values())
                          for component, value in mass_frac.items()}

        # energy balance and equilibrium calculation
        # must remember to update the Cps and reaction delta H
        # updata_reactions_HSG must be outside the ifs,
        # otherwise K_eq is not updated when performing equilibrium calculation
        self.update_reactions_HSG(T)
        # calculate reaction rates
        self.calc_rx_rates(T.val, P.val, x, F)
        if self.energy_balance:
            self.calc_Cps_gases_at_T(T.val)

            sum_rj_deltaH = 0
            for reaction in self.model.reactions:
                rj = self.model.reactions[reaction].reaction_rate.val
                delta_H = self.model.reactions[reaction].delta_H.val
                sum_rj_deltaH += rj*delta_H

            sum_Fi_Cpi = 0
            for component in self.component_list:
                sum_Fi_Cpi += Fi[component]*self.Cp_gases_at_T[component].val

            if self.isothermal:
                dT_dW = 0
            else:
                dT_dW = ((4*self.U.val/(self.bed_packing.val*(self.D.val)))*(self.Ta.val - T.val)
                        -sum_rj_deltaH
                        )/sum_Fi_Cpi

            dQ_dW = dT_dW*sum_Fi_Cpi + sum_rj_deltaH
        else:
            dT_dW = 0
            dQ_dW = 0

        # material balances
        dF_dW_i = {}
        for component in self.component_list:
            # initialize the dF_dW_i dict
            dF_dW_i[component] = 0
            for reaction in self.model.reactions:
                # only calculate dF_dW if the component exists in the reaction
                if component in self.model.reactions[reaction].component_names:
                    # when any Fi in the reaction goes below zero, zero the reaction rate
                    if Fi[component] < 0:
                        self.model.reactions[reaction].reaction_rate = DVarType(0, 'r', self.units)
                    nu_ij = self.model.reactions[reaction].stoic[component]
                    nu_j_base = self.model.reactions[reaction].stoic[self.model.reactions[reaction].base_component]
                    dF_dW_i[component] += (nu_ij/abs(nu_j_base)
                                               )*self.model.reactions[reaction].reaction_rate.val

        dF_dW = [dF_dW_i[component] for component in dF_dW_i]

        if self.ergun:
            self.viscosity = self.viscosity_gas_mix.calculate(T = T.val,
                                                 P = P.val,
                                                 zs = list(x.values()),
                                                 ws = list(mass_frac.values()),
                                                 method = self.method_gas_viscosity_mix)

            vT = DVarType(F*self.R.val*T.val/P.val, 'v', self.units)
            # print(vT)
            Fi_MMi = [Fi[component]*self.thermodynamic_constants[component].MWs[0]
                         for component in self.component_list] # F is always mol/s so Fi_MMi is always g/s
            mass_flow = DVar(sum(Fi_MMi), 'g/s', self.units) # remember: DVar converts to the current self.units
            gas_density = DVarType(mass_flow.val/vT.val, 'd', self.units)
            gas_velocity = DVarType(vT.val/(np.pi*(self.D.val**2)/4), 'u', self.units)
            superficial_mass_velocity = gas_density.val*gas_velocity.val #kg/m²/s or g/cm²/s

            ergun_1 = -4 / (np.pi * (self.D.val**2) * self.bed_packing.val)
            ergun_2 = superficial_mass_velocity / (gas_density.val*self.Dp.val)
            ergun_3 = (1 - self.bed_porosity)/(self.bed_porosity**3)
            ergun_4 = (150*(1 - self.bed_porosity)*self.viscosity)/self.Dp.val + 1.75*superficial_mass_velocity

            dP_dW = ergun_1 * ergun_2 * ergun_3 * ergun_4
        else:
            dP_dW = 0

        # to add more items do: dY_dW = dY_dW + new_list
        dY_dW = [dQ_dW] + [dP_dW] + [dT_dW] + dF_dW
        return dY_dW

    # during parametric study F0 and x0 can be updated
    # so must recalculate F0 before solving
    self.Fi0 = {}
    for component in self.component_list:
        if component not in self.x0.keys():
            self.Fi0[component] = self.minimal_F
        else:
            self.Fi0[component] = self.F0.val*self.x0[component]

    W_eval = np.linspace(0, self.WT.val, num=self.W_steps, endpoint=True)
    W_span = [0, self.WT.val]
    Y0 = [self.Q0.val, self.P0.val, self.T0.val]
    Y0 = Y0 + list(self.Fi0.values())

    # other variables that must be updated
    self.update_bed_porosity()
    self.update_bed_packing()

    # print(W_eval, W_span, self.Fi[0])
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    # If components of y have different scales, it might be beneficial
    # to set different atol values for different components by
    # passing array_like with shape (n,) for atol.
    solution = solve_ivp(balances, W_span, Y0, method=self.solver_method,
                              t_eval=W_eval, rtol = self.solver_rtol, atol = self.solver_atol)

    solution_W = {'W': solution.t}
    solution_Q = {'Q': solution.y[0]}
    solution_P = {'P': solution.y[1]}
    solution_T = {'T': solution.y[2]}
    # clear Fi to return the full vectors
    solution_Fi = {}
    i = 3
    for component in self.component_list:
        solution_Fi[component] = solution.y[i]
        i += 1

    self.Fi_out_models[self.model.name] = {component: array[-1] for component, array in solution_Fi.items()}

    sol_df = pd.DataFrame({**solution_W, **solution_Fi, **solution_T, **solution_Q, **solution_P})

    # # --- Memory Optimization Addition ---
    # # Downcast floats to 32-bit (or even 16-bit if precision allows)
    # float_cols = sol_df.select_dtypes(include=['float64']).columns
    # for col in float_cols:
    #     # Check if a smaller float type can be used
    #     # if sol_df[col].max() < np.finfo('float32').max and sol_df[col].min() > np.finfo('float32').min:
    #     #     sol_df[col] = sol_df[col].astype('float32')
    #     # If using Python 3.9+ and NumPy 1.23+, you could also try 'float16'
    #     sol_df[col] = pd.to_numeric(sol_df[col], downcast='float') 
    # # --- End Optimization Addition ---

    sol_df = sol_df.melt(id_vars = 'W')
    sol_df['W unit'] = Unit('m', self.units)
    sol_df['value units'] = Unit('F', self.units)
    sol_df.loc[sol_df['variable'] == 'Q', 'value units'] = Unit('Power', self.units)
    sol_df.loc[sol_df['variable'] == 'P', 'value units'] = Unit('P', self.units)
    sol_df.loc[sol_df['variable'] == 'T', 'value units'] = Unit('T', self.units)

    self.sol_dfs[self.model.name] = sol_df # sol_dfs is a dict of DataFrames, one for each model

    if self.calculate_equilibrium:
        sol_df_eq = {'W': [], 'Q': [], 'P': [], 'T': []} | {component: [] for component in self.component_list}
        isothermal = self.isothermal
        ergun = self.ergun
        energy_balance = self.energy_balance
        self.isothermal = True
        self.ergun = False
        self.energy_balance = False
        # integrate until no more change for every point in the solution above
        for i in range(len(solution.t)):
            Y0_eq = solution.y[:,i]
            y = solution.y[:,-1]
            objective_function = 2*self.equilibrium_obj_fun_tolerance
            W0 = 0
            Wf = W0 + self.WT.val
            W_span = [W0, Wf]
            W_eval = np.linspace(W0, Wf, num=self.W_steps, endpoint=True)
            # continue integration until no more change in y
            while objective_function > self.equilibrium_obj_fun_tolerance:
                solution_eq = solve_ivp(eq_balances, W_span, Y0_eq, method=self.solver_method,
                                  t_eval=W_eval, rtol = self.solver_rtol, atol = self.solver_atol)
                W0 = solution_eq.t[-1]
                Wf = W0 + self.WT.val
                W_span = [W0, Wf]
                W_eval = np.linspace(W0, Wf, num=self.W_steps, endpoint=True)
                Y0_eq = solution_eq.y[:,-1] # set next init as current output
                objective_function = sum((y - solution_eq.y[:,-1])**2)
                y = solution_eq.y[:,-1] # store current y out to calculate next diff²

                print(f'Equilibrium calculation for step {i} of {self.W_steps}')
                print(f'objective function: {objective_function}, tolerance: {self.equilibrium_obj_fun_tolerance}')
                # print(f'Wf: {Wf}')
                clear_output(wait=True)


            sol_df_eq['W'].append(solution.t[i])
            sol_df_eq['Q'].append(solution_eq.y[0,-1])
            sol_df_eq['P'].append(solution_eq.y[1,-1])
            sol_df_eq['T'].append(solution_eq.y[2,-1])
            Fi_eq = solution_eq.y[3:, -1]
            for i, component in enumerate(self.component_list):
                sol_df_eq[component].append(Fi_eq[i])

        sol_df_eq = pd.DataFrame(sol_df_eq)
        sol_df_eq = sol_df_eq.melt(id_vars = 'W')
        self.sol_dfs_eq[self.model.name] = sol_df_eq

        self.isothermal = isothermal
        self.ergun = ergun
        self.energy_balance = energy_balance

def calc_residuals(self):
    # create residuals dict
    self.residuals = {key: [] for key in ['Fexp', 'Fmodel', 'residuals', 'component', 'model']}
    # go through the kinetic data and calculate model predictions for each initial condition
    for i in range(len(self.tmp_data)): # first two lines of kinetic data store components and units
        # set initial conditions in the PBR object using data from the file provided by the user
        # lines 0 and 1 must contain the 'component_names' and 'unit'
        # that is why we use i+2 in the code below
        self.WT = DVar(self.tmp_data.iloc[i]['W'], self.kinetic_data.iloc[1]['W'], self.units)
        self.T0 = DVar(self.tmp_data.iloc[i]['Tin'], self.kinetic_data.iloc[1]['Tin'], self.units)
        self.P0 = DVar(self.tmp_data.iloc[i]['Pin'], self.kinetic_data.iloc[1]['Pin'], self.units)
        self.Ta = DVar(self.tmp_data.iloc[i]['Ta'], self.kinetic_data.iloc[1]['Ta'], self.units)

        self.Fi0 = {}
        self.Fi_out_exp = {}
        for component in self.component_list:
            # if component not in data, set it to 0
            if f'Fin_{component}' in self.tmp_data.columns:
                self.Fi0[component] = self.tmp_data.loc[i, f'Fin_{component}']
            else:
                self.Fi0[component] = 0
            if f'Fout_{component}' in self.tmp_data.columns:
                self.Fi_out_exp[component] = self.tmp_data.loc[i, f'Fout_{component}']
            else:
                self.Fi_out_exp[component] = 0

        # F0 is mol/s both in SI and cgs
        self.F0 = DVar(sum(self.Fi0.values()), 'mol/s', self.units)
        self.x0 = {component: self.Fi0[component]/self.F0.val for component in self.Fi0}

        # store the current data line in a tmp variable to provide it to the user's rate function
        self.tmp_data_to_rate_fcn = self.tmp_data.iloc[i]

        self.solve(self.model.name)

        for component in self.Fi_out_exp:
            Fi_exp = self.Fi_out_exp[component]
            Fi_model = self.Fi_out_models[self.model.name][component]
            residual = Fi_exp - Fi_model
            self.residuals['Fexp'].append(Fi_exp)
            self.residuals['Fmodel'].append(Fi_model)
            self.residuals['residuals'].append(residual)
            self.residuals['component'].append(component)
            self.residuals['model'].append(self.model.name)

def calc_neg_logLik(self):
    # the residuals are split into train and test
    # the objective function will only see the train residuals
    residuals = self.residuals['residuals']

    self.model.neg_logLik = -norm.logpdf(residuals,
                                        np.mean(residuals),
                                        np.std(residuals)).sum()

    # calculate AIC (AIC_c = AIC for small sample size)
    kin_data = self.tmp_data
    K = len(self.model.kin_params_list) + 1
    N = len(kin_data)
    # N_test = len(kin_data.loc[kin_data['TRAIN'] == False])
    self.model.aic   = 2*K + 2*self.model.neg_logLik
    self.model.aic_c = self.model.aic + (2*K**2 + 2*K)/(N - K - 1)

def objective_f(self, params, bootstrap = False, i = None):
    # must send the params estimation to the reactions dict
    # the solver takes the params from this dict
    # if this is not done, solver will always use the initial
    # guesses provided when adding the reactions
    self.model.store_kin_params(params)
    self.calc_residuals()
    self.calc_neg_logLik()
    if bootstrap:
        print(f'Bootstrap iteration: {i}')
    print(f"Model: {self.model.name}")
    print("Parameters:")
    print(f"{list(self.model.kin_param_dict.keys())}")
    print(f"{self.model.kin_params_list}")
    print(f"Objective function: {self.model.neg_logLik}")
    print("")
    clear_output(wait=True)
    return self.model.neg_logLik

def fit(self, method = 'Nelder-Mead', maxfev_per_param = 200,
        train_size = 0.8, random_seed = None,
        file_name = "fit_results"):

    directory = os.path.dirname(f'{file_name}.xlsx')
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True) # Create intermediate directories

    fit_result = {'model': [], 'params': [], 'params_names': [],
                  'units': [], 'aic_c': [], 'aic': [],
                  'fun': [], 'nfev': [], 'success': [], 'message': []
            }
    # when fitting use only the trainning data to speed up
    self.trainning_data, self.test_data = train_test_split(self.kinetic_data.iloc[2:],
                                                                 train_size=train_size,
                                                                 random_state=random_seed)
    # reset indexes
    self.trainning_data = self.trainning_data.reset_index()
    self.test_data = self.test_data.reset_index()

    # put the trainning data into tmp_data, residuals are calculated using self.tmp_data
    self.tmp_data = self.trainning_data

    for model in self.models:
        # select the model
        self.model = self.models[model]

        self.model.minimum = minimize(
                                       self.objective_f,
                                       self.model.kin_params_list,
                                       method = method,
                                       bounds = self.model.kin_params_bnds_list,
                                       options={'maxfev': maxfev_per_param*len(self.model.kin_params_list)},
                                    )

        self.model.kin_params_list = self.model.minimum.x
        # store the minimized parameters in dict
        self.model.store_kin_params(self.model.kin_params_list)
        # store the results in the KineticModel object
        self.models[self.model.name].kin_param_dict = self.model.kin_param_dict
        self.models[self.model.name].aic_c          = self.model.aic_c
        self.models[self.model.name].aic            = self.model.aic
        self.models[self.model.name].minimum        = self.model.minimum
        with open(f'{file_name}_{self.model.name}_fitted_model_params_dict.pkl', 'wb') as f:
            pickle.dump(self.model.kin_param_dict, f)

        fit_result['model'].append(self.model.name)
        fit_result['params'].append(list(self.model.kin_param_dict.values()))
        fit_result['params_names'].append(list(self.model.kin_param_dict.keys()))
        fit_result['units'].append(self.model.units)
        fit_result['aic_c'].append(self.model.aic_c)
        fit_result['aic'].append(self.model.aic)
        fit_result['fun'].append(self.model.minimum.fun)
        fit_result['nfev'].append(self.model.minimum.nfev)
        fit_result['success'].append(self.model.minimum.success)
        fit_result['message'].append(self.model.minimum.message)

    # after fitting is finished set the best model
    aic_c_dict = {model_name: model.aic_c for model_name, model in self.models.items()}
    self.best_model = min(aic_c_dict, key = aic_c_dict.get)
    self.best_model_params = self.models[self.best_model].kin_param_dict
    print(f"Best model according to AIC_c: {self.best_model}")
    print(f"Best model parameters: {self.best_model_params}")
    self.fit_result = pd.DataFrame(fit_result)
    self.fit_result['relative_likelihood'] = exp**((self.fit_result['aic_c'].min()-self.fit_result['aic_c'])/2)

    # save the results
    with pd.ExcelWriter(f'{file_name}.xlsx',
        mode="w",
        engine="openpyxl",
        ) as writer:
            sheet_name = os.path.basename(file_name)
            self.fit_result.to_excel(writer, sheet_name=sheet_name)

    with open(f'{file_name}.pkl', 'wb') as f:
        pickle.dump(self.fit_result, f)

    return self.fit_result

def bootstrap(self, iterations = 10, train_size = 0.8, random_state = None,
              maxfev_per_param = None, method = 'Nelder-Mead',
              file_name = "bootstrap_results",
              confidence_level = 0.95):
    # random_state must be passed when working with parallel processes
    # otherwise processes inherit the random_state from the 1st process
    # making all processes sample the data equaly, defeating the purpose of parallelism

    # I will not give the ability for the user to bootstrap only one loaded model
    # this can lead to spurious comparisons of bootstrapped and non bootstrapped models
    # so if the user does not want to bootstrap he can just not load the model
    alpha = 1 - confidence_level # the probability a confidence interval will not include the population parameter

    # create dict to store all CIs and means
    CIs_dict = {}

    # initialize a dict to store the parameters distributions
    params_distribution = {}

    # random samples for bootstrapping will only be taken from trainning data
    # no bootstrapping is performed in the testing data
    self.trainning_data, self.test_data = train_test_split(self.kinetic_data.iloc[2:],
                                                           train_size=train_size, random_state = random_state)
    # reset indexes
    self.trainning_data = self.trainning_data.reset_index()
    self.test_data = self.test_data.reset_index()

    for model in self.models:
        # select the model
        self.model = self.models[model]

        # initialize a dict to store the parameters distributions
        params_distribution[model] = {key: [] for key in self.model.kin_param_dict.keys()}
        params_distribution[model]['aic'] = []
        params_distribution[model]['aic_c'] = []
        params_distribution[model]['fun'] = []
        params_distribution[model]['nfev'] = []
        params_distribution[model]['success'] = []
        params_distribution[model]['message'] = []

        # loop n_samples times
        for i in range(iterations):
            self.tmp_data = self.trainning_data.sample(frac = 1, replace = True,
                                                       ignore_index = True, random_state = random_state)

            # bootstrap MINIMIZATION
            minimum = minimize(
                               self.objective_f,
                               self.model.kin_params_list, # initial guess
                               method = method,
                               bounds = self.model.kin_params_bnds_list,
                               options={'maxfev': maxfev_per_param*len(self.model.kin_params_list)},
                               args = (bootstrap := True, i := i),)

            for i, key in enumerate(self.model.kin_param_dict):
                params_distribution[model][key].append(minimum.x[i])
            params_distribution[model]['aic'].append(self.model.aic)
            params_distribution[model]['aic_c'].append(self.model.aic_c)
            params_distribution[model]['fun'].append(minimum.fun)
            params_distribution[model]['nfev'].append(minimum.nfev)
            params_distribution[model]['success'].append(minimum.success)
            params_distribution[model]['message'].append(minimum.message)

        # clear params list
        self.model.kin_params_list = []
        # get the confidence intervals
        for key in self.model.kin_param_dict.keys():
            param_mean = np.mean(params_distribution[model][key])
            CIs_dict[f'{key}_{model}'] = {'lower CI':             np.quantile(params_distribution[model][key], q = alpha/2),
                                          'mean'    :             param_mean,
                                          'upper CI':             np.quantile(params_distribution[model][key], q = 1 - alpha/2),
                                          'confidence level':     confidence_level,
                                          'bootstrap iterations': iterations,
                                          'train size':           train_size,
                                          'model':                model,
                                         }
            # store results
            self.model.kin_params_list.append(param_mean)
        # store the minimized parameters in the reactions dict
        self.model.store_kin_params(self.model.kin_params_list)
        # store the results in the KineticModel object
        self.models[model].kin_param_dict = self.model.kin_param_dict

        # calculate residuals again with the new bootstrapped parameters
        self.calc_residuals()
        # put the AICs in the CIs_dict too
        for key in self.model.kin_param_dict.keys():
            CIs_dict[f'{key}_{model}'] = CIs_dict[f'{key}_{model}'] | {
                'AIC_c': self.model.aic_c,
                'AIC'  : self.model.aic,
            }

    aic_c_dict = {model_name: model.aic_c for model_name, model in self.models.items()}
    best_boot_model = min(aic_c_dict, key = aic_c_dict.get)

    for key in CIs_dict:
        CIs_dict[key] = CIs_dict[key] | {'best bootstrap model': best_boot_model}

    CIs_df = pd.DataFrame(CIs_dict)

    directory = os.path.dirname(f'{file_name}.xlsx')
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True) # Create intermediate directories

    with pd.ExcelWriter(f'{file_name}.xlsx', mode="w", engine="openpyxl") as writer:
        CIs_df.to_excel(writer, sheet_name='confidence_intervals')
        for model in params_distribution:
            param_df = pd.DataFrame(params_distribution[model])
            sheet_name = os.path.basename(file_name)
            param_df.to_excel(writer, sheet_name=sheet_name)

    with open(f'{file_name}_CIs_df.pkl', 'wb') as f:
        pickle.dump(CIs_df, f)
    for model in params_distribution:
        with open(f'{file_name}_{model}_parameters_distribution.pkl', 'wb') as f:
            pickle.dump(pd.DataFrame(params_distribution[model]), f)

    return CIs_df

def parametric_study(self, parametric_dict, file_name = 'parametric_study', conversions = []):

    directory = os.path.dirname(file_name)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True) # Create intermediate directories

    param_study_dfs = {model: pd.DataFrame() for model in self.models}

    dimensional_vars = ('T0', 'P0', 'F0', 'U', 'D', 'Dp', 'Ta')

    dimensional_vars = [key for key in dimensional_vars if key in parametric_dict.keys()]

    units_dict = {}

    # separate units and param arrays in two separate dicts
    for key in dimensional_vars:
        units_dict[key] = parametric_dict[key][1]
        parametric_dict[key] = parametric_dict[key][0]

    # create permutations dicts from the params dict
    keys, values = zip(*parametric_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i in range(len(permutations_dicts)):
        print(f'Solution {i+1} of {len(permutations_dicts)}')
        print(permutations_dicts[i])
        clear_output(wait=True)
        # set the class attributes according to the permutations
        for key, value in permutations_dicts[i].items():
            if key in dimensional_vars:
                value = DVar(value, units_dict[key], self.units)
            setattr(self, key, value)
        # solve for the current attributes
        for model in self.models:
            self.model = self.models[model]
            self.solve(model)
            # put the parameters in the solution df
            for key, value in permutations_dicts[i].items():
                if key == 'x0':
                    for x0_key in value:
                        self.sol_dfs[model][f'x0_{x0_key}'] = value[x0_key]
                else:
                    if key in dimensional_vars:
                        self.sol_dfs[model][f'{key} ({units_dict[key]})'] = value
                    else:
                        self.sol_dfs[model][key] = value
            self.calc_conversions(conversions)
            param_study_dfs[model] = pd.concat((param_study_dfs[model], self.sol_dfs[model]))

    with pd.ExcelWriter(f'{file_name}.xlsx', mode="w", engine="openpyxl") as writer:
        for model in param_study_dfs:
            param_study_dfs[model].to_excel(writer, sheet_name=model)
    with open(f'{file_name}.pkl', 'wb') as f:
        pickle.dump(param_study_dfs, f)

    return param_study_dfs

def calc_conversions(self, components = []):
    # calculate conversions
    for model in self.sol_dfs:
        sol_df_model = self.sol_dfs[model]
        sol_df_model['X'] = None
        for component in components:
            component_sol_df = sol_df_model.loc[sol_df_model['variable'] == component].copy()
            component_sol_df['X'] = (component_sol_df['value'].iloc[0]
                                     - component_sol_df['value'])/component_sol_df['value'].iloc[0]
            # put the conversion in the sol_dfs dict
            self.sol_dfs[model].loc[self.sol_dfs[model]['variable'] == component, 'X'] = component_sol_df['X'].values

def plot(self, model_name = "",
         figsize = (10,6), leg_fontsize = 8, max_legends = 10, yscale = 'linear', leg_loc = [0, 0, 0, 0, 0],
         plot_conversions = []):
    # if the user does not select a model, plot the current model
    if model_name == "":
        print("No model selected, plotting the last model.")
        model_name = self.model.name
    # clear the figure before plotting again
    plt.clf()

    self.calc_conversions(plot_conversions)

    # num is the Figure number to force the subplots to be created always in the same figure
    fig, ax = plt.subplots(num = model_name, nrows = 2, ncols = 3)
    fig.set_size_inches(figsize)
    fig.suptitle(model_name)

    sns.lineplot(x = 'W', y = 'X',
                 data = self.sol_dfs[model_name].loc[self.sol_dfs[model_name]['X'].notna()],
                 hue = 'variable',
                 style = 'variable',
                 palette = 'bright',
                 ax = ax[0,0]
                )
    ax[0,0].legend(fontsize = leg_fontsize, loc = leg_loc[0])
    ax[0,0].set_xlabel('W')
    ax[0,0].set_ylabel('X')
    ax[0,0].set_ylim(None, 1.05)
    ax[0,0].grid('both')

    sns.lineplot(x = 'W', y = 'value',
                 data = self.sol_dfs[
model_name].loc[
                     (self.sol_dfs[model_name]['variable'] != 'T') &
                     (self.sol_dfs[model_name]['variable'] != 'P') &
                     (self.sol_dfs[model_name]['variable'] != 'Q'), :],
                 hue = 'variable',
                 style = 'variable',
                 palette = 'bright',
                 ax = ax[0,1]
                )
    if len(self.component_list) < max_legends:
        ax[0,1].legend(fontsize = leg_fontsize, loc = leg_loc[1])
    else:
        handles, labels = ax[0,1].get_legend_handles_labels()
        ax[0,1].legend(handles[0:max_legends], labels[0:max_legends], fontsize = leg_fontsize, loc = leg_loc[1])
        ax[0,1].set_yscale(yscale)
        print('Too many legend labels, limiting the size to 10, others are hidden.')
    ax[0,1].set_xlabel('W')
    ax[0,1].set_ylabel(f'F ({Unit("F", self.units)})')
    ax[0,1].grid('both')

    T_in_C_df = self.sol_dfs[model_name].loc[(self.sol_dfs[model_name]['variable'] == 'T'), :].copy()
    T_in_C_df['value'] = T_in_C_df['value'] - 273.15
    T_in_C_df['unit'] = 'C'
    sns.lineplot(x = 'W', y = 'value',
                 data = T_in_C_df,
                 hue = 'variable',
                 style = 'variable',
                 palette = 'bright',
                 ax = ax[0,2]
                )
    ax[0,2].legend(fontsize = leg_fontsize, loc = leg_loc[2])
    ax[0,2].set_xlabel('W')
    ax[0,2].set_ylabel('T (ºC)')
    ax[0,2].grid('both')

    sns.lineplot(x = 'W', y = 'value',
                 data = self.sol_dfs[model_name].loc[(self.sol_dfs[model_name]['variable'] == 'Q'), :],
                 hue = 'variable',
                 style = 'variable',
                 palette = 'bright',
                 ax = ax[1,0]
                )
    ax[1,0].legend(fontsize = leg_fontsize, loc = leg_loc[3])
    ax[1,0].set_xlabel('W')
    ax[1,0].set_ylabel(r'$Q_\text{cumulative}$' + f'({Unit("Power", self.units)})')
    ax[1,0].grid('both')

    sns.lineplot(x = 'W', y = 'value',
                 data = self.sol_dfs[model_name].loc[(self.sol_dfs[model_name]['variable'] == 'P'), :],
                 hue = 'variable',
                 style = 'variable',
                 palette = 'bright',
                 ax = ax[1,1]
                )
    ax[1,1].legend(fontsize = leg_fontsize, loc = leg_loc[4])
    ax[1,1].set_xlabel('W')
    ax[1,1].set_ylabel(f'P ({Unit("P", self.units)})')
    ax[1,1].set_ylim(0, self.P0.val*1.1)
    ax[1,1].grid('both')

    plt.tight_layout()
    plt.show()

def ploteq(self, model_name = "",
         figsize = (10,6), leg_fontsize = 8, max_legends = 10, yscale = 'linear', leg_loc = [0, 0, 0, 0, 0],
         plot_conversions = []):
    if len(self.sol_dfs_eq) == 0:
        raise ValueError('Found no equilibrium data. To calculate equilibrium set the "calculate_equilibrium" PBR object attribute to "True". To plot without equilibrium use the "plot" function.')
    # if the user does not select a model, plot the current model
    if model_name == "":
        print("No model selected, plotting the last model.")
        model_name = self.model.name
    # clear the figure before plotting again
    plt.clf()

    # calculate conversions
    for model in self.sol_dfs:
        sol_df_model = self.sol_dfs[model]
        sol_df_model['X'] = None
        sol_df_eq_model = self.sol_dfs_eq[model]
        sol_df_eq_model['X'] = None
        for component in plot_conversions:
            component_sol_df = sol_df_model.loc[sol_df_model['variable'] == component].copy()
            component_sol_df_eq = sol_df_eq_model.loc[sol_df_eq_model['variable'] == component].copy()
            component_sol_df['X'] = (component_sol_df['value'].iloc[0]
                                     - component_sol_df['value'])/component_sol_df['value'].iloc[0]
            component_sol_df_eq['X'] = (component_sol_df['value'].iloc[0]
                                     - component_sol_df_eq['value'])/component_sol_df['value'].iloc[0]
            # put the conversion in the sol_dfs dict
            self.sol_dfs[model].loc[self.sol_dfs[model]['variable'] == component, 'X'] = component_sol_df['X'].values
            self.sol_dfs_eq[model].loc[self.sol_dfs_eq[model]['variable'] == component, 'X'] = component_sol_df_eq['X'].values

    # num is the Figure number to force the subplots to be created always in the same figure
    fig, ax = plt.subplots(num = model_name, nrows = 2, ncols = 3)
    fig.set_size_inches(figsize)
    fig.suptitle(model_name)

    sns.lineplot(x = 'W', y = 'X',
                 data = self.sol_dfs[model_name].loc[self.sol_dfs[model_name]['X'].notna()],
                 hue = 'variable',
                 palette = 'bright',
                 linestyle = '-',
                 ax = ax[0,0]
                )
    sns.lineplot(x = 'W', y = 'X',
                 data = self.sol_dfs_eq[model_name].loc[self.sol_dfs_eq[model_name]['X'].notna()],
                 hue = 'variable',
                 palette = 'bright',
                 linestyle = '--',
                 ax = ax[0,0],
                 legend = False,
                )
    ax[0,0].set_xlabel('W')
    ax[0,0].set_ylabel('X')
    ax[0,0].set_ylim(None, 1.05)
    ax[0,0].grid('both')
    if len(self.component_list) < max_legends:
        ax[0,0].legend(fontsize = leg_fontsize, loc = leg_loc[0])
    else:
        handles, labels = ax[0,0].get_legend_handles_labels()
        ax[0,0].legend(handles[0:max_legends], labels[0:max_legends], fontsize = leg_fontsize, loc = leg_loc[0])
        ax[0,0].set_yscale(yscale)
        print(f'Too many legend labels, limiting the size to {max_legends}, others are hidden.')

    # plot molar flows
    sns.lineplot(x = 'W', y = 'value',
                 data = self.sol_dfs[model_name].loc[
                     (self.sol_dfs[model_name]['variable'] != 'T') &
                     (self.sol_dfs[model_name]['variable'] != 'P') &
                     (self.sol_dfs[model_name]['variable'] != 'Q'), :],
                 hue = 'variable',
                 palette = 'bright',
                 linestyle = '-',
                 ax = ax[0,1]
                )
    sns.lineplot(x = 'W', y = 'value',
                 data = self.sol_dfs_eq[model_name].loc[
                     (self.sol_dfs_eq[model_name]['variable'] != 'T') &
                     (self.sol_dfs_eq[model_name]['variable'] != 'P') &
                     (self.sol_dfs_eq[model_name]['variable'] != 'Q'), :],
                 hue = 'variable',
                 palette = 'bright',
                 linestyle = '--',
                 ax = ax[0,1],
                 legend = False,
                )
    if len(self.component_list) < max_legends:
        ax[0,1].legend(fontsize = leg_fontsize, loc = leg_loc[1])
    else:
        handles, labels = ax[0,1].get_legend_handles_labels()
        ax[0,1].legend(handles[0:max_legends], labels[0:max_legends], fontsize = leg_fontsize, loc = leg_loc[1])
        ax[0,1].set_yscale(yscale)
        print(f'Too many legend labels, limiting the size to {max_legends}, others are hidden.')
    ax[0,1].set_xlabel('W')
    ax[0,1].set_ylabel(f'F ({Unit("F", self.units)})')
    ax[0,1].grid('both')

    T_in_C_df = self.sol_dfs[model_name].loc[(self.sol_dfs[model_name]['variable'] == 'T'), :].copy()
    T_in_C_df['value'] = T_in_C_df['value'] - 273.15
    T_in_C_df['unit'] = 'C'
    sns.lineplot(x = 'W', y = 'value',
                 data = T_in_C_df,
                 hue = 'variable',
                 style = 'variable',
                 palette = 'bright',
                 ax = ax[0,2]
                )
    ax[0,2].legend(fontsize = leg_fontsize, loc = leg_loc[2])
    ax[0,2].set_xlabel('W')
    ax[0,2].set_ylabel('T (ºC)')
    ax[0,2].grid('both')

    sns.lineplot(x = 'W', y = 'value',
                 data = self.sol_dfs[model_name].loc[(self.sol_dfs[model_name]['variable'] == 'Q'), :],
                 hue = 'variable',
                 style = 'variable',
                 palette = 'bright',
                 ax = ax[1,0]
                )
    ax[1,0].legend(fontsize = leg_fontsize, loc = leg_loc[3])
    ax[1,0].set_xlabel('W')
    ax[1,0].set_ylabel(r'$Q_\text{cumulative}$' + f'({Unit("Power", self.units)})')
    ax[1,0].grid('both')

    sns.lineplot(x = 'W', y = 'value',
                 data = self.sol_dfs[model_name].loc[(self.sol_dfs[model_name]['variable'] == 'P'), :],
                 hue = 'variable',
                 style = 'variable',
                 palette = 'bright',
                 ax = ax[1,1]
                )
    ax[1,1].legend(fontsize = leg_fontsize, loc = leg_loc[4])
    ax[1,1].set_xlabel('W')
    ax[1,1].set_ylabel(f'P ({Unit("P", self.units)})')
    ax[1,1].set_ylim(0, self.P0.val*1.1)
    ax[1,1].grid('both')

    plt.tight_layout()
    plt.show()

def plot_parity(self, ncols, nrows, figsize,
                leg_fontsize = 8, max_legends = 10, scales = 'linear',
                file_name = None):

    def plot(title:str, residuals_df):
        fig_title = title
        fig, subplots = plt.subplots(nrows, ncols, figsize=(figsize[0],figsize[1]*nrows), num = fig_title)
        fig.suptitle(fig_title)
        # when there is more than one line
        # subplots is a 2D array (e.g., [[ax1, ax2], [ax3, ax4]])
        # so we have to flatten the subplots array into a 1D iterable list of axes objects
        for ax, model_name in zip(subplots.flatten(), self.models.keys()):
            plot_df = residuals_df.loc[residuals_df['model'] == model_name]
            Fmax = plot_df.select_dtypes(include = 'number').max().max()
            sns.scatterplot(x = 'Fexp', y = 'Fmodel',
                            data = plot_df,
                            hue = 'component', style = 'component', palette = 'bright', ax = ax)
            ax.plot((0,Fmax*1.1), (0,Fmax*1.1), c = 'gray')
            ax.set_title(f"{model_name}")
            ax.set_xlabel(r'$F_\mathrm{experimental}$' + f'({Unit("F", self.units)})')
            ax.set_ylabel(r'$F_\mathrm{model}$' + f'({Unit("F", self.units)})')
            ax.grid('both')
            ax.set_xscale(scales)
            ax.set_yscale(scales)
            ax.ticklabel_format(style='sci', scilimits=(0,0))
            if len(self.component_list) < max_legends:
                ax.legend(fontsize = leg_fontsize)
            else:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[0:max_legends], labels[0:max_legends], fontsize = leg_fontsize)
                print(f'Too many legend labels, limiting the size to {max_legends}, others are hidden.')

        plt.tight_layout()
        if file_name is not None:
            plt.savefig(f'{file_name}_{title}.pdf')
        plt.show()

    # if user passes a directory file path create it if it does not exist
    directory = os.path.dirname(f'{file_name}.xlsx')
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True) # Create intermediate directories

    # calculate the residuals
    trainning_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for model in self.models:
        self.model = self.models[model]
        self.tmp_data = self.trainning_data
        self.calc_residuals()
        trainning_df = pd.concat((trainning_df, pd.DataFrame(self.residuals)))

        self.tmp_data = self.test_data
        self.calc_residuals()
        test_df = pd.concat((test_df, pd.DataFrame(self.residuals)))
    plot('Training data parity plot', trainning_df)
    plot('Testing data parity plot', test_df)

    if file_name is not None:
        with open(f'{file_name}_training_df.pkl', 'wb') as f:
            pickle.dump(trainning_df, f)
        with open(f'{file_name}_test_df.pkl', 'wb') as f:
            pickle.dump(test_df, f)

def load_data(self, path):
    try:
        self.kinetic_data = pd.read_csv(path)
        # filling index numbers
        self.kinetic_data.loc[2:len(self.kinetic_data) - 1, 'index'] = list(range(len(self.kinetic_data) - 2))
        self.kinetic_data = self.kinetic_data.replace(to_replace=',', value='.', regex=True)
        # if data not numeric: convert to numeric
        self.kinetic_data.iloc[2:] = self.kinetic_data.iloc[2:].apply(pd.to_numeric, axis=0)
        for col_title in self.kinetic_data.columns:
            component = self.kinetic_data.loc[0, col_title]
            # if component not in component list, append it
            if 'Fin' in col_title:
                self.kinetic_data = self.kinetic_data.rename(columns = {col_title: f"Fin_{component}"})
                # if component not in component list, append it
                if component not in self.component_list:
                    self.component_list.append(component)
            elif 'Fout' in col_title:
                self.kinetic_data = self.kinetic_data.rename(columns = {col_title: f"Fout_{component}"})
                if component not in self.component_list:
                    self.component_list.append(component)

        # populate the thermo constants
        self.get_thermo_constants()
    except TypeError as error:
        print(error)

def get_thermo_constants(self):
    self.thermodynamic_constants = {}
    self.Cp_gases_objs = {}
    self.Cp_liq_objs = {}
    self.viscosity_gas_objs = {}
    for component in self.component_list:
        self.thermodynamic_constants[component] = ChemicalConstantsPackage.constants_from_IDs([component])
        self.Cp_gases_objs[component] = HeatCapacityGas(CASRN=self.thermodynamic_constants[component].CASs[0])
        self.Cp_liq_objs[component] = HeatCapacityLiquid(CASRN=self.thermodynamic_constants[component].CASs[0])
        self.viscosity_gas_objs[component] = ViscosityGas(
            CASRN  = self.thermodynamic_constants[component].CASs[0],
            MW     = self.thermodynamic_constants[component].MWs[0],
            Tc     = self.thermodynamic_constants[component].Tcs[0],
            Pc     = self.thermodynamic_constants[component].Pcs[0],
            Zc     = self.thermodynamic_constants[component].Zcs[0],
            dipole = self.thermodynamic_constants[component].dipoles[0],
            method = self.method_gas_viscosity,
            )
    MWs                 = [prop.MWs[0]                 for prop in self.thermodynamic_constants.values()]
    molecular_diameters = [prop.molecular_diameters[0] for prop in self.thermodynamic_constants.values()]
    Stockmayers         = [prop.Stockmayers[0]         for prop in self.thermodynamic_constants.values()]
    CASs                = [prop.CASs[0]                for prop in self.thermodynamic_constants.values()]
    self.viscosity_gas_mix = ViscosityGasMixture(
            MWs                   = MWs,
            molecular_diameters   = molecular_diameters,
            Stockmayers           = Stockmayers,
            CASs                  = CASs,
            ViscosityGases        = list(self.viscosity_gas_objs.values()),
            correct_pressure_pure = True)

def create_csv_template(self):
    import csv

    with open('data_template.csv', 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # datawriter.writerow(csv_heading)
        csvfile.write(TEMPLATE_DATA)

def calc_rx_rates(self, T, P, x, F):
    """This function calculates reaction rates
    At each iteration it is called to update the 'reaction_rates' in the PBR.reactions object
    """
    ''
    for reaction in self.model.reactions:

        eta = self.model.reactions[reaction].eff_factor
        K_eq = self.model.reactions[reaction].equilibrium_constant
        R = self.R.val # give R to the user in the correct units
        kin_param_dict = self.model.kin_param_dict
        data = self.tmp_data_to_rate_fcn
        stoic = sum(self.model.reactions[reaction].stoic.values())
        P_ref = {'SI': 1e5, 'cgs': 1e6}
        product_x = 1
        for component in self.model.reactions[reaction].component_names:
            stoic_i = self.model.reactions[reaction].stoic[component]
            product_x = product_x * x[component]**stoic_i
        # the distance from equilibrium
        eq_distance = min(1, ( P / P_ref[self.units] )**stoic * product_x / K_eq)
        eq_distance = 1 - eq_distance
        self.model.reactions[reaction].eq_distance = eq_distance
        # print(epsilon)
        # clear_output(wait=True)

        try:
            rate = eta*self.model.reactions[reaction].rate_function(kin_param_dict, T, P, x, F, K_eq, eq_distance, R, data)
        except Exception as exception:
            warnings.warn(f'Warning: Somethig went wrong with the reaction rate calculation. Setting the reaction rate to zero. Exception: {exception}')
            rate = 0
        rate_unit = self.model.reactions[reaction].rate_unit
        # do not call DVarType to store the rate
        # must create a DVar to convert the rate units provided by the user by the PBR class unit system
        self.model.reactions[reaction].reaction_rate = DVar(rate, rate_unit, self.units)

def calc_Cps_gases_at_T(self, T):
    self.Cp_gases_at_T = {}
    for component in self.component_list:
        Cp = self.Cp_gases_objs[component](T = T)
        # convert to the unit system
        # the themo pkg uses J/mol/K, if the system is other
        # the DVar obj will handle the convertion upon initialization
        Cp = DVar(Cp, 'J/mol/K', self.units)
        self.Cp_gases_at_T[component] = Cp

def DVarType(value, var_ID, units_system):
    """DVarType creates DVars of type var_ID
    Ex: to create temperature DVar: DVarType(25, 'T', 'SI') will
    create a DVar to represent 25 C in SI, which is 298 K"""
    return DVar(value, Unit(var_ID, units_system), units_system)

def add_model(self, model):
    if model.name in self.models.keys():
        raise ValueError(f"A model named '{model.name}' already exists.")

    self.models[model.name] = model

    for reaction in model.reactions.values():
        for component in reaction.component_names:
            if component not in self.component_list:
                self.component_list.append(component)
                # insert the component as 0 in x0 and Fi0, as the user did not provide a value
                print(f'x0 not provided for {component}, assuming value is 0.')
                self.x0[component] = 0
                self.Fi0[component] = 0

class PBR:
    """This PBR reactor class is for modeling and kinetic fitting of a heat exchancher type packed bed reactor
    Attributes:
        T0            type=float       Inlet temperature
        P0            type=float       Inlet pressure
        x0            type=list        Inlet molar composition
        F0            type=float       Inlet Feed molar flow
        isothermal    type=boolean     If True the reactor is isothermal, if not it is a heat exchanger reactor
        liquid_phase  type=boolean     This class only supports one phase, if it is liquid set this to 
                                       True, default is False (gas phase)
        ergun         type=boolean     If True calculate the pressure drop using the Ergun equation
        reactions     type=dict        The reactions dictionary, set by the 'add_reaction' method
        W             type=float       Total catalyst mass
        W_steps       type=int         Number of steps for model integration
                                       W_steps controls the point in which to save the solution,
                                       not the steps the solver takes.
                                       Ex: if one is interested only in reactor input and output,
                                       one can set W_steps to 2, which will only store the solution
                                       at W = 0 and W = WT
                                       The solver step is determined by the error.
                                       Set rtol and atol to change the error tolerances.
        U             type=float       The overall heat exchange coefficient, default is 0, optional for isothermal reactor
    """
    def update_reactions_HSG(self, T):
        for reaction in self.model.reactions:
            # update the reaction temperature before calculating HSG
            self.model.reactions[reaction].liquid_phase = self.liquid_phase
            self.model.reactions[reaction].T = T
            self.model.reactions[reaction].calc_reaction_HSG()
    def update_bed_porosity(self):
        # if no bed porosity provided calculate it using Pushnov's correlation
        # A. S. Pushnov, Chem. Pet. Eng. 2006, 42 (1), 14–17.
        # https://doi.org/10.1007/s10556-006-0045-x
        self.bed_porosity = 0.9198/((self.D.val/self.Dp.val)**2) + 0.3414
    def update_bed_packing(self):
        self.bed_packing = DVarType((1-self.bed_porosity)*self.pellet_density.val, 'd', self.units)

    def __init__(self,
                # optional args
                 units = 'SI',
                 # obs: do not declare DVar here, only its arguments (they will by passed to DVar upon init)
                 T0 = (25, 'C'), P0 = (1, 'bar'), x0 = {}, F0 = (1, 'mol/s'),
                 isothermal = False, ergun = False, energy_balance = True,
                 WT = (1, 'kg'), W_steps = 100,
                 U = (100, 'W/m2/K'),
                 bed_porosity = None, pellet_density = (2530, 'kg/m3'), bed_packing = None,
                 D = (10, 'cm'), Dp = (4, 'mm'), Ta = (25, 'C'),
                 R = (8.314, 'J/mol/K'),
                 liquid_phase = False,
                 solver_method = 'RK23', rtol = 1e-4, atol= 1e-6,
                 method_gas_viscosity = 'GHARAGHEIZI', method_gas_viscosity_mix = 'WILKE', log = False,
                 calculate_equilibrium = False, equilibrium_obj_fun_tolerance = 1e-12,
                 minimal_F = 1e-32,
                ):
        self.log = log
        if self.log:
            logging.basicConfig(filename='log.log',
                                level=logging.INFO,
                                # level=logging.WARNING,
                               )
            logging.captureWarnings(True)

        # add methods
        PBR.solve = solve
        PBR.solve_liq = solve_liq
        PBR.solve_gas = solve_gas
        PBR.plot = plot
        PBR.load_data = load_data
        PBR.calc_Cps_gases_at_T = calc_Cps_gases_at_T
        PBR.calc_rx_rates = calc_rx_rates
        PBR.calc_residuals = calc_residuals
        PBR.plot_parity = plot_parity
        PBR.calc_neg_logLik = calc_neg_logLik
        PBR.objective_f = objective_f
        PBR.fit = fit
        PBR.get_thermo_constants = get_thermo_constants
        PBR.add_model = add_model
        PBR.bootstrap = bootstrap
        PBR.ploteq = ploteq
        PBR.calc_conversions = calc_conversions
        PBR.parametric_study = parametric_study

        self.valid_unit_systems = ('SI', 'cgs')
        self.units = units # if units changed must reset it
        # The * operator unpacks iterables into function arguments
        self.T0 = DVar(*T0, self.units) # self.units passes the unit system to DVar, DVar always converts to self.units
        self.P0 = DVar(*P0, self.units)
        self.x0 = x0
        self.F0 = DVar(*F0, self.units)
        self.Fi0 = {component: self.F0.val*self.x0[component] for component in self.x0}
        self.Q0 = DVar(0, 'W', self.units)
        self.WT = DVar(*WT, self.units)
        self.U  = DVar(*U, self.units)
        self.D  = DVar(*D, self.units)
        self.Dp = DVar(*Dp, self.units)
        self.Ta = DVar(*Ta, self.units)
        self.pellet_density = DVar(*pellet_density, self.units)
        self.R = DVar(*R, self.units)
        self.isothermal = isothermal
        self.energy_balance = energy_balance
        self.liquid_phase = liquid_phase
        self.ergun = ergun
        self.W_steps = W_steps # number of steps for model integration
        self.models = {} # stores the current selected model
        self.sol_dfs = {}
        self.sol_dfs_eq = {}
        self.Fi_out_models = {}
        # generate component names list
        self.component_list = list(self.x0.keys()) # this is a list of all the components, more can be added by methods, such as when adding reactions
        self.dF_dW_i = {} # molar material balance dictionary
        create_csv_template(self)
        self.solver_method = solver_method
        self.solver_rtol = rtol
        self.solver_atol = atol

        self.best_model = None
        self.best_model_params = {}

        self.method_gas_viscosity = method_gas_viscosity
        self.method_gas_viscosity_mix = method_gas_viscosity_mix

        self.tmp_data_to_rate_fcn = {}
        self.calculate_equilibrium = calculate_equilibrium
        self.equilibrium_obj_fun_tolerance = equilibrium_obj_fun_tolerance
        self.minimal_F = minimal_F

        self.bed_porosity = bed_porosity
        if self.bed_porosity is None:
            self.update_bed_porosity()
        self.bed_packing = bed_packing
        if self.bed_packing is None:
            self.update_bed_packing()

    def __str__(self):
        description = f"T0: {self.T0}, P0: {self.P0}, F0: {self.F0}\n"
        description += f"x0: {self.x0}\n"
        description += f"W: {self.WT}, W_steps: {self.W_steps}\n"
        description += f"isothermal: {self.isothermal}, ergun: {self.ergun}, liquid_phase: {self.liquid_phase}\n"
        if (not self.isothermal) & (not self.ergun):
            description += f"U: {self.U}, D: {self.D}, Ta: {self.Ta}\n"
        if (not self.isothermal) & (self.ergun):
            description += f"D: {self.D}, Dp: {self.Dp}\n"
            description += f"bed porosity: {self.bed_porosity}, pellet density: {self.pellet_density}\n"
            description += f"bed packing: {self.bed_packing}\n"
        description += f"components: {self.component_list}\n"
        description += f"Kinetic models: {list(self.models.keys())}\n"
        if self.best_model is not None:
            description += f"Best model: {self.best_model}\n"
            description += f"Best model parameters: {self.best_model_params}\n"
        return description
