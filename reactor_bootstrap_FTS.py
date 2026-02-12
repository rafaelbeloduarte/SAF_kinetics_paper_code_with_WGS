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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%


from math import e
from PBR import PBR
from Reaction import Reaction
from KineticModel import KineticModel
import multiprocessing
import pickle


# %%
from models_FTS.Yates            import Yates
from models_FTS.PowerLaw         import PowerLaw
from models_FTS.Botes            import Botes
from models_FTS.Ojeda            import Ojeda
from models_FTS.Mousavi          import Mousavi
from models_FTS.PowerLaw2        import PowerLaw2

# %%
Yates       =      Yates()
PowerLaw    =      PowerLaw()
Botes       =      Botes()
Ojeda       =      Ojeda()
Mousavi     =      Mousavi()
PowerLaw2   =      PowerLaw2()

# %%
model_names = (
                   'PowerLaw', 'Yates',
                   'Botes', 'Ojeda',
                   'Mousavi', 'PowerLaw2',
                  )


# %%
# load fitted models
d = {}
for model in model_names:
    with open(f'fit_results_FTS/{model}_fitted_model_params_dict.pkl', 'rb') as f:
        d[model] = pickle.load(f)


# %%


Yates.kin_param_dict = d['Yates']
Yates.store_kin_params(list(Yates.kin_param_dict.values()))

PowerLaw.kin_param_dict = d['PowerLaw']
PowerLaw.store_kin_params(list(PowerLaw.kin_param_dict.values()))

# Botes.kin_param_dict = d['Botes']
# Botes.store_kin_params(list(Botes.kin_param_dict.values()))

# Ojeda.kin_param_dict = d['Ojeda']
# Ojeda.store_kin_params(list(Ojeda.kin_param_dict.values()))

# Mousavi.kin_param_dict = d['Mousavi']
# Mousavi.store_kin_params(list(Mousavi.kin_param_dict.values()))

# PowerLaw2.kin_param_dict = d['PowerLaw2']
# PowerLaw2.store_kin_params(list(PowerLaw2.kin_param_dict.values()))

# %%
def run_single_bootstrap(args):
    reactor_kwargs, run = args
    reactor = PBR(**reactor_kwargs)
    reactor.load_data('data_FTS.csv')
    # reactor.add_model(PowerLaw)
    reactor.add_model(Yates)
    # reactor.add_model(Botes)
    # reactor.add_model(vanSteen)
    # reactor.add_model(Ojeda)
    # reactor.add_model(Mousavi)
    # reactor.add_model(MousaviPLaw)
    # reactor.add_model(Wang)
    # reactor.add_model(WangSimple)
    # reactor.add_model(Elementary)
    CIs_df = reactor.bootstrap(iterations = 4, train_size = 0.5, random_state = None,
              maxfev_per_param = 1, method = 'Nelder-Mead',
              file_name = f"bootstrap_results_FTS/Yates_run_{run}",
              confidence_level = 0.95)
    return CIs_df


# %%


if __name__ == '__main__':

    reactor_kwargs = {'isothermal'    : True,
                      'ergun'         : False,
                      'energy_balance': False,
                      'W_steps'       : 2,
                      'solver_method' : 'BDF',
                      'rtol'          : 1e-3,
                      'atol'          : 1e-5,
                      'log'           : True,
    }

    all_runs_input = [(reactor_kwargs, run) for run in range(24)]

    # --- Parallel Execution ---

    # Get the number of CPU cores available for parallel processing
    num_processes = multiprocessing.cpu_count()
    if len(all_runs_input) < num_processes:
        num_processes = len(all_runs_input)
    print(f"Using {num_processes} cores for parallel processing.")

    # Create a Pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # pool.map() applies the run_single_bootstrap function to every item
        # in the all_runs_input list and waits for results.
        results = pool.map(run_single_bootstrap, all_runs_input)


# %%
