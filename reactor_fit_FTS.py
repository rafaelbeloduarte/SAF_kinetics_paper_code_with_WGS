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
def run_single_fit(args):
    reactor_kwargs, model_name = args
    random_seed = 3
    reactor = PBR(**reactor_kwargs)
    reactor.load_data('data_FTS.csv')
    match model_name:
        case 'PowerLaw':
            reactor.add_model(PowerLaw)
        case 'Yates':
            reactor.add_model(Yates)
        case 'Botes':
            reactor.add_model(Botes)
        case 'Ojeda':
            reactor.add_model(Ojeda)
        case 'Mousavi':
            reactor.add_model(Mousavi)
        case 'PowerLaw2':
            reactor.add_model(PowerLaw2)
    fit_result = reactor.fit(method = 'Nelder-Mead', maxfev_per_param = 200, 
            train_size = 0.7, random_seed = random_seed,
            file_name = f"fit_results_FTS/{model_name}"
           )
    return fit_result


# %%
if __name__ == '__main__':

    model_names = (
                   'PowerLaw', 'Yates',
                   'Botes', 'Ojeda',
                   'Mousavi', 'PowerLaw2',
                  )
    
    reactor_kwargs = {'isothermal'    : True,
                      'ergun'         : False,
                      'energy_balance': False,
                      'W_steps'       : 2,
                      'solver_method' : 'BDF',
                      'rtol'          : 1e-3,
                      'atol'          : 1e-5,
                      'log'           : True,
    }
    
    all_runs_input = [(reactor_kwargs, model) for model in model_names]
    
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
        results = pool.map(run_single_fit, all_runs_input)

# %%
