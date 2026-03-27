import clr 
import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import ParameterGrid

# --- CONFIGURATION ---
# We define a worker function that will run in each process
def simulate_point(index, row, dwsim_path, sim_path):
    # Each process must import its own references and initialize its own Automation manager
    import clr
    import os
    from System import String
    
    # Add DWSIM references inside the worker
    clr.AddReference(os.path.join(dwsim_path, "DWSIM.Automation.dll"))
    clr.AddReference(os.path.join(dwsim_path, "DWSIM.Interfaces.dll"))
    clr.AddReference(os.path.join(dwsim_path, "ThermoCS\\ThermoCS.dll"))
    from DWSIM.Automation import Automation3

    # Initialize Automation and Load Flowsheet locally for this process
    interf = Automation3()
    Flowsheet = interf.LoadFlowsheet(sim_path)
    
    # Get necessary objects
    H2 = Flowsheet.GetFlowsheetSimulationObject('H2').GetAsObject()
    CO = Flowsheet.GetFlowsheetSimulationObject('CO').GetAsObject()
    compressor = Flowsheet.GetFlowsheetSimulationObject('C-1').GetAsObject()
    cooler = Flowsheet.GetFlowsheetSimulationObject('CL-1').GetAsObject()
    syncrude = Flowsheet.GetFlowsheetSimulationObject('syncrude').GetAsObject()
    syncrude_phase = syncrude.GetPhase('Overall')
    PFR_1 = Flowsheet.GetFlowsheetSimulationObject('PFR-1').GetAsObject()
    E_reactor = Flowsheet.GetFlowsheetSimulationObject('E1').GetAsObject()
    PFR_1.set_dV(0.02)
    PFR_1.Volume = row['V_m3'] # m3

    # hacking the catalyst load to set the effectiveness factor, can do this because r = r' * rho_c
    PFR_1.CatalystLoading = 1648 * row['eff'] # kg/m³

    # Set parameters
    H2.SetMolarFlow(row['F_H2_in'])
    CO.SetMolarFlow(row['F_CO_in'])
    compressor.POut = row['P_Pa']
    cooler.OutletTemperature = row['T_K']
    
    # Solve
    interf.CalculateFlowsheet2(Flowsheet)

    # check if solved
    if not Flowsheet.Solved:
        raise ValueError(f'Something went wrong at index {index}, check your simulation.')

    # Collect results
    results = {}
    results['Reactor Energy Flow (kW)'] = E_reactor.EnergyFlow
    
    for component in syncrude_phase.Compounds.keys():
        results[f'F_{component}_out_dwsim'] = syncrude_phase.Compounds[component].MolarFlow
    
    return index, results

# --- MAIN EXECUTION ---
def parametric_study(parametric_data, num_workers, dwsim_path, sim_path):
    # Determine number of workers (e.g., 4 or use os.cpu_count())
    num_workers = num_workers
    
    tqdm.write(f"Starting parallel simulation with {num_workers} workers...")
    
    # Prepare the list of tasks
    tasks = [(idx, row, dwsim_path, sim_path) for idx, row in parametric_data.iterrows()]
    
    # Execute in parallel
    results_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # We use a wrapper or starmap-like approach
        futures = [executor.submit(simulate_point, *task) for task in tasks]
        
        for future in tqdm(futures, total=len(futures)):
            idx, res = future.result()
            # Update the dataframe with results
            for col, val in res.items():
                parametric_data.loc[idx, col] = val

    tqdm.write("Simulations complete.")
    return parametric_data