import clr 
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
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
    from DWSIM.Automation import Automation3

    # Initialize Automation and Load Flowsheet locally for this process
    interf = Automation3()
    Flowsheet = interf.LoadFlowsheet(sim_path)
    
    # Get necessary objects
    H2 = Flowsheet.GetFlowsheetSimulationObject('H2').GetAsObject()
    CO = Flowsheet.GetFlowsheetSimulationObject('CO').GetAsObject()
    compressor = Flowsheet.GetFlowsheetSimulationObject('C-1').GetAsObject()
    cooler = Flowsheet.GetFlowsheetSimulationObject('CL-1').GetAsObject()
    C1_C8 = Flowsheet.GetFlowsheetSimulationObject('C1-C8').GetAsObject()
    C9_C15 = Flowsheet.GetFlowsheetSimulationObject('C9-C15').GetAsObject()
    C16p = Flowsheet.GetFlowsheetSimulationObject('C16p').GetAsObject()
    unreacted_syngas = Flowsheet.GetFlowsheetSimulationObject('unreacted syngas').GetAsObject()
    CO2 = Flowsheet.GetFlowsheetSimulationObject('CO2').GetAsObject()
    H2O = Flowsheet.GetFlowsheetSimulationObject('H2O').GetAsObject()
    syncrude = Flowsheet.GetFlowsheetSimulationObject('syncrude').GetAsObject()
    syncrude_phase = syncrude.GetPhase('Overall')
    PFR_1 = Flowsheet.GetFlowsheetSimulationObject('PFR-1').GetAsObject()
    E_reactor = Flowsheet.GetFlowsheetSimulationObject('E1').GetAsObject()
    PFR_1.set_dV(0.02)

    phi_T = row['phi_220C'] * ( 
        ( 493 / row['T_K'] ) * np.exp( - ( 147203.9 / 8.314 ) * ( 1/row['T_K'] - 1/493 ) ) 
    ) ** (1/2)
    eff = np.tanh(phi_T)/phi_T

    # hacking the catalyst load to set the effectiveness factor, can do this because r = r' * rho_c
    PFR_1.CatalystLoading = 1648 * eff # kg/m³

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
    results['CO in mass flow (kg/d)'] = CO.GetMassFlow() * 86400
    results['H2 in mass flow (kg/d)'] = H2.GetMassFlow() * 86400
    results['C1-C8 mass flow (kg/d)'] = C1_C8.GetMassFlow() * 86400
    results['C9-C15 mass flow (kg/d)'] = C9_C15.GetMassFlow() * 86400
    results['C16p mass flow (kg/d)'] = C16p.GetMassFlow() * 86400
    results['unreacted syngas mass flow (kg/d)'] = unreacted_syngas.GetMassFlow() * 86400
    results['water mass flow (kg/d)'] = H2O.GetMassFlow() * 86400
    results['CO2 mass flow (kg/d)'] = CO2.GetMassFlow() * 86400
    
    for component in syncrude_phase.Compounds.keys():
        results[f'F_{component}_out_dwsim'] = syncrude_phase.Compounds[component].MolarFlow
    
    return index, results

# --- MAIN EXECUTION ---
def parametric_study(parametric_data, num_workers, dwsim_path, sim_path):
    # Determine number of workers (e.g., 4 or use os.cpu_count())
    num_workers = num_workers
    
    print(f"Starting parallel simulation with {num_workers} workers...")
    
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

    print("Simulations complete.")
    return parametric_data