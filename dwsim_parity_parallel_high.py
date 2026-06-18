import clr 
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

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

    print(index)

    # Initialize Automation and Load Flowsheet locally for this process
    interf = Automation3()
    flowsheet = interf.LoadFlowsheet(sim_path)
    
    # Get necessary objects
    h2 = flowsheet.GetFlowsheetSimulationObject('H2').GetAsObject()
    co = flowsheet.GetFlowsheetSimulationObject('CO').GetAsObject()
    compressor = flowsheet.GetFlowsheetSimulationObject('C-1').GetAsObject()
    cooler = flowsheet.GetFlowsheetSimulationObject('CL-1').GetAsObject()
    syncrude = flowsheet.GetFlowsheetSimulationObject('syncrude').GetAsObject()
    syncrude_phase = syncrude.GetPhase('Overall')
    PFR_1 = flowsheet.GetFlowsheetSimulationObject('PFR-1').GetAsObject()
    PFR_1.set_dV(0.005)

    # Set parameters
    h2.SetMolarFlow(row['F_H2_e_mol_s'])
    co.SetMolarFlow(row['F_CO_e_mol_s'])
    compressor.POut = row['P_abs_Pa']
    cooler.OutletTemperature = row['T_R_K']
    
    # Solve
    interf.CalculateFlowsheet2(flowsheet)

    # check if solved
    if not flowsheet.Solved:
        raise ValueError(f'Something went wrong at index {i}, check your simulation.')

    # Collect results
    results = {}
    for component in syncrude_phase.Compounds.keys():
        results[f'F_{component}_out_dwsim'] = syncrude_phase.Compounds[component].MolarFlow
    
    return index, results

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Load your data
    kinetic_data = pd.read_pickle('reconciled_data.pkl')
    kinetic_data = kinetic_data.loc[kinetic_data.CINÉTICA == True].copy()
    
    # Paths
    dwsim_path = r"C:\Users\user\AppData\Local\DWSIM\\"
    sim_path = r"Z:\GoogleDrive\uem\Doutorado\Ensaios\SAF\SAF_8_08_08_2025\SAF_kinetics_paper_code_with_WGS\dwsim_bench_model_high.dwxmz"
    
    # Determine number of workers (e.g., 4 or use os.cpu_count())
    num_workers = 4
    
    print(f"Starting parallel simulation with {num_workers} workers...")
    
    # Prepare the list of tasks
    tasks = [(idx, row, dwsim_path, sim_path) for idx, row in kinetic_data.iterrows()]
    
    # Execute in parallel
    results_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # We use a wrapper or starmap-like approach
        futures = [executor.submit(simulate_point, *task) for task in tasks]
        
        for future in futures:
            idx, res = future.result()
            # Update the dataframe with results
            for col, val in res.items():
                kinetic_data.loc[idx, col] = val

    # Save final results
    kinetic_data.to_pickle('kinetic_data_dwsim_high.pkl')
    kinetic_data.to_excel('kinetic_data_dwsim_high.ods')
    print("Simulations complete.")