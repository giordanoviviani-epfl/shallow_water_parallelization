## Import libraries
import numpy as np
from pathlib import Path
import visualize as vis
import time
import yaml
import csv

## Functions ------------------------------------------------------------------------------------------------
def calculate_time_step(h, hu, hv, Tn, fast, optimize):
    global G
    if optimize:
        GH_factor = np.sqrt(G*h)
        huh = hu/h
        hvh = hv/h
        under_root = 2*np.max(((np.maximum(np.abs(huh+GH_factor), np.abs(huh-GH_factor)))**2+
                      (np.maximum(np.abs(hvh+GH_factor), np.abs(hvh-GH_factor)))**2))
        dT = DeltaX/(np.sqrt(under_root))
    else:
        # Calculate the time step
        under_root = ((np.maximum(np.abs(hu/h + np.sqrt(G*h)), np.abs(hu/h - np.sqrt(G*h))))**2 +
                        (np.maximum(np.abs(hv/h + np.sqrt(G*h)), np.abs(hv/h - np.sqrt(G*h))))**2)
        if fast:
            # Square rott is calculateon the scalar and not the vector
            nu = np.sqrt(np.max(under_root))
            dT = DeltaX / (np.sqrt(2) * nu)
        else:
            nu = np.max(np.sqrt(under_root))
            dT = DeltaX / (np.sqrt(2) * nu)

    # Check if the time step is too large (and exceed the time limit) and adjust it
    if Tn + dT > Tend:
        dT = Tend - Tn
    
    return dT

def compute_step(ht, hut, hvt, h, hu, hv, zdx, zdy, dT, DeltaX, idx_i, idx_j):
    global G
    
    # Compute the constant factors
    mult_factor = 0.5 * dT / DeltaX 

    # Define useful functions
    first_term = lambda matrix: 0.25 * (matrix[idx_i[0], idx_j[-1]] + matrix[idx_i[0], idx_j[1]] + matrix[idx_i[-1], idx_j[0]] + matrix[idx_i[1], idx_j[0]])
    second_term = lambda matrix: dT * G * h[idx_i[0], idx_j[0]] * matrix[idx_i[0], idx_j[0]] # for hu and hv matrices (variables are zdx and zdy) 
    ratios_difference = lambda i1, j1, i2, j2: mult_factor * ((hut[i1, j1] * hvt[i1, j1])/ht[i1, j1] - (hut[i2, j2] * hvt[i2, j2])/ht[i2, j2])
    sum_with_g = lambda mat, i1, j1: ((mat[i1, j1]**2) / ht[i1, j1]) + 0.5*G*(ht[i1, j1]**2)  
    g_difference = lambda mat, i1, j1, i2, j2: mult_factor * (sum_with_g(mat, i1, j1) - sum_with_g(mat, i2, j2))
    #g_difference2 = lambda mat, i1, j1, i2, j2: mult_factor * (((mat[i1, j1]**2) / ht[i1, j1]) + 0.5*G*(ht[i1, j1]**2) - ((mat[i2, j2]**2) / ht[i2, j2]) - 0.5*G*(ht[i2, j2]**2))

    # Compute the fluxes
    h[idx_i[0], idx_j[0]] = first_term(ht) + mult_factor * (hut[idx_i[0], idx_j[-1]] - hut[idx_i[0], idx_j[1]] + hvt[idx_i[-1], idx_j[0]] - hvt[idx_i[1], idx_j[0]])

    hu[idx_i[0], idx_j[0]] =  first_term(hut) - second_term(zdx) + ratios_difference(idx_i[-1], idx_j[0], idx_i[1], idx_j[0]) + g_difference(hut, idx_i[0], idx_j[-1], idx_i[0], idx_j[1])

    hv[idx_i[0], idx_j[0]] = first_term(hvt) - second_term(zdy) + ratios_difference(idx_i[0], idx_j[-1], idx_i[0], idx_j[1]) + g_difference(hvt, idx_i[-1], idx_j[0], idx_i[1], idx_j[0])


## End Functions --------------------------------------------------------------------------------------------
def simulation():
    ## Start timer
    start_time = time.time()

 
    ## Load the data
    # Initial conditions
    h = np.fromfile(open(data_file.with_name(data_file.name + "_h.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T
    hu = np.fromfile(open(data_file.with_name(data_file.name + "_hu.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T
    hv = np.fromfile(open(data_file.with_name(data_file.name + "_hv.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T

    # Topography
    zdx = np.fromfile(open(data_file.with_name(data_file.name + "_Zdx.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T
    zdy = np.fromfile(open(data_file.with_name(data_file.name + "_Zdy.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T

    ## Initialize empty arrays for the values on the grid (t = temporary)
    ht = np.zeros((Nx,Nx))
    hut = np.zeros((Nx,Nx))
    hvt = np.zeros((Nx,Nx)) 

    # Slices of the arrays of the matrices need for the calculations
    # First element is the centrated matrix (i), whereas the second is the positively shifted matrix (i+1), and the third is the negatively shifted matrix (i-1) 
    idx_i = [slice(1, Nx-1), slice(2, Nx), slice(0, Nx-2)]
    idx_j = [slice(1, Nx-1), slice(2, Nx), slice(0, Nx-2)]

    ## Compute the solution
    Tn = 0 # Current time
    n_steps = 0 # Time step\
    fast=True
    optimize=True

    ## Get initialization time
    initialization_time = time.time() - start_time

    ##Start loop time
    start_loop_time = time.time()

    ## While loop -----------------------------------------------------------------------------------------------
    while Tn < Tend:

        dT = calculate_time_step(h, hu, hv, Tn, fast, optimize)

        # Report Status
        if n_steps % 50 == 0:
            print(f'{n_steps:04d} - Computing T: {Tn + dT} ({100 * (Tn + dT) / Tend}%) - dT: {dT} hr - exc. time {time.time() - start_time}')

        # Copy solution to temporary variables
        # ht = h.copy()
        # hut = hu.copy()
        # hvt = hv.copy()
        ht = h
        del h
        hut = hu
        del hu
        hvt = hv
        del hv

        h, hu, hv = np.zeros_like(ht), np.zeros_like(hut), np.zeros_like(hvt)

        # Force boundary conditions
        for hh in [ht, hut, hvt]:
            hh[0, :] = hh[1, :]
            hh[-1, :] = hh[-2, :]
            hh[:, 0] = hh[:, 1]
            hh[:, -1] = hh[:, -2]


        compute_step(ht, hut, hvt, h, hu, hv, zdx, zdy, dT, DeltaX, idx_i, idx_j)

        # Impose tolerances
        h[h <= 0] = 0.00001
        hu[h <= 0.0005] = 0
        hv[h <= 0.0005] = 0

        #Update time
        Tn += dT
        n_steps += 1
    ## End while loop -------------------------------------------------------------------------------------------

    ## Stop loop timer
    loop_time_elapsed = time.time() - start_loop_time

    ## Save the results
    # Save solution to disk
    h_transposed = np.array(h).T
    h_transposed.tofile(solution_file)

    # Stop timer
    time_elapsed = time.time() - start_time

    # Communicate time-to-compute
    calc_dT = 15
    check_T_Tend = 2
    n_calc_h_per_cell = 11
    n_calc_hu_per_cell = 30
    n_calc_hv_per_cell = 30
    calc_update_T = 1
    calc_report = 2

    ops = n_steps * ((n_calc_h_per_cell + n_calc_hu_per_cell + n_calc_hv_per_cell)*(Nx**2) + calc_report + calc_dT + check_T_Tend + calc_update_T)
    flops = ops / time_elapsed

    old_ops = n_steps * (15 + 2 + 11 + 30 + 30 + 1) * Nx ** 2
    old_flops = old_ops / time_elapsed

    global_time_str = f'Global time to compute solution: {time_elapsed} seconds'
    initialization_time_str = f'Time to initialize solution: {initialization_time} seconds'
    loop_time_str = f'Time to compute solution: {loop_time_elapsed} seconds'
    old_gflops_str = f'Average performance (old): {old_flops / 1.0e9} gflops'
    new_gflops_str = f'Average performance (new): {flops / 1.0e9} gflops'

    print(global_time_str)
    print(initialization_time_str)
    print(loop_time_str)

    print(old_gflops_str)
    print(new_gflops_str)

    # Save solution to txt file
    with open(solution_txt, 'w') as f:
        f.write(solution_file.stem.__str__() + '\n')
        f.write(global_time_str + '\n')
        f.write(initialization_time_str + '\n')
        f.write(loop_time_str + '\n')
        f.write(old_gflops_str + '\n')
        f.write(new_gflops_str + '\n\n')

        f.write(f'Number of time steps: {n_steps}\n')
        f.write(f'Number of operations per time step (new): {ops}\n')
        f.write(f'Number of operations per time step (old): {old_ops}\n')
        f.write(f'MapSize: {MapSize} km\n')
        f.write(f'Nx: {Nx}\n')
        f.write(f'DeltaX: {DeltaX} km\n')
        f.write(f'Tend: {Tend} hr\n')

        f.write(f'Data file: {data_file.stem}\n')

    ## Plot final conditions
    Topology = np.fromfile(open(path_to_data / f"Fig_nx{Nx}_{MapSize}km_Typography.bin", 'rb'), dtype=np.double).reshape((Nx, Nx))
    vis.plot_tsunami(h_transposed, MapSize, Nx, Topology, title='Result of the simulation', 
                        save=True, save_path=path_to_human_results, save_name='Result', tag=run_tag,
                        highlight_waves=False, save_spec=f"np{number_of_processes}_nn{number_of_nodes}_ncpt{number_of_cpus_per_task}")
    vis.plot_tsunami(h_transposed, MapSize, Nx, Topology, title='Result of the simulation', 
                        save=True, save_path=path_to_human_results, save_name='Result', tag=run_tag,
                        highlight_waves=True, save_spec=f"np{number_of_processes}_nn{number_of_nodes}_ncpt{number_of_cpus_per_task}")

    ## Plot initial conditions
    H_initial = np.fromfile(open(data_file.with_name(data_file.name + "_h.bin"), "rb"), dtype=np.double).reshape(Nx,Nx)
    vis.plot_tsunami(H_initial, MapSize, Nx, Topology, title='Initial conditions', 
                        save=True, save_path=path_to_human_results, save_name='Init', tag=run_tag,
                        highlight_waves=False, save_spec=f"np{number_of_processes}_nn{number_of_nodes}_ncpt{number_of_cpus_per_task}")
    vis.plot_tsunami(H_initial, MapSize, Nx, Topology, title='Initial conditions', 
                        save=True, save_path=path_to_human_results, save_name='Init', tag=run_tag,
                        highlight_waves=True, save_spec=f"np{number_of_processes}_nn{number_of_nodes}_ncpt{number_of_cpus_per_task}")

    ## Output the data to csv file
    data_row = [(solution_file.stem, run_tag, number_of_processes, number_of_nodes, number_of_cpus_per_task, Nx, MapSize, 
                DeltaX, Tend, n_steps, old_ops, ops, time_elapsed, initialization_time, 
                loop_time_elapsed, 0)]

    with open(table_results_csv, 'a') as f:

        writer = csv.writer(f)
        # If the csv file is empty, write the header
        if f.tell() == 0:
            writer.writerow(('name', 'tag', 'N_processes', 'N_nodes', 'N_cpus_per_task', 'Nx', 'map_size', 
                                'dx', 'Tend', 'N_time_steps', 'old_N_operations', 
                                'N_operations', 'total_time', 'initialization_time', 
                                'loop_time', 'reconstruction_time'))
            
        writer.writerows(data_row)

if __name__ == "__main__":
    ## Import the config file and set the parameters
    # Read yaml file
    current_file_position = Path(__file__).resolve().parent
    config_file = current_file_position / 'config.yaml'
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Get parameters
    G = config['G']
    MapSize = config['MapSize']
    Nx = config['Nx']
    Tend = config['Tend']
    number_of_processes = config['number_of_processes']
    number_of_nodes = config['number_of_nodes']
    number_of_cpus_per_task = config['number_of_cpus_per_task']
    run_tag = config['run_tag']
    DeltaX = MapSize / Nx # Grid spacing

    # Create paths
    path_to_data = current_file_position / config['path_to_data']
    path_to_results = current_file_position / config['path_to_results']
    path_to_human_results = path_to_results / "human"
    path_to_data.mkdir(parents=True, exist_ok=True)
    path_to_results.mkdir(parents=True, exist_ok=True)
    path_to_human_results.mkdir(parents=True, exist_ok=True)

    # Create filenames
    data_file = path_to_data / f"Data_nx{Nx}_{MapSize}km_T{Tend}"
    solution_file = path_to_results /f"Solution_nx{Nx}_{MapSize}km_T{str(Tend).split('.')[1]}_np{number_of_processes}_nn{number_of_nodes}_ncpt{number_of_cpus_per_task}_h.bin"
    solution_txt = path_to_human_results /f"Solution_nx{Nx}_{MapSize}km_T{str(Tend).split('.')[1]}_np{number_of_processes}_nn{number_of_nodes}_ncpt{number_of_cpus_per_task}_h.txt"
    table_results_csv = path_to_human_results /f"table_results.csv"
    if run_tag != '':
        solution_file = path_to_results /f"Solution_nx{Nx}_{MapSize}km_T{str(Tend).split('.')[1]}_np{number_of_processes}_nn{number_of_nodes}_ncpt{number_of_cpus_per_task}_{run_tag}_h.bin"
        solution_txt = path_to_human_results /f"Solution_nx{Nx}_{MapSize}km_T{str(Tend).split('.')[1]}_np{number_of_processes}_nn{number_of_nodes}_ncpt{number_of_cpus_per_task}_{run_tag}_h.txt"
    
    simulation()
   