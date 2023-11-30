## Import libraries
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import yaml
import sys
import visualize as vis
import csv

## Import MPI
from mpi4py import MPI

## Set DEBUG and PLOTS flag
DEBUG = False
PLOT_CARTESIAN_GRID = False
PLOT_TSUNAMI = False

## Functions ------------------------------------------------------------------------------------------------
def calculate_time_step(h, hu, hv, mpi_rank, communicator, Tn, optimize):
    global G, DEBUG
    if optimize:
        GH_factor = np.sqrt(G*h)
        huh = hu/h
        hvh = hv/h
        under_root = 2*np.max((np.maximum(np.abs(huh+GH_factor), np.abs(huh-GH_factor)))**2+
                      (np.maximum(np.abs(hvh+GH_factor), np.abs(hvh-GH_factor)))**2)
        max_under_root = communicator.allreduce(under_root, op=MPI.MAX)
        dT = DeltaX/(np.sqrt(max_under_root))
    else:
        # Calculate the time step
        under_root = ((np.maximum(np.abs(hu/h + np.sqrt(G*h)), np.abs(hu/h - np.sqrt(G*h))))**2 +
                        (np.maximum(np.abs(hv/h + np.sqrt(G*h)), np.abs(hv/h - np.sqrt(G*h))))**2)
        max_under_root = np.max(under_root)
        max_under_root_array = communicator.gather(max_under_root, root=0)

        # ## All gather version
        # mura = np.empty(cart_size)
        # cartesian2d.Allgather(max_under_root, mura)

        dT = None
        if mpi_rank == 0:
            nu = np.sqrt(np.max(max_under_root_array))
            dT = DeltaX / (np.sqrt(2) * nu)
            if Tn + dT > Tend:
                dT = Tend - Tn

        ## Broadcast the time step to all processes
        dT = communicator.bcast(dT, root=0)

        if DEBUG:
            print(f'bcast: {mpi_rank} - {max_under_root_array} - dT: {dT}')
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
    # g_difference2 = lambda mat, i1, j1, i2, j2: mult_factor * (((mat[i1, j1]**2) / ht[i1, j1]) + 0.5*G*(ht[i1, j1]**2) - ((mat[i2, j2]**2) / ht[i2, j2]) - 0.5*G*(ht[i2, j2]**2))

    # Compute the fluxes
    h[idx_i[0], idx_j[0]] = first_term(ht) + mult_factor * (hut[idx_i[0], idx_j[-1]] - hut[idx_i[0], idx_j[1]] + hvt[idx_i[-1], idx_j[0]] - hvt[idx_i[1], idx_j[0]])

    hu[idx_i[0], idx_j[0]] =  first_term(hut) - second_term(zdx) + ratios_difference(idx_i[-1], idx_j[0], idx_i[1], idx_j[0]) + g_difference(hut, idx_i[0], idx_j[-1], idx_i[0], idx_j[1])

    hv[idx_i[0], idx_j[0]] = first_term(hvt) - second_term(zdy) + ratios_difference(idx_i[0], idx_j[-1], idx_i[0], idx_j[1]) + g_difference(hvt, idx_i[-1], idx_j[0], idx_i[1], idx_j[0])


## End Functions --------------------------------------------------------------------------------------------

def simulation():
    ## Start timer
    start_time = time.time()

    ## Start global MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f'Number of processes: {size}') if rank == 0 else None

    ## Create cartesian communicator
    dimensions_cartesian2d = MPI.Compute_dims(size, 2) # Get the dimensions of the cartesian communicator
    cartesian2d = comm.Create_cart(dims=dimensions_cartesian2d, 
                                periods=[False, False], 
                                reorder=False)

    cart_coords = cartesian2d.Get_coords(rank)
    cart_rank = cartesian2d.Get_rank()
    cart_size = cartesian2d.Get_size()
    print(f"Dimensions of the cartesian communicator: {dimensions_cartesian2d}") if rank == 0 else None

    ## Assert consistency
    assert size == cart_size, "The number of MPI processes must remain constant"
    assert cart_rank == rank, "The rank of the MPI process must remain the same"
    assert number_of_processes == size, "The number of MPI processes spcified in config.yaml must be the same as the number of MPI processes"


    ## Compute local size of the grid
    # Initialize some variables
    global_size = np.array([Nx, Nx])
    local_offset = np.zeros(2, dtype=np.int32)
    ghost_corrected_local_offset = np.zeros(2, dtype=np.int32)
    ghosts = np.zeros(2, dtype=np.int32)
    matrix_slices = [slice(None), slice(None)]

    # Compute the local size of the grid
    local_size = global_size // dimensions_cartesian2d + \
                (cart_coords < global_size % dimensions_cartesian2d) # add 1 to the first n processes where n is the remainder of the division
    local_size_no_ghost = local_size.copy()
    for i in range(2):
        if dimensions_cartesian2d[i] > 1:
            ghosts[i] = 1 if (cart_coords[i] == 0 or cart_coords[i] == dimensions_cartesian2d[i] - 1) else 2
            local_size[i] += ghosts[i]  

        # Compute the local offset
        local_offset[i] = (global_size[i] // dimensions_cartesian2d[i]) * cart_coords[i]
        if cart_coords[i] < global_size[i] % dimensions_cartesian2d[i]:
            local_offset[i] += cart_coords[i]
        else:
            local_offset[i] += global_size[i] % dimensions_cartesian2d[i]

        ghost_corrected_local_offset[i] = local_offset[i] 
        if (cart_coords[i] != 0):
            ghost_corrected_local_offset[i] -= 1
        
        matrix_slices[i] = slice(ghost_corrected_local_offset[i], ghost_corrected_local_offset[i] + local_size[i])

    ## Determine Neighbors
    north, south = cartesian2d.Shift(0, 1) # Get the neighbors in the first dimension
    left, right = cartesian2d.Shift(1, 1) # Get the neighbors in the second dimension

    if DEBUG:
        print(f"""rank: {rank} - local_size: {local_size} - local_offset: [{local_offset[0]:04d} {local_offset[1]:04d}] 
            - ghosts: {ghosts} - cart_coords: {cart_coords} - matrix_slices: {matrix_slices} 
            - north: {north} - south: {south} - left: {left} - right: {right}""")


    if PLOT_CARTESIAN_GRID:
        # Get lists of values from all processes
        matrix_slices_array = cartesian2d.gather(matrix_slices, root=0)
        cart_coords_array = cartesian2d.gather(cart_coords, root=0)
        ghost_corrected_local_offset_array = cartesian2d.gather(ghost_corrected_local_offset, root=0)
        local_size_array = cartesian2d.gather(local_size, root=0)
        n_ghosts_array = cartesian2d.gather(ghosts, root=0)

        if rank==0:
            vis.cartesian_grid(MapSize, Nx, dimensions_cartesian2d, local_size_array, matrix_slices_array,
                            n_ghosts_array, cart_coords_array, ghost_corrected_local_offset_array, 
                            cartesian_grid_plot_file)

    ## Load the data
    # Initial conditions
    h = (np.fromfile(open(data_file.with_name(data_file.name + "_h.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T)[matrix_slices[0], matrix_slices[1]]
    hu = (np.fromfile(open(data_file.with_name(data_file.name + "_hu.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T)[matrix_slices[0], matrix_slices[1]]
    hv = (np.fromfile(open(data_file.with_name(data_file.name + "_hv.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T)[matrix_slices[0], matrix_slices[1]]

    # Topography
    zdx = (np.fromfile(open(data_file.with_name(data_file.name + "_Zdx.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T)[matrix_slices[0], matrix_slices[1]]
    zdy = (np.fromfile(open(data_file.with_name(data_file.name + "_Zdy.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T)[matrix_slices[0], matrix_slices[1]]

    ## Initialize empty arrays for the values on the grid (t = temporary)
    ht = np.zeros_like(h)
    hut = np.zeros_like(h)
    hvt = np.zeros_like(h) 

    # Slices of the arrays of the matrices need for the calculations
    # First element is the centrated matrix (i), whereas the second is the positively shifted matrix (i+1), and the third is the negatively shifted matrix (i-1) 
    idx_i = [slice(1, local_size[0]-1), slice(2, local_size[0]), slice(0, local_size[0]-2)]
    idx_j = [slice(1, local_size[1]-1), slice(2, local_size[1]), slice(0, local_size[1]-2)]

    ## Compute the solution
    Tn = 0 # Current time
    n_steps = 0 # Time step\
    optimize=True

    ## Get initialization time
    initialization_time = time.time() - start_time

    ## Start loop timer
    start_loop_time = time.time()

    ## While loop -----------------------------------------------------------------------------------------------
    print('START COMPUTATION') if rank == 0 else None
    while Tn < Tend:

        dT = calculate_time_step(h, hu, hv, rank, cartesian2d, Tn, optimize)

        # ## All reduce version
        # dTT = DeltaX / (np.sqrt(2) * cartesian2d.allreduce(max_under_root, op=MPI.MAX))
        # if DEBUG:
        #     print(f'bcast: {rank} - {max_under_root} - dT: {dT} - dTT: {dTT}')

        # Report Status
        print(f'{n_steps:04d} - Computing T: {Tn + dT} ({100 * (Tn + dT) / Tend}%) - dT: {dT} hr - exc. time {time.time() - start_time}') if (rank==0) and (n_steps%50==0) else None

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
            if cart_coords[0] == 0:
                hh[0, :] = hh[1, :]
            if cart_coords[0] == dimensions_cartesian2d[0] - 1:
                hh[-1, :] = hh[-2, :]
            if cart_coords[1] == 0:                
                hh[:, 0] = hh[:, 1]
            if cart_coords[1] == dimensions_cartesian2d[1] - 1:
                hh[:, -1] = hh[:, -2]

        compute_step(ht, hut, hvt, h, hu, hv, zdx, zdy, dT, DeltaX, idx_i, idx_j)

        ## Communicate ghost cells
        # Going up (so receiving from bottom)
        for i, hh in enumerate([h, hu, hv]):
            hh_row = cartesian2d.sendrecv(hh[1, :], dest=north, sendtag=i, recvbuf=None, source=south, recvtag=i)
            if isinstance(hh_row, np.ndarray):
                hh[-1, :] = hh_row

        # Going down (so receiving from top)
        for i, hh in enumerate([h, hu, hv]):
            hh_row = cartesian2d.sendrecv(hh[-2, :], dest=south, sendtag=i, recvbuf=None, source=north, recvtag=i)
            if isinstance(hh_row, np.ndarray):
                hh[0, :] = hh_row
        
        # Going left (so receiving from right)
        for i, hh in enumerate([h, hu, hv]):
            hh_col = cartesian2d.sendrecv(hh[:, 1], dest=left, sendtag=i, recvbuf=None, source=right, recvtag=i)
            if isinstance(hh_col, np.ndarray):
                hh[:, -1] = hh_col
        
        # Going right (so receiving from left)
        for i, hh in enumerate([h, hu, hv]):
            hh_col = cartesian2d.sendrecv(hh[:, -2], dest=right, sendtag=i, recvbuf=None, source=left, recvtag=i)
            if isinstance(hh_col, np.ndarray):
                hh[:, 0] = hh_col

        # Impose tolerances
        h[h <= 0] = 0.00001
        hu[h <= 0.0005] = 0
        hv[h <= 0.0005] = 0


        #Update time
        Tn += dT
        n_steps += 1

    print('END OF COMPUTATION') if rank == 0 else None
    ## End while loop -------------------------------------------------------------------------------------------

    ## Stop loop timer
    loop_time_elapsed = time.time() - start_loop_time

    ## Start reconstruction timer
    start_reconstruction_time = time.time()

    ## Saving the results
    # Gather the results back to the root process
    shift_i_due_to_ghosts = 0 if cart_coords[0] == 0 else 1
    shift_j_due_to_ghosts = 0 if cart_coords[1] == 0 else 1
    idx_i_matrix_to_return = slice(0+shift_i_due_to_ghosts, local_size_no_ghost[0]+shift_i_due_to_ghosts) 
    idx_j_matrix_to_return = slice(0+shift_j_due_to_ghosts, local_size_no_ghost[1]+shift_j_due_to_ghosts)

    matrix_to_return = np.ascontiguousarray(h[idx_i_matrix_to_return, idx_j_matrix_to_return])
    H = cartesian2d.gather(matrix_to_return, root=0)

    if rank == 0:
        
        ## Structure the results to recompose the initial grid
        H_structured = []
        for dim_y in range(dimensions_cartesian2d[0]):
            H_structured.append(H[dim_y*dimensions_cartesian2d[1]:(dim_y+1)*dimensions_cartesian2d[1]])

        ## Save the results
        # Save solution to disk
        H_transposed = np.block(H_structured).T
        H_transposed.tofile(solution_file)
        
        if DEBUG:
            print(f'H.shape: {H_transposed.shape} - H.dtype: {H_transposed.dtype}') 

        ## Stop timers
        # Stop reconstruction timer
        reconstruction_time_elapsed = time.time() - start_reconstruction_time
        # Stop global timer
        time_elapsed = time.time() - start_time
        # Communicate timers

        global_time_str = f'Global time to compute solution: {time_elapsed} seconds'
        initialization_time_str = f'Time to initialize solution: {initialization_time} seconds'
        loop_time_str = f'Time to compute solution: {loop_time_elapsed} seconds'
        reconstruction_time_str = f'Time to reconstruct solution: {reconstruction_time_elapsed} seconds'
        print(global_time_str)
        print(initialization_time_str)
        print(loop_time_str)
        print(reconstruction_time_str)

        ## Calculate gflops
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

        old_gflops_str = f'Average performance (old): {old_flops / 1.0e9} gflops'
        new_gflops_str = f'Average performance (new): {flops / 1.0e9} gflops'

        print(old_gflops_str)
        print(new_gflops_str)

        ## Save solution to txt file
        with open(solution_txt, 'w') as f:
            f.write(solution_file.stem.__str__() + '\n')
            f.write(global_time_str + '\n')
            f.write(initialization_time_str + '\n')
            f.write(loop_time_str + '\n')
            f.write(reconstruction_time_str + '\n')
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
        if PLOT_TSUNAMI:
            Topology = np.fromfile(open(path_to_data / f"Fig_nx{Nx}_{MapSize}km_Typography.bin", 'rb'), dtype=np.double).reshape((Nx, Nx))
            vis.plot_tsunami(H_transposed, MapSize, Nx, Topology, title='Result of the simulation', 
                            save=True, save_path=path_to_human_results, save_name='Result', tag=run_tag,
                            highlight_waves=False, save_spec=f"np{number_of_processes}_nn{number_of_nodes}_ncpt{number_of_cpus_per_task}")
            vis.plot_tsunami(H_transposed, MapSize, Nx, Topology, title='Result of the simulation', 
                            save=True, save_path=path_to_human_results, save_name='Result', tag=run_tag,
                            highlight_waves=True, save_spec=f"np{number_of_processes}_nn{number_of_nodes}_ncpt{number_of_cpus_per_task}")

            ## Plot initial conditions
            H_initial = np.fromfile(open(data_file.with_name(data_file.name+ "_h.bin"), "rb"), dtype=np.double).reshape(Nx,Nx)
            vis.plot_tsunami(H_initial, MapSize, Nx, Topology, title='Initial conditions', 
                            save=True, save_path=path_to_human_results, save_name='Init', tag=run_tag,
                            highlight_waves=False, save_spec=f"np{number_of_processes}_nn{number_of_nodes}_ncpt{number_of_cpus_per_task}")
            vis.plot_tsunami(H_initial, MapSize, Nx, Topology, title='Initial conditions', 
                            save=True, save_path=path_to_human_results, save_name='Init', tag=run_tag,
                            highlight_waves=True, save_spec=f"np{number_of_processes}_nn{number_of_nodes}_ncpt{number_of_cpus_per_task}")
        
        ## Output the data to csv file
        data_row = [(solution_file.stem, run_tag, number_of_processes, number_of_nodes, number_of_cpus_per_task, 
                    Nx, MapSize, DeltaX, Tend, n_steps, old_ops, ops, time_elapsed, initialization_time, 
                loop_time_elapsed, reconstruction_time_elapsed)]

        with open(table_results_csv, 'a') as f:

            writer = csv.writer(f)
            # If the csv file is empty, write the header
            if f.tell() == 0:
                writer.writerow(('name', 'tag', 'N_processes', 'N_nodes', 'N_cpus_per_task', 'Nx', 'map_size', 
                                'dx', 'Tend', 'N_time_steps', 'old_N_operations', 
                                'N_operations', 'total_time', 'initialization_time', 
                                'loop_time', 'reconstruction_time'))
                
            writer.writerows(data_row)

if __name__=="__main__":

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
    path_to_results = current_file_position/ config['path_to_results']
    path_to_human_results = path_to_results / 'human'
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
    cartesian_grid_plot_file = path_to_human_results /f"CartesianGrid_nx{Nx}_{MapSize}km_T{str(Tend).split('.')[1]}_np{number_of_processes}.pdf"

    simulation()