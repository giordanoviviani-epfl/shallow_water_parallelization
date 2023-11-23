## Import libraries
import numpy as np
from pathlib import Path
import time
import yaml
import sys

## Import MPI
from mpi4py import MPI

## Set DEBUG flag
DEBUG = False

## Start timer
start_time = time.time()

## Start MPI thread
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f'rank: {rank} - size: {size}') if DEBUG else None
print(f'Number of processes: {size}') if rank == 0 else None

## Create cartesian communicator
dimensions_cartesian2d = MPI.Compute_dims(size, 2) # Get the dimensions of the cartesian communicator

cartesian2d = comm.Create_cart(dims=dimensions_cartesian2d, 
                               periods=[False, False], 
                               reorder=False)

cart_coords = cartesian2d.Get_coords(rank)
cart_rank = cartesian2d.Get_rank()
cart_size = cartesian2d.Get_size()
assert size == cart_size, "The number of MPI processes must remain constant"
assert cart_rank == rank, "The rank of the MPI process must remain the same"

## Import the config file and set the parameters
current_file_position = Path(__file__).resolve().parent
config_file = current_file_position / 'config.yaml'
with open(config_file) as f:
    config = yaml.safe_load(f)

G = config['G']
MapSize = config['MapSize']
Nx = config['Nx']
Tend = config['Tend']
number_of_processes = config['number_of_processes']

assert number_of_processes == size, "The number of MPI processes spcified in config.yaml must be the same as the number of MPI processes"
DeltaX = MapSize / Nx # Grid spacing

path_to_data = current_file_position / config['path_to_data']
path_to_results = current_file_position/ config['path_to_results']
path_to_human_results = path_to_results / 'human'
path_to_data.mkdir(parents=True, exist_ok=True)
path_to_results.mkdir(parents=True, exist_ok=True)
path_to_human_results.mkdir(parents=True, exist_ok=True)

filename =f"Data_nx{Nx}_{MapSize}km_T{Tend}"
solution_file = path_to_results /f'Solution_nx{Nx}_{MapSize}km_T{Tend}_nprocess{number_of_processes}_h.bin'
solution_txt = path_to_human_results /f'Solution_nx{Nx}_{MapSize}km_T{Tend}_nprocess{number_of_processes}_h.txt'

## Compute local size of the grid
global_size = np.array([Nx, Nx])
local_offset = np.zeros(2, dtype=np.int32)
ghosts = np.zeros(2, dtype=np.int32)
matrix_slices = [slice(None), slice(None)]

local_size = global_size // dimensions_cartesian2d + \
            (cart_coords < global_size % dimensions_cartesian2d) # add 1 to the first n processes where n is the remainder of the division
for i in range(2):
    if dimensions_cartesian2d[i] > 1:
        ghosts[i] = 1 if (cart_coords[i] == 0 or cart_coords[i] == dimensions_cartesian2d[i] - 1) else 2
        local_size[i] += ghosts[i]  

    # Compute the local offset
    local_offset[i] = (global_size[i] // dimensions_cartesian2d[i]) * cart_coords[i] #TODO CHECK THIS
    if cart_coords[i] < global_size[i] % dimensions_cartesian2d[i]:
        local_offset[i] += cart_coords[i]
    else:
        local_offset[i] += global_size[i] % dimensions_cartesian2d[i]

    local_offset_ghost_corr = local_offset[i] 
    if (cart_coords[i] != 0):
        local_offset_ghost_corr -= 1
    
    matrix_slices[i] = slice(local_offset_ghost_corr, local_offset_ghost_corr + local_size[i])


if DEBUG:
    print(f"""rank: {rank} - local_size: {local_size} - local_offset: [{local_offset[0]:04d} {local_offset[1]:04d}] 
          - ghosts: {ghosts} - cart_coords: {cart_coords} - matrix_slices: {matrix_slices}""")

## Determine Neighbors
north, south = cartesian2d.Shift(0, 1) # Get the neighbors in the first dimension
left, right = cartesian2d.Shift(1, 1) # Get the neighbors in the second dimension

if DEBUG:
    print(f'rank: {rank} - north: {north} - south: {south} - left: {left} - right: {right}')

## Load the data
# Initial conditions
h = (np.fromfile(open(path_to_data/ (filename+"_h.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T)[matrix_slices[0], matrix_slices[1]]
hu = (np.fromfile(open(path_to_data/ (filename+"_hu.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T)[matrix_slices[0], matrix_slices[1]]
hv = (np.fromfile(open(path_to_data/ (filename+"_hv.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T)[matrix_slices[0], matrix_slices[1]]

# Topography
zdx = (np.fromfile(open(path_to_data/ (filename+"_Zdx.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T)[matrix_slices[0], matrix_slices[1]]
zdy = (np.fromfile(open(path_to_data/ (filename+"_Zdy.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T)[matrix_slices[0], matrix_slices[1]]

## Initialize empty arrays for the values on the grid (t = temporary)
ht = np.zeros_like(h)
hut = np.zeros_like(h)
hvt = np.zeros_like(h) 

if DEBUG:
    print(f'rank: {rank} - ht.shape: {ht.shape} - hut.shape: {hut.shape} - hvt.shape: {hvt.shape}')

# Slices of the arrays of the matrices need for the calculations
# First element is the centrated matrix (i), whereas the second is the positively shifted matrix (i+1), and the third is the negatively shifted matrix (i-1) 
idx_i = [slice(1, local_size[0]-1), slice(2, local_size[0]), slice(0, local_size[0]-2)]
idx_j = [slice(1, local_size[1]-1), slice(2, local_size[1]), slice(0, local_size[1]-2)]


## Compute the solution
Tn = 0 # Current time
n_steps = 0 # Time step\

#sys.exit('STOP')

## While loop -----------------------------------------------------------------------------------------------
print('START COMPUTATION') if rank == 0 else None
while Tn < Tend:
    # Calculate the time step
    under_root = ((np.maximum(np.abs(hu/h + np.sqrt(G*h)), np.abs(hu/h - np.sqrt(G*h))))**2 +
                    (np.maximum(np.abs(hv/h + np.sqrt(G*h)), np.abs(hv/h - np.sqrt(G*h))))**2)
    max_under_root = np.max(under_root)
    max_under_root_array = cartesian2d.gather(max_under_root, root=0)
    # ## All gather version
    # mura = np.empty(cart_size)
    # cartesian2d.Allgather(max_under_root, mura)
    dT = None
    if rank == 0:
        nu = np.sqrt(np.max(max_under_root_array))
        dT = DeltaX / (np.sqrt(2) * nu)
        if Tn + dT > Tend:
            dT = Tend - Tn

    dT = cartesian2d.bcast(dT, root=0)
    # ## All reduce version
    # dTT = DeltaX / (np.sqrt(2) * cartesian2d.allreduce(max_under_root, op=MPI.MAX))
    # if DEBUG:
    #     print(f'bcast: {rank} - {max_under_root} - dT: {dT} - dTT: {dTT}')

    if DEBUG:
        print(f'bcast: {rank} - {max_under_root_array} - dT: {dT}')
    # Report Status
    print(f'{n_steps:04d} - Computing T: {Tn + dT} ({100 * (Tn + dT) / Tend}%) - dT: {dT} hr - exc. time {time.time() - start_time}') if (rank==0) and (n_steps%50==0) else None

    # Copy solution to temporary variables
    ht = h.copy()
    hut = hu.copy()
    hvt = hv.copy()

    # Force boundary conditions
    for hh in [ht, hut, hvt]:
        if cart_coords[0] == 0:
            hh[:, 0] = hh[:, 1]
        if cart_coords[0] == dimensions_cartesian2d[0] - 1:
            hh[:, -1] = hh[:, -2]
        if cart_coords[1] == 0:
            hh[0, :] = hh[1, :]
        if cart_coords[1] == dimensions_cartesian2d[1] - 1:
            hh[-1, :] = hh[-2, :]

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

    ## Communicate ghost cells
    # sendrecv(sendobj, dest, sendtag=0, recvbuf=None, source=ANY_SOURCE, recvtag=ANY_TAG, status=None)

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

    if cart_coords == [0, 0]:
        print(f'rank: {rank} - h: {h[-1, :]}', flush=True)
        print(f'rank: {rank} - hcol: {h[:, -1]}', flush=True)
    if cart_coords == [1, 0]:
        print(f'rank: {rank} - h: {h[1, :]}', flush=True)
    if cart_coords == [0, 1]:
        print(f'rank: {rank} - hcol: {h[:, 1]}', flush=True)



    sys.exit('STOP')
    """
    h[-1, :] = cartesian2d.sendrecv(h[1, :], dest=north, sendtag=0, recvbuf=None, source=south, recvtag=0)
    hu[-1, :] = cartesian2d.sendrecv(hu[1, :], dest=north, sendtag=1, recvbuf=None, source=south, recvtag=1)
    hv[-1, :] = cartesian2d.sendrecv(hv[1, :], dest=north, sendtag=2, recvbuf=None, source=south, recvtag=2)

    # Going down (so receiving from top)
    h[0, :] = cartesian2d.sendrecv(h[-2, :], dest=south, sendtag=0, recvbuf=None, source=north, recvtag=0)
    hu[0, :] = cartesian2d.sendrecv(hu[-2, :], dest=south, sendtag=1, recvbuf=None, source=north, recvtag=1)
    hv[0, :] = cartesian2d.sendrecv(hv[-2, :], dest=south, sendtag=2, recvbuf=None, source=north, recvtag=2)

    # Going left (so receiving from right)
    h[:, -1] = cartesian2d.sendrecv(h[:, 1], dest=left, sendtag=0, recvbuf=None, source=right, recvtag=0)
    hu[:, -1] = cartesian2d.sendrecv(hu[:, 1], dest=left, sendtag=1, recvbuf=None, source=right, recvtag=1)
    hv[:, -1] = cartesian2d.sendrecv(hv[:, 1], dest=left, sendtag=2, recvbuf=None, source=right, recvtag=2)

    # Going right (so receiving from left)
    h[:, 0] = cartesian2d.sendrecv(h[:, -2], dest=right, sendtag=0, recvbuf=None, source=left, recvtag=0)
    hu[:, 0] = cartesian2d.sendrecv(hu[:, -2], dest=right, sendtag=1, recvbuf=None, source=left, recvtag=1)
    hv[:, 0] = cartesian2d.sendrecv(hv[:, -2], dest=right, sendtag=2, recvbuf=None, source=left, recvtag=2)
    """

    # Impose tolerances
    h[h <= 0] = 0.00001
    hu[h <= 0.0005] = 0
    hv[h <= 0.0005] = 0


    #Update time
    Tn += dT
    n_steps += 1

print('END OF COMPUTATION') if rank == 0 else None
## End while loop -------------------------------------------------------------------------------------------a
# Initialize the result matrix H
H = np.ascontiguousarray(np.empty((Nx, Nx), dtype=np.double)) if rank == 0 else None

# Gather the results back to the root process
cartesian2d.Gatherv(np.ascontiguousarray(h[idx_i[0], idx_j[0]]), H, root=0)

if DEBUG:
    print(f'H.shape: {H.shape} - H.dtype: {H.dtype}') if rank == 0 else None

if rank == 0:
    ## Save the results
    # Save solution to disk
    H_transposed = np.array(H).T
    H_transposed.tofile(solution_file)

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

    time_str = f'Time to compute solution: {time_elapsed} seconds'
    old_gflops_str = f'Average performance (old): {old_flops / 1.0e9} gflops'
    new_gflops_str = f'Average performance (new): {flops / 1.0e9} gflops'

    print(time_str)
    print(old_gflops_str)
    print(new_gflops_str)

    # Save solution to txt file
    with open(solution_txt, 'w') as f:
        f.write(solution_file.__str__() + '\n')
        f.write(time_str + '\n')
        f.write(old_gflops_str + '\n')
        f.write(new_gflops_str + '\n\n')

        f.write(f'Number of time steps: {n_steps}\n')
        f.write(f'Number of operations per time step (new): {ops}\n')
        f.write(f'Number of operations per time step (old): {old_ops}\n')
        f.write(f'MapSize: {MapSize} km\n')
        f.write(f'Nx: {Nx}\n')
        f.write(f'DeltaX: {DeltaX} km\n')
        f.write(f'Tend: {Tend} hr\n')

        f.write(f'Data file: {filename}\n')

