## Import libraries
import numpy as np
from pathlib import Path
import time
import yaml

## Start timer
start_time = time.time()

## Import the config file and set the parameters
current_file_position = Path(__file__).resolve().parent
config_file = current_file_position / 'config.yaml'
with open(config_file) as f:
    config = yaml.safe_load(f)

G = config['G']
MapSize = config['MapSize']
Nx = config['Nx']
Tend = config['Tend']

DeltaX = MapSize / Nx # Grid spacing

path_to_data = current_file_position / config['path_to_data']
path_to_results = current_file_position/ config['path_to_results']
path_to_data.mkdir(parents=True, exist_ok=True)
path_to_results.mkdir(parents=True, exist_ok=True)

filename =f"Data_nx{Nx}_{MapSize}km_T{Tend}"
solution_file = path_to_results /f'Solution_nx{Nx}_{MapSize}km_T{Tend}_h.bin'
solution_txt = path_to_results /f'Solution_nx{Nx}_{MapSize}km_T{Tend}_h.txt'

## Load the data
# Initial conditions
h = np.fromfile(open(path_to_data/ (filename+"_h.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T
hu = np.fromfile(open(path_to_data/ (filename+"_hu.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T
hv = np.fromfile(open(path_to_data/ (filename+"_hv.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T

# Topography
zdx = np.fromfile(open(path_to_data/ (filename+"_Zdx.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T
zdy = np.fromfile(open(path_to_data/ (filename+"_Zdy.bin"), "rb"), dtype=np.double).reshape(Nx,Nx).T

## Initialize empty arrays for the values on the grid (t = temporary)
ht = np.zeros((Nx,Nx))
hut = np.zeros((Nx,Nx))
hvt = np.zeros((Nx,Nx)) 

## Compute the solution
Tn = 0 # Current time
n_steps = 0 # Time step\
fast = False
while Tn < Tend:

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

    # Report Status
    if n_steps % 10 == 0:
        print(f'{n_steps:04d} - Computing T: {Tn + dT} ({100 * (Tn + dT) / Tend}%) - dT: {dT} hr ')

    # Copy solution to temporary variables
    ht = h.copy()
    hut = hu.copy()
    hvt = hv.copy()

    # Force boundary conditions
    for hh in [ht, hut, hvt]:
        hh[0, :] = hh[1, :]
        hh[-1, :] = hh[-2, :]
        hh[:, 0] = hh[:, 1]
        hh[:, -1] = hh[:, -2]

    # Compute the constant factors
    mult_factor = 0.5 * dT / DeltaX 


    # Slices of the arrays of the matrices need for the calculations
    # First element is the centrated matrix (i), whereas the second is the positively shifted matrix (i+1), and the third is the negatively shifted matrix (i-1) 
    i = [slice(1, Nx-1), slice(2, Nx), slice(0, Nx-2)]
    j = [slice(1, Nx-1), slice(2, Nx), slice(0, Nx-2)]

    # Define useful functions
    first_term = lambda matrix: 0.25 * (matrix[i[0], j[-1]] + matrix[i[0], j[1]] + matrix[i[-1], j[0]] + matrix[i[1], j[0]])
    second_term = lambda matrix: dT * G * h[i[0], j[0]] * matrix[i[0], j[0]] # for hu and hv matrices (variables are zdx and zdy) 
    ratios_difference = lambda i1, j1, i2, j2: mult_factor * ((hut[i1, j1] * hvt[i1, j1])/ht[i1, j1] - (hut[i2, j2] * hvt[i2, j2])/ht[i2, j2])
    sum_with_g = lambda mat, i1, j1: ((mat[i1, j1]**2) / ht[i1, j1]) + 0.5*G*(ht[i1, j1]**2)  
    g_difference = lambda mat, i1, j1, i2, j2: mult_factor * (sum_with_g(mat, i1, j1) - sum_with_g(mat, i2, j2))
    #g_difference2 = lambda mat, i1, j1, i2, j2: mult_factor * (((mat[i1, j1]**2) / ht[i1, j1]) + 0.5*G*(ht[i1, j1]**2) - ((mat[i2, j2]**2) / ht[i2, j2]) - 0.5*G*(ht[i2, j2]**2))

    # Compute the fluxes
    h[i[0], j[0]] = first_term(ht) + mult_factor * (hut[i[0], j[-1]] - hut[i[0], j[1]] + hvt[i[-1], j[0]] - hvt[i[1], j[0]])

    hu[i[0], j[0]] =  first_term(hut) - second_term(zdx) + ratios_difference(i[-1], j[0], i[1], j[0]) + g_difference(hut, i[0], j[-1], i[0], j[1])

    hv[i[0], j[0]] = first_term(hvt) - second_term(zdy) + ratios_difference(i[0], j[-1], i[0], j[1]) + g_difference(hvt, i[-1], j[0], i[1], j[0])

    # Impose tolerances
    h[h <= 0] = 0.00001
    hu[h <= 0.0005] = 0
    hv[h <= 0.0005] = 0

    #Update time
    Tn += dT
    n_steps += 1

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

