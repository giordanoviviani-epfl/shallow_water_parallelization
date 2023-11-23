import numpy as np
import yaml
from pathlib import Path
from matplotlib import colors
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt

def plot_tsunami(water_level, map_size, nx, topography, title='', highlight_waves=False, cstride=20, rstride=20):

    fontsize = 12
    mag_factor_z = 10
    light_source = LightSource(azdeg=315, altdeg=80)
    subplot_kw = {
        'projection': '3d',
        'computed_zorder': False,
    }

    land_idx= np.where(water_level <= 0.0005)
    undersea_idx = np.where(water_level > 0.0005)

    fig, ax = plt.subplots(subplot_kw=subplot_kw, figsize=(6, 6))

    # Meshgrid for the plot
    x, y = np.meshgrid(np.linspace(0, map_size, nx), np.linspace(0, map_size, nx))

    # Compute the magnified topography
    z_topography = mag_factor_z * topography

    # Colormap topography
    colors_undersea = cm.gist_earth(np.linspace(0, 0.20, 256))
    colors_land = cm.terrain(np.linspace(0.25, 0.9, 256))
    all_colors = np.vstack((colors_undersea, colors_land))
    terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)
    divnorm = colors.TwoSlopeNorm(vmin=np.min(z_topography), vcenter=0, vmax=np.max(z_topography)) # make the norm with ofsset center si the land has more dynamic range

    # Plot topography
    rgb_topography = light_source.shade(z_topography, cmap=terrain_map, norm=divnorm, vert_exag=10, blend_mode='overlay') # Possible colormaps: gist_earth, terrain, ocean, binary
    z_land = z_topography.copy()
    z_undersea = z_topography.copy()
    z_land[undersea_idx] = np.nan
    z_undersea[land_idx] = np.nan
    ax.plot_surface(x, y, z_undersea, cstride=cstride, rstride=rstride, facecolors=rgb_topography,
                    linewidth=0, antialiased=False, shade=False, alpha=1, zorder=0)
    ax.plot_surface(x, y, z_land, cstride=cstride, rstride=rstride, facecolors=rgb_topography,
                    linewidth=0, antialiased=False, shade=False, alpha=1, zorder=2)


    # Compute surface height
    water_surface_height = topography + water_level

    water_surface_height[land_idx] = np.nan
    z_water_surface_height = mag_factor_z * water_surface_height

    # Colormap water surface

    if highlight_waves:
        """
        # Previous version
        topo_lifted = np.zeros_like(water_level)
        idx = np.where(topography > 0)
        topo_lifted[idx] = topography[idx]
        color_data = topography - 40 * np.abs(water_surface_height - topo_lifted) - 1.5
        seanorm = colors.TwoSlopeNorm(vmin=np.nanmin(color_data), vcenter=np.median(color_data), vmax=np.nanmax(color_data))
        rgb_ocean = light_source.shade(color_data, cmap=cm.RdBu, norm=seanorm, vert_exag=10, blend_mode='overlay')
        """
        topo_lifted = np.zeros_like(water_level)
        idx = np.where(topography > 0)
        topo_lifted[idx] = topography[idx]
        color_data = -1000*(water_surface_height - topo_lifted) 
        seanorm = colors.TwoSlopeNorm(vmin=np.nanmin(color_data), vcenter=0, vmax=np.nanmax(color_data))
        rgb_ocean = light_source.shade(color_data, cmap=cm.coolwarm, norm=seanorm, vert_exag=10, blend_mode='overlay')
        
    else:
        color_sea = cm.ocean(np.linspace(0.3, 0.85, 256))
        ocean_map = colors.LinearSegmentedColormap.from_list('ocean_map', color_sea)
        seanorm = colors.TwoSlopeNorm(vmin=np.nanmin(z_water_surface_height), vcenter=0, vmax=np.nanmax(z_water_surface_height))
        rgb_ocean = light_source.shade(z_water_surface_height, cmap=ocean_map, norm=seanorm, vert_exag=10, blend_mode='overlay')

    ax.plot_surface(x, y, z_water_surface_height, cstride=cstride, rstride=rstride, facecolors=rgb_ocean,
                    linewidth=0, antialiased=False, shade=False, alpha=1, zorder=1)

    # Set axis limits
    zlim = ax.get_zlim()
    if zlim[0] > 0:
        zlim = (0, zlim[1])
    if zlim[1] < mag_factor_z:
        zlim = (zlim[0], mag_factor_z)
    
    # Set axis tick_labels
    tick_sep = mag_factor_z
    tick_values = np.arange(zlim[0]//tick_sep*tick_sep, zlim[1]//tick_sep*tick_sep + 2*tick_sep, tick_sep)
    ax.set_zticks(tick_values)
    ax.set_zticklabels([f'{val:0.0f}' for val in tick_values])  

    # Force axis limits
    ax.set_xlim([0, map_size])
    ax.set_ylim([0, map_size])
    ax.set_zlim([zlim[0], zlim[1]])
    ax.set_box_aspect([1, 1, 1])



    # Set axis labels
    ax.set_title(title, fontsize=fontsize+2)
    ax.set_xlabel('West-East [km]', fontweight='bold', labelpad=5, fontsize=fontsize)  # Adjust labelpad to reduce spacing
    ax.set_ylabel('North-South [km]', fontweight='bold', labelpad=5, fontsize=fontsize)
    ax.set_zlabel('Elevation [km]', fontweight='bold', labelpad=5, fontsize=fontsize)  # Adjust labelpad to reduce spacing
    plt.draw()

    name_ext = f'MapSize{map_size}km_Nx{nx}_cstride{cstride}_rstride{rstride}'
    if highlight_waves:
        name_ext += '_highlight_waves'
    

    return fig, ax, name_ext

def plot_initial_and_final_conditions(**plot_kwargs):

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

    DeltaX = MapSize / Nx # Grid spacing

    path_to_data = current_file_position / config['path_to_data']
    path_to_results = current_file_position/ config['path_to_results']
    path_to_human_results = path_to_results / 'human'
    path_to_data.mkdir(parents=True, exist_ok=True)
    path_to_results.mkdir(parents=True, exist_ok=True)
    path_to_human_results.mkdir(parents=True, exist_ok=True)

    data_file = path_to_data / f"Data_nx{Nx}_{MapSize}km_T{Tend}_h.bin"
    solution_file = path_to_results / f'Solution_nx{Nx}_{MapSize}km_T{Tend}_nprocess{number_of_processes}_h.bin'

    # Retieve topography
    topo = np.fromfile(open(path_to_data / f"Fig_nx{Nx}_{MapSize}km_Typography.bin", 'rb'), dtype='double').reshape((Nx, Nx)).T
    plots = []
    # Plot initial condition
    if data_file.exists():
        print(f'Plotting initial condition from {data_file}')
        H_initial_cond = np.fromfile(open(data_file, 'rb'), dtype='double').reshape((Nx, Nx)).T
        plot_kw = {'title': 'Initial conditions'}
        plot_kwargs.update(plot_kw)
        fig, ax, name_ext = plot_tsunami(H_initial_cond, MapSize, Nx, topo, **plot_kwargs)
        fig.savefig(path_to_human_results / f"Initial_condition_{name_ext}_T{str(Tend).split('.')[1]}.png", dpi=200, format='png', bbox_inches='tight')
    else:
        print(f'Did not find: {data_file}. Check your data paths.')
    plots.append([fig, ax, name_ext])
    # Plot solution state at Tend
    if solution_file.exists():
        print(f'Plotting solution from {solution_file}')
        H_solution = np.fromfile(open(solution_file, 'rb'), dtype='double').reshape((Nx, Nx)).T
        plot_kw = {'title': 'Result of the simulation'}
        plot_kwargs.update(plot_kw)
        fig, ax, name_ext = plot_tsunami(H_solution, MapSize, Nx, topo, **plot_kwargs)
        fig.savefig(path_to_human_results / f"Result_{name_ext}_T{str(Tend).split('.')[1]}.png", dpi=200, format='png', bbox_inches='tight')
    else:
        print(f'Did not find: {solution_file}. Did you remember to run compute.py file ?')
    plots.append([fig, ax, name_ext])

    return plots

if __name__ == '__main__':
    plot_initial_and_final_conditions(highlight_waves=True)