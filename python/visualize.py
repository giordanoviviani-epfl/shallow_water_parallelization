import numpy as np
import yaml
import re
from pathlib import Path
from matplotlib import colors
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt

def plot_tsunami(water_level, map_size, nx, topography, title='', highlight_waves=False, cstride=20, rstride=20,
                  save=False, save_path=None, save_name='Result', tag='', return_fig=False):

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

    if save: 
        if tag != '':
            save_name += f'_{tag}'
        save_name += f"_{map_size}km_nx{nx}_cstride{cstride}_rstride{rstride}"
        if highlight_waves:
            save_name += '_highlight_waves'
        save_name += '.png'
        
        save_complete_path = save_path / save_name
        print(f'Saving plot: {save_complete_path.stem}')
        fig.savefig(save_complete_path, dpi=200, format='png', bbox_inches='tight')

    if return_fig:    
        return fig, ax

def plot_tsunami_from_file(path_file, **plot_kwargs):

    find_number_prefix = lambda string, prefix: re.findall(r'%s\d+' % prefix, string)
    find_number_suffix = lambda string, suffix: re.findall(r'\d+%s' % suffix, string)

    ## Get file name
    path_file = Path(path_file)
    file_name = path_file.stem

    Nx = int(find_number_prefix(file_name, 'nx')[0])
    MapSize = int(find_number_suffix(file_name, 'km')[0])

    ## Load data
    water_level = np.fromfile(open(path_file, 'rb'), dtype=np.double).reshape((Nx, Nx))
    topo = np.fromfile(open(path_file.parent / f"Fig_nx{Nx}_{MapSize}km_Typography.bin", 'rb'), dtype=np.double).reshape((Nx, Nx))

    return plot_tsunami(water_level, MapSize, Nx, topo, **plot_kwargs)

def cartesian_grid(map_size, nx, dimensions_cartesian2d, local_size_array, slices_array, n_ghosts_array, cart_coords_array, ghost_corrected_offset_array, save_path): 

    size = dimensions_cartesian2d[0] * dimensions_cartesian2d[1]
    print(f"Plotting cartesian grid size={size}: ", flush=True, end='')

    height_ratios = np.ones(dimensions_cartesian2d[0], dtype=np.int32)
    width_ratios = np.ones(dimensions_cartesian2d[1], dtype=np.int32)

    # Plot the cartesian grid
    figsize = (dimensions_cartesian2d[1]*4, dimensions_cartesian2d[0]*4)
    fig, axs = plt.subplots(dimensions_cartesian2d[0], dimensions_cartesian2d[1], figsize=figsize, sharex=True, sharey=True, gridspec_kw={'height_ratios': height_ratios, 'width_ratios': width_ratios})
    for i_ax, ax in enumerate(axs.flat):
        matrix = np.zeros((nx, nx))
        matrix[slices_array[i_ax][0], slices_array[i_ax][1]] = 1 # Set entire collected matrix to 1
        
        submatrix_dims = np.array((matrix[slices_array[i_ax][0], slices_array[i_ax][1]]).shape)
        thickness_ghosts = submatrix_dims // 50
        # Get the ghost lines
        # Ghost rows
        row_index_ghost, col_index_ghost = [], []
        if n_ghosts_array[i_ax][0] == 1:
            if cart_coords_array[i_ax][0] == 0:
                matrix[slices_array[i_ax][0], slices_array[i_ax][1]][-thickness_ghosts[0]:, :] = -1
                row_index_ghost.append(ghost_corrected_offset_array[i_ax][0] + submatrix_dims[0] - 1)
            else:
                matrix[slices_array[i_ax][0], slices_array[i_ax][1]][0:thickness_ghosts[0]:, :] = -1
                row_index_ghost.append(ghost_corrected_offset_array[i_ax][0])

        elif n_ghosts_array[i_ax][0] == 2:
            matrix[slices_array[i_ax][0], slices_array[i_ax][1]][0:thickness_ghosts[0], :] = -1
            matrix[slices_array[i_ax][0], slices_array[i_ax][1]][-thickness_ghosts[0]:, :] = -1
            row_index_ghost.extend((ghost_corrected_offset_array[i_ax][0], ghost_corrected_offset_array[i_ax][0] + submatrix_dims[0] - 1))

        elif n_ghosts_array[i_ax][0] == 0:
            col_index_ghost.append(None)

        else:
            raise ValueError(f'ghosts_array[{i_ax}][0] must be either 1 or 2, but it is {n_ghosts_array[i_ax][0]}')
        # Ghost columns
        if n_ghosts_array[i_ax][1] == 1:
            if cart_coords_array[i_ax][1] == 0:
                matrix[slices_array[i_ax][0], slices_array[i_ax][1]][:, -thickness_ghosts[1]:] = -1
                col_index_ghost.append(ghost_corrected_offset_array[i_ax][1] + submatrix_dims[1] - 1)
            else:
                matrix[slices_array[i_ax][0], slices_array[i_ax][1]][:, 0:thickness_ghosts[1]] = -1
                col_index_ghost.append(ghost_corrected_offset_array[i_ax][1])

        elif n_ghosts_array[i_ax][1] == 2:
            matrix[slices_array[i_ax][0], slices_array[i_ax][1]][:, -thickness_ghosts[1]:] = -1
            matrix[slices_array[i_ax][0], slices_array[i_ax][1]][:, 0:thickness_ghosts[1]] = -1
            col_index_ghost.extend((ghost_corrected_offset_array[i_ax][1], ghost_corrected_offset_array[i_ax][1] + submatrix_dims[1] - 1))

        elif n_ghosts_array[i_ax][1] == 0:
            col_index_ghost.append(None)

        else:
            raise ValueError(f'ghosts_array[{i_ax}][1] must be either 1 or 2 or 0, but it is {n_ghosts_array[i_ax][1]}')

        ax.imshow(matrix, cmap='RdBu', vmin=-1.5, vmax=1.5, origin='upper', extent=[0, map_size, map_size, 0])

        yrange = [ghost_corrected_offset_array[i_ax][0], ghost_corrected_offset_array[i_ax][0] + local_size_array[i_ax][0]-1]
        xrange = [ghost_corrected_offset_array[i_ax][1], ghost_corrected_offset_array[i_ax][1] + local_size_array[i_ax][1]-1]  
        ax.set_xlabel(f'West-East [km]\n gcol {col_index_ghost} \n {xrange}', labelpad=1, fontsize=10)
        ax.set_ylabel(f'North-South [km]\n grow {row_index_ghost} \n {yrange}', labelpad=1, fontsize=10)
        ax.set_title(f'rank: {i_ax}\ncoord: {cart_coords_array[i_ax]}', fontweight='bold', fontsize=12)
        print(f' {i_ax+1} ', flush=True, end='')

    #Save the plot
    plt.tight_layout()
    print(f'\nSaving cartesian grid plot: {save_path.name}')
    fig.savefig(save_path, format='pdf')
