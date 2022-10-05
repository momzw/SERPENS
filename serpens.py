import rebound
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os as os
import sys
import json
import shutil
from datetime import datetime
from plotting import plotting
from init import Parameters

"""


       _____                                
      / ___/___  _________  ___  ____  _____
      \__ \/ _ \/ ___/ __ \/ _ \/ __ \/ ___/
     ___/ /  __/ /  / /_/ /  __/ / / (__  ) 
    /____/\___/_/  / .___/\___/_/ /_/____/  
                  /_/                       

    Simulating the Evolution of Ring Particles Emergent from Natural Satellites using Python


"""

Params = Parameters()

# Plotting
# ---------------------
save = True
save_archive = False
plot_freq = 5  # Plot at each *plot_freq* advance

showfig = False
showhist = False
showvel = False
show_column_density = False
"""
    ===============================================================================================================
"""


def getHistogram(sim, xdata, ydata, bins, xboundary="default", yboundary="default", plane='xy'):
    """
    Calculates a 2d histogram essentially defining a density distribution.
    :param sim: rebound simulation object.
    :param xdata: array_like. x-positions of particles to be analyzed.
    :param ydata: array_like. y-positions of particles to be analyzed.
    :param bins: int or array_like or [int, int] or [array, array]. See documentation for np.histogram2d.
    :return: H: ndarray (shape(nx, ny)), xedges: ndarray (shape(nx+1,)), yedges: ndarray (shape(ny+1,)). 2d-Histogram and bin edges along x- and y-axis.
    """
    r_max = Params.int_spec["r_max"]
    moon_exists = Params.int_spec["moon"]

    if moon_exists:
        moon_a = sim.particles["moon"].calculate_orbit(primary=sim.particles["planet"]).a
        if xboundary == "default":
            xboundary = r_max * moon_a
        if yboundary == "default":
            yboundary = r_max * moon_a
        if plane == 'xy':
            H, xedges, yedges = np.histogram2d(xdata, ydata, range=[
                [-xboundary + sim.particles["planet"].x, xboundary + sim.particles["planet"].x],
                [-yboundary + sim.particles["planet"].y, yboundary + sim.particles["planet"].y]], bins=bins)
        elif plane == 'yz':
            H, xedges, yedges = np.histogram2d(xdata, ydata, range=[
                [-xboundary + sim.particles["planet"].y, xboundary + sim.particles["planet"].y],
                [-yboundary + sim.particles["planet"].z, yboundary + sim.particles["planet"].z]], bins=bins)
        else:
            raise ValueError('Invalid plane in getHistogram.')
    else:
        planet_a = sim.particles["planet"].calculate_orbit(primary=sim.particles[0]).a
        if xboundary == "default":
            xboundary = r_max * planet_a
        if yboundary == "default":
            yboundary = r_max * planet_a
        H, xedges, yedges = np.histogram2d(xdata, ydata, range=[
            [-xboundary, xboundary],
            [-yboundary, yboundary]], bins=bins)

    H = H.T
    return H, xedges, yedges


def pngToGif(max_PNG_index, step):
    """
    TOOL TO COMBINE PNG TO GIF
    """
    import imageio
    filenames = [f'plots/sim_{i}.png' for i in range(0, max_PNG_index, step)]
    # with imageio.get_writer('mygif.gif', mode='I') as writer:
    #    for filename in filenames:
    #        image = imageio.imread(filename)
    #        writer.append_data(image)

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('movie.gif', images, fps=1)


if __name__ == "__main__":

    moon_exists = Params.int_spec["moon"]

    with open("hash_library.json") as f:
        hash_supdict = json.load(f)

    sa = rebound.SimulationArchive("archive.bin", process_warnings=False)
    if save:
        if moon_exists:
            path = datetime.utcnow().strftime("%d%m%Y--%H-%M") + "_moonsource"
        else:
            path = datetime.utcnow().strftime("%d%m%Y--%H-%M") + "_planetsource"
        os.makedirs(f'output/{path}/plots')
        with open(f"output/{path}/Parameters.txt", "w") as text_file:
            text_file.write(f"{Params.__str__()}")
        if save_archive:
            shutil.copy2(f"{os.getcwd()}/archive.bin", f"{os.getcwd()}/output/{path}")
            shutil.copy2(f"{os.getcwd()}/hash_library.json", f"{os.getcwd()}/output/{path}")

    for i, sim_instance in enumerate(sa):
        ps = sim_instance.particles

        species_names = []
        for ns in range(Params.num_species):
            species = Params.get_species(ns + 1)
            species_names.append(species.name)

        if not i == 0:
            hash_dict_current = hash_supdict[str(i)]
        else:
            hash_dict_current = {}

        temp = "id"
        ids = [val[temp] for key, val in hash_dict_current.items() if temp in val]

        xdata = np.zeros((sim_instance.N - sim_instance.N_active, Params.num_species))
        ydata = np.zeros((sim_instance.N - sim_instance.N_active, Params.num_species))
        zdata = np.zeros((sim_instance.N - sim_instance.N_active, Params.num_species))

        vxdata = np.zeros((sim_instance.N - sim_instance.N_active, Params.num_species))
        vydata = np.zeros((sim_instance.N - sim_instance.N_active, Params.num_species))
        vzdata = np.zeros((sim_instance.N - sim_instance.N_active, Params.num_species))

        bulk_velocities = np.zeros((sim_instance.N, 3), dtype="float64")
        sim_instance.serialize_particle_data(vxvyvz=bulk_velocities)

        rdata = np.zeros((sim_instance.N - sim_instance.N_active, Params.num_species))  # Distance from primary

        for k in range(sim_instance.N_active, sim_instance.N):
            ps_species = hash_dict_current[str(ps[k].hash.value)]["id"]
            species_num = np.where(np.unique(ids) == ps_species)

            xdata[int(k - sim_instance.N_active)][species_num] = ps[k].x
            ydata[int(k - sim_instance.N_active)][species_num] = ps[k].y
            zdata[int(k - sim_instance.N_active)][species_num] = ps[k].z

            vxdata[int(k - sim_instance.N_active)][species_num] = ps[k].vx
            vydata[int(k - sim_instance.N_active)][species_num] = ps[k].vy
            vzdata[int(k - sim_instance.N_active)][species_num] = ps[k].vz

            if moon_exists:
                rdata[int(k - sim_instance.N_active)][species_num] = (np.sqrt(
                    (ps[k].x - ps["planet"].x) ** 2 + (ps[k].y - ps["planet"].y) ** 2)) / ps["planet"].r
            else:
                rdata[int(k - sim_instance.N_active)][species_num] = (np.sqrt(
                    (ps[k].x - ps[0].x) ** 2 + (ps[k].y - ps[0].y) ** 2)) / ps[0].r

        if i % plot_freq == 0:

            def top_down_column(bins=160):
                # IMSHOW TOP DOWN COLUMN DENSITY PLOTS
                # ====================================
                subplot_rows = int(np.ceil(Params.num_species / 3))
                subplot_columns = Params.num_species if Params.num_species <= 3 else 3
                fig, axs = plt.subplots(subplot_rows, subplot_columns, figsize=(25, 12))
                for k in range(Params.num_species):
                    species = Params.get_species(k + 1)

                    H, xedges, yedges = getHistogram(sim_instance, xdata[:, k][xdata[:, k] != 0],
                                                     ydata[:, k][xdata[:, k] != 0], bins=bins)

                    bin_size = (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])

                    mass_inject_per_advance = species.mass_per_sec * Params.int_spec["sim_advance"] * sim_instance.particles["planet"].P
                    if not (species.n_th == 0 and species.n_sp == 0):
                        weight = species.particles_per_superparticle(mass_inject_per_advance) / bin_size / 10000
                    else:
                        weight = 1

                    ax_species = np.ndarray.flatten(axs)[k] if Params.num_species > 1 else axs
                    plotting(fig, ax_species, sim_instance, save=save, show=showfig, iter=i, histogram=H * weight,
                             xedges=xedges, yedges=yedges,
                             density=True)

                    species_total = np.count_nonzero(np.asarray(ids) == species.id)
                    ax_species.set_title(f"{species_names[k]} \n Number of superparticles: {species_total}", c='k',
                                         size='x-large', y=1.05)
                    ax_species.tick_params(axis='both', labelsize=8)
                    plt.setp(ax_species.get_xticklabels(), rotation=30, horizontalalignment='right')
                    ax_species.legend(borderpad = 1.2, fontsize = 'large')

                if moon_exists:
                    fig.suptitle(
                        f"Particle Simulation around Planetary Body \n Number of superparticles {sim_instance.N}",
                        size='xx-large')
                else:
                    fig.suptitle(
                        f"Particle Simulation around Stellar Body \n Number of superparticles {sim_instance.N}",
                        size='xx-large')
                plt.tight_layout()
                if save:
                    if moon_exists:
                        orbit_phase = np.around(sim_instance.particles["moon"].calculate_orbit(
                            primary=sim_instance.particles["planet"]).f * 180 / np.pi, 2)
                    else:
                        orbit_phase = np.around(sim_instance.particles["planet"].calculate_orbit(
                            primary=sim_instance.particles[0]).f * 180 / np.pi, 2)
                    frame_identifier = f"ColumnDensity_TopDown_{i}_{orbit_phase}"
                    plt.savefig(f'output/{path}/plots/{frame_identifier}.png')
                if showfig:
                    plt.show()
                plt.close()


            def toroidal_hist():
                # RADIAL HISTOGRAM
                # ================
                if not i == 0:
                    fig, ax = plt.subplots(figsize=(8, 8))
                    for k in range(Params.num_species):
                        counts, bin_edges = np.histogram(rdata[:, k][rdata[:, k] != 0], 100,
                                                         range=(0, 50))  # bins=int(np.sqrt(len(rdata[:,k])))

                        vx = vxdata[:, k][vxdata[:, k] != 0]
                        vy = vydata[:, k][vxdata[:, k] != 0]
                        vz = vzdata[:, k][vxdata[:, k] != 0]

                        if moon_exists:
                            speeds = [np.linalg.norm(np.vstack((vx, vy, vz)).T[j, :] - bulk_velocities[2, :]) for j in
                                      range(np.size(vx))]

                            v_ej = np.mean(speeds)
                            a_s = sim_instance.particles["moon"].calculate_orbit(
                                primary=sim_instance.particles["planet"]).a
                            v_orb = sim_instance.particles["moon"].calculate_orbit(
                                primary=sim_instance.particles["planet"]).v
                            V_tor = 2 * np.pi ** 2 * a_s ** 3 * (v_ej / v_orb) ** 2

                            weights = counts / V_tor / 1e6
                        else:
                            speeds = [np.linalg.norm(np.vstack((vx, vy, vz)).T[j, :] - bulk_velocities[1, :]) for j in
                                      range(np.size(vx))]

                            v_ej = np.mean(speeds)
                            a_s = sim_instance.particles["planet"].a
                            v_orb = sim_instance.particles["planet"].v
                            V_tor = 2 * np.pi ** 2 * a_s ** 3 * (v_ej / v_orb) ** 2

                            weights = counts / V_tor / 1e6

                        mass_inject_per_advance = Params.get_species(k + 1).mass_per_sec * Params.int_spec[
                            "sim_advance"] * sim_instance.particles["planet"].P
                        weights_phys = weights * Params.get_species(k + 1).particles_per_superparticle(
                            mass_inject_per_advance)

                        bincenters = (bin_edges[1:] + bin_edges[:-1]) / 2

                        ax.plot(bincenters[weights != 0], weights_phys[weights != 0], '-',
                                label=f"{Params.get_species(k + 1).name}", alpha=1)
                        ax.scatter(bincenters[weights != 0], weights_phys[weights != 0], marker='x')

                    ax.set_yscale("log")
                    ax.set_xlabel("Distance from primary in primary radii")
                    ax.set_ylabel(r"Toroidal Number Density $\frac{N}{V_{tor}}$")
                    ax.set_title("Toroidal Number Density")
                    plt.legend()
                    plt.grid(True)
                    if save:
                        frame_identifier = f"Toroidal_Density_{i}"
                        plt.savefig(f'output/{path}/plots/{frame_identifier}.png')
                    if showhist:
                        plt.show()
                    plt.close()
                else:
                    pass


            def los_column_and_velocity_dist(bins=160):
                # COLUMN DENSITY & VELOCITY DISTRIBUTION
                # ======================================
                subplot_rows = int(np.ceil(Params.num_species / 3))
                subplot_columns = Params.num_species if Params.num_species <= 3 else 3
                fig, axs = plt.subplots(subplot_rows, subplot_columns, figsize=(25, 12))
                fig.suptitle(r"Particle density in 1/cm$^2$", size='xx-large')

                for k in range(Params.num_species):
                    species = Params.get_species(k + 1)

                    if moon_exists:
                        zboundaryC = 10 * sim_instance.particles["planet"].r
                        mass_inject_per_advance = species.mass_per_sec * Params.int_spec["sim_advance"] * sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).P
                        orbit_phase = np.around(sim_instance.particles["moon"].calculate_orbit(
                            primary=sim_instance.particles["planet"]).f * 180 / np.pi, 2)

                        # H_Io = 100e3
                        # chapman = 2 * np.pi * sim_instance.particles["moon"].r / H_Io

                    else:
                        zboundaryC = 6 * sim_instance.particles[0].r
                        mass_inject_per_advance = species.mass_per_sec * Params.int_spec["sim_advance"] * sim_instance.particles["planet"].P
                        orbit_phase = np.around(sim_instance.particles["planet"].f * 180 / np.pi,2)

                    HC, yedgesC, zedgesC = getHistogram(sim_instance, ydata[:, k][ydata[:, k] != 0],
                                                        zdata[:, k][ydata[:, k] != 0], bins=bins, yboundary=zboundaryC,
                                                        plane='yz')
                    bin_size = (yedgesC[1] - yedgesC[0]) * (zedgesC[1] - zedgesC[0])
                    weight = species.particles_per_superparticle(mass_inject_per_advance) / bin_size / 10000

                    ax_species = np.ndarray.flatten(axs)[k] if Params.num_species > 1 else axs
                    plotting(fig, ax_species, sim_instance, save=save, show=showfig, iter=i, histogram=HC * weight,
                             xedges=yedgesC, yedges=zedgesC,
                             density=True, plane='yz')

                    species_total = np.count_nonzero(np.asarray(ids) == species.id)
                    ax_species.set_title(f"{species_names[k]} \n Number of superparticles: {species_total}", c='k',
                                         size='x-large')
                    ax_species.tick_params(axis='both', labelsize=8)
                    plt.setp(ax_species.get_xticklabels(), rotation=30, horizontalalignment='right')
                    ax_species.legend(borderpad=1.2, fontsize='large')
                plt.tight_layout()
                if save:
                    frame_identifier = f"ColumnDensity_LOS_{i}_{orbit_phase}"
                    plt.savefig(f'output/{path}/plots/{frame_identifier}.png')
                if show_column_density:
                    plt.show()
                plt.close()


            def vel_dist():
                if not i == 0:
                    if moon_exists:
                        orbit_phase = np.around(sim_instance.particles["moon"].calculate_orbit(
                            primary=sim_instance.particles["planet"]).f * 180 / np.pi, 2)
                        speeds = [np.linalg.norm(bulk_velocities[j, :] - bulk_velocities[2, :]) for j in
                                  range(np.size(bulk_velocities[:, 0]))]
                    else:
                        orbit_phase = np.around(sim_instance.particles["planet"].calculate_orbit(
                            primary=sim_instance.particles[0]).f * 180 / np.pi, 2)
                        speeds = [np.linalg.norm(bulk_velocities[j, :] - bulk_velocities[1, :]) for j in
                                  range(np.size(bulk_velocities[:, 0]))]

                    scale = Params.sput_spec["model_maxwell_max"] / np.sqrt(2)
                    def maxwell_func(x):
                        return np.sqrt(2/np.pi) * (x/scale)**2 * np.exp(-(x/scale)**2 / 2) / scale

                    a = Params.sput_spec['model_smyth_a']
                    v_M = Params.sput_spec['model_smyth_v_M']
                    v_b = Params.sput_spec['model_smyth_v_b']
                    x_vals = np.linspace(0, v_M, 10000)
                    def sput_func(x, a, v_b, v_M):
                        f_v = 1 / v_b * (x / v_b) ** 3 \
                              * (v_b ** 2 / (v_b ** 2 + x ** 2)) ** a \
                              * (1 - np.sqrt((x ** 2 + v_b ** 2) / (v_M ** 2)))
                        return f_v
                    def sput_func_int(x, a, v_b, v_M):
                        integral_bracket = (1 + x ** 2) ** (5 / 2) * v_b / v_M / (2 * a - 5) - (1 + x ** 2) ** (
                                    3 / 2) * v_b / v_M / (
                                                   2 * a - 3) - x ** 4 / (2 * (a - 2)) - a * x ** 2 / (
                                                       2 * (a - 2) * (a - 1)) - 1 / (
                                                   2 * (a - 2) * (a - 1))

                        f_v_integrated = integral_bracket * (1 + x ** 2) ** (-a)
                        return f_v_integrated
                    upper_bound = np.sqrt((v_M / v_b) ** 2 - 1)
                    normalization = 1 / (sput_func_int(upper_bound, a, v_b, v_M) - sput_func_int(0, a, v_b, v_M))
                    f_pdf = normalization * sput_func(x_vals, a, v_b, v_M)

                    plt.hist(speeds[sim_instance.N_active:], 100, range=(0, v_M), density=True)
                    plt.plot(x_vals, f_pdf, c='r', label='Smyth')
                    plt.plot(x_vals, maxwell_func(x_vals), c="b", label="Maxwell")
                    plt.ylabel("Probability", fontsize='x-large')
                    plt.xlabel("Velocity in m/s", fontsize='x-large')
                    plt.title("Velocity Distribution")
                    plt.legend()
                    if save:
                        frame_identifier = f"Velocity_Distribution_{i}_{orbit_phase}"
                        plt.savefig(f'output/{path}/plots/{frame_identifier}.png')
                    if showvel:
                        plt.show()
                    plt.close()
                else:
                    pass


            def mayavi_3D_density():
                if not i == 0:
                    from scipy import stats
                    from mayavi import mlab
                    import multiprocessing

                    def calc_kde(data):
                        return kde(data.T)

                    x = xdata[:, 0]
                    y = ydata[:, 0]
                    z = zdata[:, 0]

                    xyz = np.vstack([x, y, z])
                    kde = stats.gaussian_kde(xyz)

                    # Evaluate kde on a grid
                    xmin, ymin, zmin = x.min(), y.min(), z.min()
                    xmax, ymax, zmax = x.max(), y.max(), z.max()
                    xi, yi, zi = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
                    coords = np.vstack([item.ravel() for item in [xi, yi, zi]])

                    # Multiprocessing
                    cores = multiprocessing.cpu_count()
                    pool = multiprocessing.Pool(processes=cores)
                    results = pool.map(calc_kde, np.array_split(coords.T, 2))
                    density = np.concatenate(results).reshape(xi.shape)

                    # Plot scatter with mayavi
                    figure = mlab.figure('DensityPlot')

                    grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
                    min = density.min()
                    max = density.max()
                    mlab.pipeline.volume(grid, vmin=min, vmax=min + .5 * (max - min))

                    mlab.axes()
                    mlab.show()
                else:
                    pass


            top_down_column(bins=200)
            # mayavi_3D_density()
            los_column_and_velocity_dist(bins = 200)
            toroidal_hist()
            vel_dist()
