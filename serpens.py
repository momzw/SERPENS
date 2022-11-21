import rebound
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
mplstyle.use('fast')
import time
import numpy as np
import os as os
import shutil

import pickle

from datetime import datetime
from plotting import plotting
from init import Parameters
from visualize import Visualize

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut
import KDEpy
from KDEpy.bw_selection import improved_sheather_jones

import DTFE
import DTFE3D

"""


       _____                                
      / ___/___  _________  ___  ____  _____
      \__ \/ _ \/ ___/ __ \/ _ \/ __ \/ ___/
     ___/ /  __/ /  / /_/ /  __/ / / (__  ) 
    /____/\___/_/  / .___/\___/_/ /_/____/  
                  /_/                       

    Simulating the Evolution of Ring Particles Emergent from Natural Satellites using Python 


"""

# TODO: Ions and ENAs around torus
# TODO: Multiprocessing improvements
# TODO: MAYBE: Check if in-between advance modifications can be written as a reboundx operator.


Params = Parameters()

# Plotting
# ---------------------
save = False
save_archive = False
save_particles = False
plot_freq = 20  # Plot at each *plot_freq* advance

showfig = True
showhist = False
showvel = False
show_column_density = False
skip_first = True

"""
    ===============================================================================================================
"""


def getHistogram(sim, xdata, ydata, weights, bins, xboundary="default", yboundary="default", plane='xy', mask=False, density=False, **kwargs):
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

        if mask:
            dist_to_moon = np.linalg.norm(sim.particles["moon"].xyz)
            xdata[dist_to_moon < 2 * sim.particles["moon"].r] = 0
            ydata[dist_to_moon < 2 * sim.particles["moon"].r] = 0
        if xboundary == "default":
            xboundary = r_max * moon_a
        if yboundary == "default":
            yboundary = r_max * moon_a

        if plane == 'xy':
            H, xedges, yedges = np.histogram2d(xdata, ydata, range=[
                [-xboundary + sim.particles["planet"].x, xboundary + sim.particles["planet"].x],
                [-yboundary + sim.particles["planet"].y, yboundary + sim.particles["planet"].y]], bins=bins)        # weights=weights
            H = H.T
            return H, xedges, yedges

        elif plane == 'yz':
            H, xedges, yedges = np.histogram2d(xdata, ydata, range=[
                [-xboundary + sim.particles["planet"].y, xboundary + sim.particles["planet"].y],
                [-yboundary + sim.particles["planet"].z, yboundary + sim.particles["planet"].z]], bins=bins, weights=weights)
            H = H.T
            return H, xedges, yedges

        elif plane == '3d':
            zboundary = kwargs.get("zboundary", r_max * moon_a)
            zdata = kwargs.get("zdata", None)
            if zdata.any() is None:
                raise AttributeError("zdata needed for a 3d-histogram.")
            if mask:
                zdata = np.delete(zdata, np.where(dist_to_moon < 10 * sim.particles["planet"].r))
            H3d, edges3d = np.histogramdd(np.vstack((xdata, ydata, zdata)).T, bins=(bins, bins, bins), range=[
                [-xboundary + sim.particles["planet"].x, xboundary + sim.particles["planet"].x],
                [-yboundary + sim.particles["planet"].y, yboundary + sim.particles["planet"].y],
                [-zboundary + sim.particles["planet"].z, zboundary + sim.particles["planet"].z]], weights=weights)
            return H3d, edges3d

        else:
            raise ValueError('Invalid plane in getHistogram.')

    else:
        planet_a = sim.particles["planet"].calculate_orbit(primary=sim.particles[0]).a

        if mask:
            dist_to_planet = np.sqrt((xdata - sim.particles["planet"].x)**2 + (ydata - sim.particles["planet"].y)**2)
            xdata = np.delete(xdata, np.where(dist_to_planet < 10 * sim.particles["planet"].r))
            ydata = np.delete(ydata, np.where(dist_to_planet < 10 * sim.particles["planet"].r))
        if xboundary == "default":
            xboundary = r_max * planet_a
        if yboundary == "default":
            yboundary = r_max * planet_a

        if plane=='3d':
            zboundary = kwargs.get("zboundary", r_max * planet_a)
            zdata = kwargs.get("zdata", None)
            if zdata.any() == None:
                raise AttributeError("zdata needed for a 3d-histogram.")
            if mask:
                zdata = np.delete(zdata, np.where(dist_to_planet < 10 * sim.particles["planet"].r))
            H3d, edges3d = np.histogramdd(np.vstack((xdata, ydata, zdata)).T, bins=(bins, bins, bins), range=[
                [-xboundary, xboundary],
                [-yboundary, yboundary],
                [-zboundary, zboundary]], density=density, weights=weights)
            return H3d, edges3d

        else:
            H, xedges, yedges = np.histogram2d(xdata, ydata, range=[
                [-xboundary, xboundary],
                [-yboundary, yboundary]], bins=bins, density=density, weights=weights)
            H = H.T
            return H, xedges, yedges


if __name__ == "__main__":

    moon_exists = Params.int_spec["moon"]

    sa = rebound.SimulationArchive("archive.bin", process_warnings=False)
    if save:
        print("Copying and saving...")
        if moon_exists:
            path = datetime.utcnow().strftime("%d%m%Y--%H-%M") + "_moonsource"
        else:
            path = datetime.utcnow().strftime("%d%m%Y--%H-%M") + "_planetsource"

        os.makedirs(f'output/{path}/plots')

        try:
            os.replace('Parameters.txt', f'output/{path}/Parameters.txt')
        except:
            pass

        if save_archive:
            print("\t archive...")
            shutil.copy2(f"{os.getcwd()}/archive.bin", f"{os.getcwd()}/output/{path}")
            print("\t hash library...")
            shutil.copy2(f"{os.getcwd()}/hash_library.pickle", f"{os.getcwd()}/output/{path}")
            print("\t ...done!")

    # Use for SMALL FILES:
    # ===================
    #with open("hash_library.json", 'rb') as f:
    #    hash_supdict = json.load(f)

    # Use for LARGE FILES:
    # ===================
    #print("Collecting hash library:")
    #hash_supdict = {}
    #hash_dict = {}
    #flag = 0
    #index = plot_freq
    #with open("hash_library.json", 'rb') as f:
    #    parser = ijson.parse(f)
    #    for prefix, event, value in parser:
    #        # print((prefix, event, value))
    #        if (prefix, event) == (f'{index}', 'map_key'):
    #            particle_hash = value
    #            flag = 1
    #        elif flag == 1:
    #            if (prefix, event) == (f'{index}.{particle_hash}.identifier', 'string'):
    #                ident = value
    #            elif (prefix, event) == (f'{index}.{particle_hash}.i', 'number'):
    #                iter = value
    #            elif (prefix, event) == (f'{index}.{particle_hash}.id', 'number'):
    #                id = value
    #            elif (prefix, event) == (f'{index}.{particle_hash}', 'end_map'):
    #                hash_dict[f"{particle_hash}"] = {'identifier': ident, 'i': iter, 'id': id}
    #                flag = 0
    #        if (prefix, event) == (f'{index}', 'end_map'):
    #            hash_supdict[f'{index}'] = hash_dict.copy()
    #            print(f"\t ...collected snapshot {index}")
    #            index += plot_freq
    #print("\t ...done")

    # PICKLE:
    # ===================
    #print("WARNING: SERPENS is about to unpickle particle data. Pickle files are not secure. Make sure you trust the source!")
    #input("\t Press Enter to continue...")
    with open('hash_library.pickle', 'rb') as handle:
        hash_supdict = pickle.load(handle)

    for i, sim_instance in enumerate(sa):

        if not i % plot_freq == 0:
            continue
        elif skip_first and i == 0:
            continue

        start_time = time.time()

        ps = sim_instance.particles

        species_names = []
        for ns in range(Params.num_species):
            species = Params.get_species(ns + 1)
            species_names.append(species.description)

        if not i == 0:
            hash_dict_current = hash_supdict[str(i)]
        else:
            hash_dict_current = {}

        ids = [val["id"] for key, val in hash_dict_current.items() if "id" in val]

        particle_positions = np.zeros((sim_instance.N, 3), dtype="float64")
        particle_velocities = np.zeros((sim_instance.N, 3), dtype="float64")
        particle_hashes = np.zeros(sim_instance.N, dtype="uint32")
        particle_species = np.zeros(sim_instance.N, dtype="int")
        particle_weights = np.zeros(sim_instance.N, dtype="float64")
        sim_instance.serialize_particle_data(xyz=particle_positions, vxvyvz=particle_velocities,
                                             hash=particle_hashes)

        for k1 in range(sim_instance.N_active, sim_instance.N):
            particle_species[k1] = hash_dict_current[str(particle_hashes[k1])]["id"]
            particle_iter = hash_dict_current[str(particle_hashes[k1])]["i"]

            if moon_exists:
                particle_time = (i - particle_iter) * Params.int_spec["sim_advance"] * sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).P
            else:
                particle_time = (i - particle_iter) * Params.int_spec["sim_advance"] * sim_instance.particles["planet"].P
            chem_network = Params.get_species_by_id(particle_species[k1]).network
            reaction_rate = 0
            if not isinstance(chem_network, (int, float)):
                for l in range(np.size(chem_network[:, 0])):
                    reaction_rate += 1/float(chem_network[:,0][l])
            else:
                reaction_rate = 1/chem_network
            particle_weights[k1] = np.exp(-particle_time * reaction_rate)

            #particle_weights[k1] = hash_dict_current[str(particle_hashes[k1])]["weight"]


        def top_down_column(bins=160):
            # IMSHOW TOP DOWN COLUMN DENSITY PLOTS
            # ====================================

            for k in range(Params.num_species):
                species = Params.get_species(k + 1)

                xdata = particle_positions[:, 0][np.where(particle_species == species.id)]
                ydata = particle_positions[:, 1][np.where(particle_species == species.id)]
                zdata = particle_positions[:, 2][np.where(particle_species == species.id)]

                weights = particle_weights[np.where(particle_species == species.id)]

                H, xedges, yedges = getHistogram(sim_instance, xdata, ydata, weights, bins=bins, mask=False)
                bin_size = (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])

                H3d, edges3d = getHistogram(sim_instance, xdata, ydata, weights, bins=bins, mask=False, plane='3d', zdata=zdata)
                bin_volume = (edges3d[0][1] - edges3d[0][0]) * (edges3d[1][1] - edges3d[1][0]) * (edges3d[2][1] - edges3d[2][0])

                if moon_exists:
                    mass_inject_per_advance = species.mass_per_sec * Params.int_spec["sim_advance"] * sim_instance.particles["moon"].calculate_orbit(
                            primary=sim_instance.particles["planet"]).P
                else:
                    mass_inject_per_advance = species.mass_per_sec * Params.int_spec["sim_advance"] * sim_instance.particles["planet"].P

                if not (species.n_th == 0 and species.n_sp == 0):
                    weight = species.particles_per_superparticle(mass_inject_per_advance) / (bin_size * 10000)
                    weight3d = 1 / bin_volume * species.particles_per_superparticle(mass_inject_per_advance) / 1e6
                else:
                    weight = 1
                    weight3d = 1

                # ============================================================================================

                def kde(x, y, weights, module, z=None, bandwidth='default', bins=100j, zbins=10j, kernel='gaussian', draw_contour=False, estimate_bandwidth=False, **kwargs):

                    if module == 'sklearn':
                        # Kernel Density Estimation using SKLEARN
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        if z is None:

                            if moon_exists:
                                a = sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).a
                                boundary = Params.int_spec["r_max"] * a
                                bw = 2 * a * np.sin(2 * np.pi * Params.int_spec["sim_advance"]) if bandwidth == 'default' else bandwidth
                                xx, yy = np.mgrid[
                                         (-boundary + sim_instance.particles["planet"].x):(boundary + sim_instance.particles["planet"].x):bins,
                                         (-boundary + sim_instance.particles["planet"].y):(boundary + sim_instance.particles["planet"].y):bins]
                            else:
                                a = sim_instance.particles["planet"].a
                                boundary = Params.int_spec["r_max"] * a
                                bw = 2 * a * np.sin(2 * np.pi * Params.int_spec["sim_advance"]) if bandwidth == 'default' else bandwidth
                                xx, yy = np.mgrid[-boundary:boundary:bins, -boundary:boundary:bins]

                            data = np.vstack([y, x]).T
                            xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T

                            kernel_density = KernelDensity(kernel=kernel, bandwidth=bw).fit(data, sample_weight=weights)
                            #print("Pulling KDE probabilities")
                            Z = np.exp(kernel_density.score_samples(xy_sample).reshape(xx.shape))
                            #print("\t Done!")

                            bin_size = (xx[:,0][1] - xx[:,0][0]) * (yy[0,:][1] - yy[0,:][0])
                            N_c = len(x) * Z * bin_size
                            N_per_cm2 = N_c * species.particles_per_superparticle(mass_inject_per_advance) / (bin_size * 1e4)

                            if draw_contour:
                                ax = kwargs.get('ax', None)
                                fill = kwargs.get('contour_fill', False)
                                norm = colors.LogNorm()
                                lvls = np.logspace(np.log10(np.max(Z)) - 5, np.log10(np.max(Z)), 8)
                                if fill:
                                    ax.contourf(xx, yy, Z, cmap=matplotlib.cm.afmhot, levels=lvls, norm=norm)
                                ax.contour(xx, yy, Z, colors='w', alpha=0.25, norm=norm, levels=lvls)

                            return Z, N_per_cm2, xx, yy

                        else:
                            if not z.shape == x.shape:
                                raise ValueError("z-shape is not equal to x-shape. Cannot pass 3D data.")
                            data3d = np.vstack([zdata, ydata, xdata]).T
                            if moon_exists:
                                a = sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).a
                                boundary = Params.int_spec["r_max"] * a

                                if estimate_bandwidth:
                                    print("Estimating bandwidth. This may take a while...")
                                    bandwidths = 2 * np.pi * Params.int_spec["sim_advance"] * a * 10 ** np.linspace(-2, 3, 100)
                                    grid = GridSearchCV(KernelDensity(kernel=kernel), {'bandwidth': bandwidths}, n_jobs=-1)
                                    grid.fit(data3d)
                                    bw = grid.best_estimator_.bandwidth
                                    print(f"\t Best bandwidth: {bw}")
                                else:
                                    bw = 2 * a * np.sin(2 * np.pi * Params.int_spec["sim_advance"]) if bandwidth == 'default' else bandwidth

                                xxx, yyy, zzz = np.mgrid[
                                                (-boundary + sim_instance.particles["planet"].x):(boundary + sim_instance.particles["planet"].x):bins,
                                                (-boundary + sim_instance.particles["planet"].y):(boundary + sim_instance.particles["planet"].y):bins,
                                                (-boundary + sim_instance.particles["planet"].z)/2:(boundary + sim_instance.particles["planet"].z)/2:zbins]

                                #xmin, xmax = np.min(xdata), np.max(xdata)
                                #ymin, ymax = np.min(ydata), np.max(ydata)
                                #zmin, zmax = np.min(zdata), np.max(zdata)
                                #xxx, yyy, zzz = np.mgrid[xmin:xmax:bins, ymin:ymax:bins, zmin:zmax:zbins]
                            else:
                                a = sim_instance.particles["planet"].a
                                boundary = Params.int_spec["r_max"] * a
                                bw = 2 * a * np.sin(2 * np.pi * Params.int_spec["sim_advance"]) if bandwidth == 'default' else bandwidth
                                xxx, yyy, zzz = np.mgrid[-boundary:boundary:bins, -boundary:boundary:bins, -boundary/2:boundary/2:zbins]

                            xyz_sample = np.vstack([zzz.ravel(), yyy.ravel(), xxx.ravel()]).T

                            kde3d = KernelDensity(kernel=kernel, bandwidth=bw).fit(data3d, sample_weight=weights)
                            #print("Pulling KDE probabilities: ")
                            Z3d = np.exp(kde3d.score_samples(xyz_sample).reshape(xxx.shape))
                            #print("\t Done!")

                            volume = (xxx[1,0,0] - xxx[0,0,0]) * (yyy[0,1,0] - yyy[0,0,0]) * (zzz[0,0,1] - zzz[0,0,0])
                            w_samp = mass_inject_per_advance / species.m
                            N_per_cm3 = Z3d * w_samp / 1e6

                            return Z3d, N_per_cm3, xxx, yyy, zzz

                    elif module == 'KDEpy':
                        # Kernel Density Estimation with KDEpy
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        data = np.vstack([z, y, x]).T
                        grid_points = bins

                        if moon_exists:
                            moon_a = sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).a
                            r_max = Params.int_spec["r_max"]
                            boundary = r_max * moon_a
                            mask = [x < boundary for x in np.linalg.norm(data - np.flip(sim_instance.particles["planet"].xyz).reshape((1,3)), axis=1)]
                            data = data[mask]
                            weights = weights[mask]
                        else:
                            boundary = Params.int_spec["r_max"] * sim_instance.particles["planet"].a
                            mask = [x < boundary for x in np.linalg.norm(data, axis=1)]
                            data = data[mask]
                            weights = weights[mask]

                        if moon_exists:
                            a = sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).a
                        else:
                            a = sim_instance.particles["planet"].a
                        bw = a * np.sin(2 * np.pi * Params.int_spec["sim_advance"])
                        bwx = improved_sheather_jones(data[:,2].reshape((len(data), 1)), weights)
                        bwy = improved_sheather_jones(data[:,1].reshape((len(data), 1)), weights)
                        bwz = improved_sheather_jones(data[:,0].reshape((len(data), 1)), weights)
                        bw_factor = (bwx/data[:,2].std() + bwy/data[:,1].std() + bwz/data[:,0].std()) / 3

                        kde = KDEpy.FFTKDE(kernel=kernel, norm=2, bw=bw_factor*10)
                        grid, points = kde.fit(data, weights=weights).evaluate(grid_points)

                        z = points.reshape(grid_points, grid_points, grid_points).T
                        bin_volume = (grid[:,0][grid_points**2] - grid[:,0][grid_points**2-1]) * (grid[:,1][grid_points] - grid[:,1][grid_points - 1]) * (grid[:,2][1] - grid[:,2][0])

                        z *= species.particles_per_superparticle(mass_inject_per_advance) / (bin_volume * 1e6)

                        if draw_contour:
                            x = np.unique(grid[:,2])
                            y = np.unique(grid[:,1])
                            xx = np.repeat(x[:, np.newaxis], bins, axis=1)
                            yy = np.repeat(y[np.newaxis, :], bins, axis=0)
                            ax = kwargs.get('ax', None)
                            fill = kwargs.get('contour_fill', False)
                            norm = colors.LogNorm()
                            Z = np.max(z, axis=2)
                            lvls = np.logspace(np.log10(np.max(Z)) - 3, np.log10(np.max(Z)), 8)
                            if fill:
                                ax.contourf(xx, yy, Z, cmap=matplotlib.cm.afmhot, levels=lvls, norm=norm)
                            ax.contour(xx, yy, Z, colors='w', alpha=0.25, norm=norm, levels=lvls)

                        return z, grid

                #z, grid = kde(x=xdata, y=ydata, weights=weights, module ='KDEpy', z=zdata, bins=200, draw_contour=True, ax=ax_species, contour_fill=False)

                # ============================================================================================

                points = np.vstack([xdata,ydata, zdata]).T
                vx = particle_velocities[:, 0][np.where(particle_species == species.id)]
                vy = particle_velocities[:, 1][np.where(particle_species == species.id)]
                vz = particle_velocities[:, 2][np.where(particle_species == species.id)]
                velocities = np.vstack([vx, vy, vz]).T

                simulation_time = i * Params.int_spec["sim_advance"] * sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).P
                total_injected = i * (species.n_sp + species.n_th)
                remaining_part = len(xdata)
                mass_in_system = remaining_part / total_injected * species.mass_per_sec * simulation_time
                superpart_mass = mass_in_system / remaining_part
                number_per_superpart = superpart_mass / species.m
                phys_weights = number_per_superpart * weights


                #L = 128
                #boundary = Params.int_spec["r_max"] * sim_instance.particles["moon"].calculate_orbit(
                #    primary=sim_instance.particles["planet"]).a
                #X, Y = np.meshgrid(np.linspace(-boundary + sim_instance.particles["planet"].x, boundary + sim_instance.particles["planet"].x, L),
                #                   np.linspace(-boundary + sim_instance.particles["planet"].y, boundary + sim_instance.particles["planet"].y, L))

                print("Constructing DTFE ...")
                dtfe = DTFE3D.DTFE(points, velocities, superpart_mass)
                # dens = dtfe.density(X.flat, Y.flat, Z.flat).reshape((L,L,L))
                dtfe2d = DTFE.DTFE(points[:,:2], velocities[:,:2], phys_weights)
                dens_plot = dtfe.density(points[:, 0], points[:, 1], points[:, 2]) / 1e6 / species.m * weights
                print("\t ... done!")

                vis = Visualize(sim_instance)
                vis.add_histogram(k, H, xedges, yedges, perspective="topdown")
                vis.add_triplot(k, points[:, 0], points[:, 1], dtfe2d.delaunay.simplices)
                #vis.add_dtfe(k, points[:, 0], points[:, 1], dens_plot)
                vis()

        def toroidal_hist():
            # RADIAL HISTOGRAM
            # ================
            if not i == 0:
                fig, ax = plt.subplots(figsize=(8, 8))
                for k in range(Params.num_species):
                    species = Params.get_species(k + 1)

                    psx = particle_positions[:, 0][np.where(particle_species == species.id)][sim_instance.N_active:]
                    psy = particle_positions[:, 1][np.where(particle_species == species.id)][sim_instance.N_active:]
                    psz = particle_positions[:, 2][np.where(particle_species == species.id)][sim_instance.N_active:]

                    psvx = particle_velocities[:, 0][np.where(particle_species == species.id)][sim_instance.N_active:]
                    psvy = particle_velocities[:, 0][np.where(particle_species == species.id)][sim_instance.N_active:]
                    psvz = particle_velocities[:, 0][np.where(particle_species == species.id)][sim_instance.N_active:]

                    if moon_exists:

                        rdata = np.sqrt((psx - ps["planet"].x) ** 2 + (psy - ps["planet"].y) ** 2 + (psz - ps["planet"].z) ** 2) / ps["planet"].r

                        counts, bin_edges = np.histogram(rdata, 100, range=(0, 50))  # bins=int(np.sqrt(len(rdata[:,k])))

                        speeds = [np.linalg.norm(np.vstack((psvx, psvy, psvz)).T[j, :] - particle_velocities[2, :]) for j in range(np.size(psvx))]

                        v_ej = np.mean(speeds)
                        a_s = sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).a
                        v_orb = sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).v
                        V_tor = 2 * np.pi ** 2 * a_s ** 3 * (v_ej / v_orb) ** 2

                        weights = counts / V_tor / 1e6

                        mass_inject_per_advance = Params.get_species(k + 1).mass_per_sec * Params.int_spec["sim_advance"] * sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).P
                        weights_phys = weights * Params.get_species(k + 1).particles_per_superparticle( mass_inject_per_advance)

                    else:

                        rdata = np.sqrt(psx ** 2 + psy ** 2 + psz ** 2) / ps["planet"].r

                        counts, bin_edges = np.histogram(rdata, 100, range=(0, 50))

                        speeds = [np.linalg.norm(np.vstack((psvx, psvy, psvz)).T[j, :] - particle_velocities[1, :]) for j in range(np.size(psvx))]

                        v_ej = np.mean(speeds)
                        a_s = sim_instance.particles["planet"].a
                        v_orb = sim_instance.particles["planet"].v
                        V_tor = 2 * np.pi ** 2 * a_s ** 3 * (v_ej / v_orb) ** 2

                        weights = counts / V_tor / 1e6

                        mass_inject_per_advance = Params.get_species(k + 1).mass_per_sec * Params.int_spec["sim_advance"] * sim_instance.particles["planet"].P
                        weights_phys = weights * Params.get_species(k + 1).particles_per_superparticle(mass_inject_per_advance)

                    bincenters = (bin_edges[1:] + bin_edges[:-1]) / 2

                    ax.plot(bincenters[weights != 0], weights_phys[weights != 0], '-', label=f"{Params.get_species(k + 1).description}", alpha=1)
                    ax.scatter(bincenters[weights != 0], weights_phys[weights != 0], marker='x')

                ax.set_yscale("log")
                ax.set_xlabel("Distance from primary in primary radii")
                ax.set_ylabel(r"Toroidal Number Density $\frac{N}{V_{tor}}$")
                ax.set_title("Toroidal Number Density")
                plt.legend(loc='upper right')
                plt.grid(True)
                if save:
                    frame_identifier = f"Toroidal_Density_{i}"
                    plt.savefig(f'output/{path}/plots/{frame_identifier}.png')
                if showhist:
                    plt.show()
                plt.close()
            else:
                pass

        def los_column_and_velocity_dist(bins=100):
            # COLUMN DENSITY & VELOCITY DISTRIBUTION
            # ======================================

            for k in range(Params.num_species):
                species = Params.get_species(k + 1)

                ydata = particle_positions[:, 1][np.where(particle_species == species.id)]
                zdata = particle_positions[:, 2][np.where(particle_species == species.id)]

                weights = particle_weights[np.where(particle_species == species.id)]

                if moon_exists:
                    zboundaryC = 10 * sim_instance.particles["planet"].r
                    mass_inject_per_advance = species.mass_per_sec * Params.int_spec["sim_advance"] * sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).P
                    #H_Io = 100e3
                    #chapman = np.sqrt(2 * np.pi * sim_instance.particles["moon"].r / H_Io)

                else:
                    zboundaryC = 6 * sim_instance.particles[0].r
                    mass_inject_per_advance = species.mass_per_sec * Params.int_spec["sim_advance"] * sim_instance.particles["planet"].P

                HC, yedgesC, zedgesC = getHistogram(sim_instance, ydata, zdata, weights=weights, bins=bins, yboundary=zboundaryC, plane='yz')
                bin_size = (yedgesC[1] - yedgesC[0]) * (zedgesC[1] - zedgesC[0])
                weight = species.particles_per_superparticle(mass_inject_per_advance) / bin_size / 10000

                #kde = KDEpy.FFTKDE(kernel='gaussian', norm=2, bw=0.001)
                #grid_points = bins
                #grid, points = kde.fit(np.vstack([zdata, ydata]).T, weights=weights).evaluate(grid_points)
                #z = points.reshape(grid_points, grid_points).T
                #bin_size = (np.unique(grid[:,0])[1] - np.unique(grid[:,0])[0]) * (np.unique(grid[:,1])[1] - np.unique(grid[:,1])[0])
                #
                #z *= species.particles_per_superparticle(mass_inject_per_advance) / (bin_size * 1e4)

                simulation_time = i * Params.int_spec["sim_advance"] * sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).P
                total_injected = i * (species.n_sp + species.n_th)
                remaining_part = len(ydata)
                mass_in_system = remaining_part / total_injected * species.mass_per_sec * simulation_time
                superpart_mass = mass_in_system / remaining_part

                points = np.vstack([ydata, zdata]).T
                vy = particle_velocities[:, 1][np.where(particle_species == species.id)]
                vz = particle_velocities[:, 2][np.where(particle_species == species.id)]
                velocities = np.vstack([vy, vz]).T

                dtfe2d = DTFE.DTFE(points, velocities, superpart_mass)
                dens_los = dtfe2d.density(points[:, 0], points[:, 1]) / 1e4 / species.m * weights

                vis = Visualize(sim_instance)
                vis.add_dtfe(k, points[:,0], points[:,1], dens_los, perspective="los", cb_format='%.2E')
                vis.add_triplot(k, points[:, 0], points[:, 1], dtfe2d.delaunay.simplices, perspective='los')
                vis()

                #xx = np.repeat(np.unique(grid[:,1])[:, np.newaxis], bins, axis=1)
                #yy = np.repeat(np.unique(grid[:,0])[np.newaxis, :], bins, axis=0)
                #norm = colors.LogNorm()
                #lvls = np.logspace(10, np.log10(np.max(z)), 12)
                #ax_species.contourf(xx, yy, z, cmap=matplotlib.cm.afmhot, levels=lvls, norm=norm)
                #ax_species.contour(xx, yy, z, colors='w', alpha=0.25, norm=norm, levels=lvls)

        def vel_dist():
            if not i == 0:

                subplot_rows = Params.num_species
                fig, axs = plt.subplots(subplot_rows, 1, figsize=(15, 10))
                fig.suptitle(r"Particle density in 1/cm$^2$", size='xx-large')

                def maxwell_func(x):
                    return np.sqrt(2/np.pi) * (x/scale)**2 * np.exp(-(x/scale)**2 / 2) / scale

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

                for k in range(Params.num_species):
                    species = Params.get_species(k + 1)

                    vel = particle_velocities[np.where(particle_species == species.id)]

                    if moon_exists:
                        orbit_phase = np.around(sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).f * 180 / np.pi, 2)
                        speeds = [np.linalg.norm(vel[j, :] - particle_velocities[2, :]) for j in range(np.size(vel[:, 0]))]
                    else:
                        orbit_phase = np.around(sim_instance.particles["planet"].f * 180 / np.pi, 2)
                        speeds = [np.linalg.norm(vel[j, :] - particle_velocities[1, :]) for j in range(np.size(vel[:, 0]))]

                    scale = species.sput_spec["model_maxwell_max"] / np.sqrt(2)

                    a = species.sput_spec['model_smyth_a']
                    v_M = species.sput_spec['model_smyth_v_M']
                    v_b = species.sput_spec['model_smyth_v_b']
                    x_vals = np.linspace(0, np.sqrt(v_M**2 - v_b**2), 10000)


                    upper_bound = np.sqrt((v_M / v_b) ** 2 - 1)
                    normalization = 1 / (sput_func_int(upper_bound, a, v_b, v_M) - sput_func_int(0, a, v_b, v_M))
                    f_pdf = normalization * sput_func(x_vals, a, v_b, v_M)

                    ax_species = axs[k] if Params.num_species > 1 else axs
                    ax_species.hist(speeds[sim_instance.N_active:], 100, range=(0, np.sqrt(v_M**2 - v_b**2)), density=True)
                    ax_species.plot(x_vals, f_pdf, c='r', label='Smyth')
                    ax_species.plot(x_vals, maxwell_func(x_vals), c="b", label="Maxwell")

                    ax_species.set_title(f"{species_names[k]}", c='k', size='x-large')
                    #ax_species.tick_params(axis='both', labelsize=8)

                    ax_species.set_ylabel("Probability", fontsize='x-large')
                    ax_species.set_xlabel("Velocity in m/s", fontsize='x-large')

                    ax_species.legend(loc = 'upper right')

                fig.tight_layout()
                if save:
                    frame_identifier = f"Velocity_Distribution_{i}_{orbit_phase}"
                    plt.savefig(f'output/{path}/plots/{frame_identifier}.png')
                if showvel:
                    plt.show()
                plt.close()
            else:
                pass


        if save_particles:
            try:
                with open(f"output/{path}/particle_pos.txt", "w") as text_file:
                    np.savetxt(text_file, particle_positions)
                with open(f"output/{path}/particle_vel.txt", "w") as text_file:
                    np.savetxt(text_file, particle_velocities)
            except:
                print("Cannot save particles. Path may not exist")
                pass


        top_down_column(bins=100)
        # mayavi_3D_density()
        los_column_and_velocity_dist(bins=100)
        #toroidal_hist()
        #vel_dist()


        print(f"Time needed for processing: {time.time() - start_time}")
    else:
        pass

