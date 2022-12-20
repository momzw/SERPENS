import rebound
import matplotlib
import matplotlib.style as mplstyle
import time
import numpy as np
import os as os
import shutil
import pickle
from datetime import datetime
from visualize import Visualize
import DTFE
import DTFE3D
from init import Species
from scheduler import NewParams
matplotlib.use('TkAgg')
mplstyle.use('fast')

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

# Plotting
# ---------------------

save = False
save_archive = False
save_particles = False
plot_freq = 400  # Plot at each *plot_freq* advance

showfig = True
showhist = False
showvel = False
show_column_density = False
skip_first = True

"""
    ===============================================================================================================
"""


if __name__ == "__main__":

    # PICKLE:
    # ===================
    # print("WARNING: SERPENS is about to unpickle particle data. Pickle files are not secure. Make sure you trust the source!")
    # input("\t Press Enter to continue...")

    with open('hash_library.pickle', 'rb') as handle:
        hash_supdict = pickle.load(handle)

    with open('Parameters.pickle', 'rb') as handle:
        Params = pickle.load(handle)

    v = NewParams(species=[Species('Na', description='Na--125285kg/s--2.5-30km/s', n_th=0, n_sp=1000, mass_per_sec=125285, lifetime=4 * 60, model_smyth_v_b=2500, model_smyth_v_M=30000)], moon=True, objects={'moon': {'m': 8.8e22, 'r': 1820e3}}, celest_set=2)
    v()
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
            shutil.copy2(f'{os.getcwd()}/Parameters.txt', f'output/{path}/Parameters.txt')
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
    # with open("hash_library.json", 'rb') as f:
    #    hash_supdict = json.load(f)

    # Use for LARGE FILES:
    # ===================
    # print("Collecting hash library:")
    # hash_supdict = {}
    # hash_dict = {}
    # flag = 0
    # index = plot_freq
    # with open("hash_library.json", 'rb') as f:
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
    # print("\t ...done")

    for i, sim_instance in enumerate(sa):

        if not i % plot_freq == 0:
            continue
        elif skip_first and i == 0:
            continue

        start_time = time.time()

        ps = sim_instance.particles

        species_names = []
        for ns in range(Params.num_species):
            species = Params.get_species(num=ns + 1)
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
                particle_time = (i - particle_iter) * Params.int_spec["sim_advance"] * sim_instance.particles[
                    "moon"].calculate_orbit(primary=sim_instance.particles["planet"]).P
            else:
                particle_time = (i - particle_iter) * Params.int_spec["sim_advance"] * sim_instance.particles[
                    "planet"].P

            if Params.get_species(id=particle_species[k1]) is None:
                continue
            chem_network = Params.get_species(id=particle_species[k1]).network
            reaction_rate = 0
            if not isinstance(chem_network, (int, float)):
                for l in range(np.size(chem_network[:, 0])):
                    reaction_rate += 1 / float(chem_network[:, 0][l])
            else:
                reaction_rate = 1 / chem_network
            particle_weights[k1] = np.exp(-particle_time * reaction_rate)

            # particle_weights[k1] = hash_dict_current[str(particle_hashes[k1])]["weight"]


        def top_down_column():
            # IMSHOW TOP DOWN COLUMN DENSITY PLOTS
            # ====================================

            vis = Visualize(sim_instance, lim=10)

            for k in range(Params.num_species):
                species = Params.get_species(num=k + 1)

                xdata = particle_positions[:, 0][np.where(particle_species == species.id)]
                ydata = particle_positions[:, 1][np.where(particle_species == species.id)]
                zdata = particle_positions[:, 2][np.where(particle_species == species.id)]

                weights = particle_weights[np.where(particle_species == species.id)]

                points = np.vstack([xdata, ydata, zdata]).T
                vx = particle_velocities[:, 0][np.where(particle_species == species.id)]
                vy = particle_velocities[:, 1][np.where(particle_species == species.id)]
                vz = particle_velocities[:, 2][np.where(particle_species == species.id)]
                velocities = np.vstack([vx, vy, vz]).T

                # Filter points:
                indices = np.unique([tuple(row) for row in points], axis=0, return_index=True)[1]
                weights = weights[np.sort(indices)]
                points = points[np.sort(indices)]
                velocities = velocities[np.sort(indices)]

                if moon_exists:
                    simulation_time = i * Params.int_spec["sim_advance"] * sim_instance.particles[
                        "moon"].calculate_orbit(primary=sim_instance.particles["planet"]).P
                    boundary = Params.int_spec["r_max"] * sim_instance.particles["moon"].calculate_orbit(
                        primary=sim_instance.particles["planet"]).a
                    X, Y = np.meshgrid(np.linspace(-boundary + sim_instance.particles["planet"].x,
                                                   boundary + sim_instance.particles["planet"].x, 100),
                                       np.linspace(-boundary + sim_instance.particles["planet"].y,
                                                   boundary + sim_instance.particles["planet"].y, 100))
                else:
                    simulation_time = i * Params.int_spec["sim_advance"] * sim_instance.particles["planet"].P
                    boundary = Params.int_spec["r_max"] * sim_instance.particles["planet"].a
                    X, Y = np.meshgrid(np.linspace(-boundary, boundary, 100),
                                       np.linspace(-boundary, boundary, 100))

                total_injected = i * (species.n_sp + species.n_th)
                remaining_part = len(xdata)
                mass_in_system = remaining_part / total_injected * species.mass_per_sec * simulation_time
                superpart_mass = mass_in_system / remaining_part
                number_per_superpart = superpart_mass / species.m
                phys_weights = number_per_superpart * weights

                print("Constructing DTFE ...")
                dtfe2d = DTFE.DTFE(points[:, :2], velocities[:, :2], phys_weights)
                dtfe3d = DTFE3D.DTFE(points, velocities, superpart_mass)
                dens3d = dtfe3d.density(points[:, 0], points[:, 1], points[:, 2]) / 1e6 * phys_weights
                dens2d = dtfe2d.density(points[:, 0], points[:, 1]) / 1e4
                print("\t ... done!")

                dens2dgrid = dtfe2d.density(X.flat, Y.flat).reshape((100, 100)) / 1e4

                #vis.add_colormesh(k, X, Y, dens2dgrid, contour=True, fill_contour=True, zorder=3, numlvls=25)
                # vis.add_triplot(k, points[:, 0], points[:, 1], dtfe3d.delaunay.simplices[:,:3], perspective="topdown")
                #vis.add_triplot(k, points[:, 0], points[:, 1], dtfe2d.delaunay.simplices, perspective="topdown", zorder=3)
                vis.add_densityscatter(k, points[:, 0], points[:, 1], dens3d, perspective="topdown", cb_format='%.2f', zorder=5, celest_colors=['y', 'sandybrown', 'b'])

                vis.set_title(r"Particle Densities $log_{10} (n[\mathrm{cm}^{-2}])$ around Planetary Body")

            if save:
                vis(show_bool=showfig, save_path=path, filename=f'{i}_td')
            else:
                vis(show_bool=showfig)
            del vis

        """
        def toroidal_hist():
            # RADIAL HISTOGRAM
            # ================
            if not i == 0:
                fig, ax = plt.subplots(figsize=(8, 8))
                for k in range(Params.num_species):
                    species = Params.get_species(num=k + 1)

                    psx = particle_positions[:, 0][np.where(particle_species == species.id)][sim_instance.N_active:]
                    psy = particle_positions[:, 1][np.where(particle_species == species.id)][sim_instance.N_active:]
                    psz = particle_positions[:, 2][np.where(particle_species == species.id)][sim_instance.N_active:]

                    psvx = particle_velocities[:, 0][np.where(particle_species == species.id)][sim_instance.N_active:]
                    psvy = particle_velocities[:, 0][np.where(particle_species == species.id)][sim_instance.N_active:]
                    psvz = particle_velocities[:, 0][np.where(particle_species == species.id)][sim_instance.N_active:]

                    if moon_exists:

                        rdata = np.sqrt(
                            (psx - ps["planet"].x) ** 2 + (psy - ps["planet"].y) ** 2 + (psz - ps["planet"].z) ** 2) / \
                                ps["planet"].r

                        counts, bin_edges = np.histogram(rdata, 100,
                                                         range=(0, 50))  # bins=int(np.sqrt(len(rdata[:,k])))

                        speeds = [np.linalg.norm(np.vstack((psvx, psvy, psvz)).T[j, :] - particle_velocities[2, :]) for
                                  j in range(np.size(psvx))]

                        v_ej = np.mean(speeds)
                        a_s = sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).a
                        v_orb = sim_instance.particles["moon"].calculate_orbit(
                            primary=sim_instance.particles["planet"]).v
                        V_tor = 2 * np.pi ** 2 * a_s ** 3 * (v_ej / v_orb) ** 2

                        weights = counts / V_tor / 1e6

                        mass_inject_per_advance = Params.get_species(num=k + 1).mass_per_sec * Params.int_spec[
                            "sim_advance"] * sim_instance.particles["moon"].calculate_orbit(
                            primary=sim_instance.particles["planet"]).P
                        weights_phys = weights * Params.get_species(num=k + 1).particles_per_superparticle(
                            mass_inject_per_advance)

                    else:

                        rdata = np.sqrt(psx ** 2 + psy ** 2 + psz ** 2) / ps["planet"].r

                        counts, bin_edges = np.histogram(rdata, 100, range=(0, 50))

                        speeds = [np.linalg.norm(np.vstack((psvx, psvy, psvz)).T[j, :] - particle_velocities[1, :]) for
                                  j in range(np.size(psvx))]

                        v_ej = np.mean(speeds)
                        a_s = sim_instance.particles["planet"].a
                        v_orb = sim_instance.particles["planet"].v
                        V_tor = 2 * np.pi ** 2 * a_s ** 3 * (v_ej / v_orb) ** 2

                        weights = counts / V_tor / 1e6

                        mass_inject_per_advance = Params.get_species(num=k + 1).mass_per_sec * Params.int_spec[
                            "sim_advance"] * sim_instance.particles["planet"].P
                        weights_phys = weights * Params.get_species(num=k + 1).particles_per_superparticle(
                            mass_inject_per_advance)

                    bincenters = (bin_edges[1:] + bin_edges[:-1]) / 2

                    ax.plot(bincenters[weights != 0], weights_phys[weights != 0], '-',
                            label=f"{Params.get_species(num=k + 1).description}", alpha=1)
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
        """

        def los_column_and_velocity_dist():
            # COLUMN DENSITY & VELOCITY DISTRIBUTION
            # ======================================
            vis = Visualize(sim_instance)
            for k in range(Params.num_species):
                species = Params.get_species(num=k + 1)

                ydata = particle_positions[:, 1][np.where(particle_species == species.id)]
                zdata = particle_positions[:, 2][np.where(particle_species == species.id)]
                vy = particle_velocities[:, 1][np.where(particle_species == species.id)]
                vz = particle_velocities[:, 2][np.where(particle_species == species.id)]

                points = np.vstack([ydata, zdata]).T
                velocities = np.vstack([vy, vz]).T

                weights = particle_weights[np.where(particle_species == species.id)]

                if moon_exists:
                    simulation_time = i * Params.int_spec["sim_advance"] * sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).P
                    # H_Io = 100e3
                    # chapman = np.sqrt(2 * np.pi * sim_instance.particles["moon"].r / H_Io)
                else:
                    simulation_time = i * Params.int_spec["sim_advance"] * sim_instance.particles["planet"].P

                total_injected = i * (species.n_sp + species.n_th)
                remaining_part = len(ydata)
                mass_in_system = remaining_part / total_injected * species.mass_per_sec * simulation_time
                superpart_mass = mass_in_system / remaining_part
                number_per_superpart = superpart_mass / species.m
                phys_weights = number_per_superpart * weights

                dtfe2d = DTFE.DTFE(points, velocities, superpart_mass)
                dens_los = dtfe2d.density(points[:, 0], points[:, 1]) / 1e4 * phys_weights

                vis.add_densityscatter(k, points[:, 0], points[:, 1], dens_los, perspective="los", cb_format='%.2E')
                vis.add_triplot(k, points[:, 0], points[:, 1], dtfe2d.delaunay.simplices, perspective='los')

            if save:
                vis(show_bool=show_column_density, save_path=path, filename=f'{i}_los')
            else:
                vis(show_bool=show_column_density)
            del vis

        """
        def vel_dist():
            if not i == 0:

                subplot_rows = Params.num_species
                fig, axs = plt.subplots(subplot_rows, 1, figsize=(15, 10))
                fig.suptitle(r"Particle density in 1/cm$^2$", size='xx-large')

                def maxwell_func(x):
                    return np.sqrt(2 / np.pi) * (x / scale) ** 2 * np.exp(-(x / scale) ** 2 / 2) / scale

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
                    species = Params.get_species(num=k + 1)

                    vel = particle_velocities[np.where(particle_species == species.id)]

                    if moon_exists:
                        orbit_phase = np.around(sim_instance.particles["moon"].calculate_orbit(
                            primary=sim_instance.particles["planet"]).f * 180 / np.pi, 2)
                        speeds = [np.linalg.norm(vel[j, :] - particle_velocities[2, :]) for j in
                                  range(np.size(vel[:, 0]))]
                    else:
                        orbit_phase = np.around(sim_instance.particles["planet"].f * 180 / np.pi, 2)
                        speeds = [np.linalg.norm(vel[j, :] - particle_velocities[1, :]) for j in
                                  range(np.size(vel[:, 0]))]

                    scale = species.sput_spec["model_maxwell_max"] / np.sqrt(2)

                    a = species.sput_spec['model_smyth_a']
                    v_M = species.sput_spec['model_smyth_v_M']
                    v_b = species.sput_spec['model_smyth_v_b']
                    x_vals = np.linspace(0, np.sqrt(v_M ** 2 - v_b ** 2), 10000)

                    upper_bound = np.sqrt((v_M / v_b) ** 2 - 1)
                    normalization = 1 / (sput_func_int(upper_bound, a, v_b, v_M) - sput_func_int(0, a, v_b, v_M))
                    f_pdf = normalization * sput_func(x_vals, a, v_b, v_M)

                    ax_species = axs[k] if Params.num_species > 1 else axs
                    ax_species.hist(speeds[sim_instance.N_active:], 100, range=(0, np.sqrt(v_M ** 2 - v_b ** 2)),
                                    density=True)
                    ax_species.plot(x_vals, f_pdf, c='r', label='Smyth')
                    ax_species.plot(x_vals, maxwell_func(x_vals), c="b", label="Maxwell")

                    ax_species.set_title(f"{species_names[k]}", c='k', size='x-large')
                    # ax_species.tick_params(axis='both', labelsize=8)

                    ax_species.set_ylabel("Probability", fontsize='x-large')
                    ax_species.set_xlabel("Velocity in m/s", fontsize='x-large')

                    ax_species.legend(loc='upper right')

                fig.tight_layout()
                if save:
                    frame_identifier = f"Velocity_Distribution_{i}_{orbit_phase}"
                    plt.savefig(f'output/{path}/plots/{frame_identifier}.png')
                if showvel:
                    plt.show()
                plt.close()
            else:
                pass
        """

        if save_particles:
            try:
                with open(f"output/{path}/particle_pos.txt", "w") as text_file:
                    np.savetxt(text_file, particle_positions)
                with open(f"output/{path}/particle_vel.txt", "w") as text_file:
                    np.savetxt(text_file, particle_velocities)
            except:
                print("Cannot save particles. Path may not exist")
                pass

        top_down_column()
        # los_column_and_velocity_dist()

        print(f"Time needed for processing: {time.time() - start_time}")
    else:
        pass
