import rebound
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import numpy as np
import os as os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from plotting import plotting
from init import Parameters

matplotlib.use('TkAgg')

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
save = False
plot_freq = 1  # Plot at each *plot_freq* advance

showfig = True
showhist = False
show_column_density = True
"""
    ===============================================================================================================
"""


def getHistogram(sim, xdata, ydata, bins):
    """
    Calculates a 2d histogram essentially defining a density distribution.
    :param sim: rebound simulation object.
    :param xdata: array_like. x-positions of particles to be analyzed.
    :param ydata: array_like. y-positions of particles to be analyzed.
    :param bins: int or array_like or [int, int] or [array, array]. See documentation for np.histogram2d.
    :return: H: ndarray (shape(nx, ny)), xedges: ndarray (shape(nx+1,)), yedges: ndarray (shape(ny+1,)). 2d-Histogram and bin edges along x- and y-axis.
    """
    r_max = Params.int_spec["r_max"]

    try:
        moon_a = sim.particles["moon"].calculate_orbit(primary=sim.particles["planet"]).a
        moon_exists = True
    except rebound.ParticleNotFound:
        planet_a = sim.particles["planet"].calculate_orbit(primary=sim.particles[0]).a
        moon_exists = False

    if moon_exists:
        boundary = r_max * moon_a
        H, xedges, yedges = np.histogram2d(xdata, ydata, range=[
            [-boundary + sim.particles["planet"].x, boundary + sim.particles["planet"].x],
            [-boundary + sim.particles["planet"].y, boundary + sim.particles["planet"].y]], bins=bins)
    else:
        boundary = r_max * planet_a
        H, xedges, yedges = np.histogram2d(xdata, ydata, range=[
            [-boundary, boundary],
            [-boundary, boundary]], bins=bins)

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

    # dat_final = np.loadtxt("particles.txt", skiprows=1, usecols=(0,1))
    # xdat_final = dat[:,0][3:]
    # ydat_final = dat[:,1][3:]

    sa = rebound.SimulationArchive("archive.bin", process_warnings=False)

    sim1 = sa[0]
    try:
        sim1.particles["moon"]
    except rebound.ParticleNotFound:
        moon_exists = False
    else:
        moon_exists = True

    if save:
        if moon_exists:
            path = datetime.utcnow().strftime("%d%m%Y--%H-%M") + "_moonsource"
        else:
            path = datetime.utcnow().strftime("%d%m%Y--%H-%M") + "_planetsource"
        os.makedirs(f'output/{path}/plots')
        with open(f"output/{path}/Parameters.txt", "w") as text_file:
            text_file.write(f"{Params.__str__()}")

    for i, sim_instance in enumerate(sa):
        ps = sim_instance.particles

        hashes_and_species = np.zeros((1,2))    # np.zeros((sim_instance.N - sim_instance.N_active, 2)) !
        species_names = []
        for ns in range(Params.num_species):
            species = Params.get_species(ns + 1)
            species_names.append(species.element)
            identifiers = [f"{species.id}_{j}_{x}" for j in range(Params.int_spec["num_sim_advances"]) for x in range(species.n_th + species.n_sp)]
            hashes = [rebound.hash(x).value for x in identifiers]
            for particle in sim_instance.particles[sim_instance.N_active:]:
                if particle.hash.value in hashes:
                    hash_and_species = np.array([[particle.hash.value, species.id]])
                    hashes_and_species = np.concatenate((hashes_and_species, hash_and_species))
        hashes_and_species = np.delete(hashes_and_species, 0, 0)

        species_occurences = np.bincount(hashes_and_species[:,1].astype(int))[1:]

        xdata = np.zeros((sim_instance.N - sim_instance.N_active, Params.num_species))
        ydata = np.zeros((sim_instance.N - sim_instance.N_active, Params.num_species))
        zdata = np.zeros((sim_instance.N - sim_instance.N_active, Params.num_species))
        rdata = np.zeros((sim_instance.N - sim_instance.N_active, Params.num_species))  # Distance from primary
        for k in range(sim_instance.N_active, sim_instance.N):
            ps_species = hashes_and_species[:,1][np.where(ps[k].hash.value == hashes_and_species[:,0])][0]

            xdata[int(k - sim_instance.N_active)][int(ps_species-1)] = ps[k].x
            ydata[int(k - sim_instance.N_active)][int(ps_species-1)] = ps[k].y
            zdata[int(k - sim_instance.N_active)][int(ps_species - 1)] = ps[k].z

            if moon_exists:
                rdata[int(k - sim_instance.N_active)][int(ps_species-1)] = (np.sqrt((ps[k].x - ps["planet"].x) ** 2 + (ps[k].y - ps["planet"].y) ** 2)) / ps["planet"].r
            else:
                rdata[int(k - sim_instance.N_active)][int(ps_species-1)] = (np.sqrt((ps[k].x - ps[0].x) ** 2 + (ps[k].y - ps[0].y) ** 2)) / ps[0].r

        if i % plot_freq == 0:

            # IMSHOW TOP DOWN COLUMN DENSITY PLOTS
            subplot_rows = int(np.ceil(Params.num_species/3))
            subplot_columns = Params.num_species if Params.num_species <= 3 else 3
            fig, axs = plt.subplots(subplot_rows, subplot_columns, figsize=(15, 8))
            for k in range(Params.num_species):
                H, xedges, yedges = getHistogram(sim_instance, xdata[:, k][xdata[:, k] != 0], ydata[:, k][ydata[:, k] != 0], bins=160)
                plotting(fig, axs[k], sim_instance, save=save, show=showfig, iter=i, histogram=H, xedges=xedges, yedges=yedges,
                         density=True)
                if not species_occurences.size == 0:
                    axs[k].set_title(f"{species_names[k]} \n Number of superparticles: {species_occurences[k]}", y=1.08, c='k', size='x-large')
            if moon_exists:
                fig.suptitle(f"Particle Simulation around Planetary Body \n Number of superparticles {sim_instance.N}", size='xx-large', y=.95)
            else:
                fig.suptitle(f"Particle Simulation around Stellar Body \n Number of superparticles {sim_instance.N}", size='xx-large', y=.95)
            plt.tight_layout()
            if save:
                if moon_exists:
                    orbit_phase = np.around(sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).f * 180/np.pi, 2)
                else:
                    orbit_phase = np.around(sim_instance.particles["planet"].calculate_orbit(primary=sim_instance.particles[0]).f * 180/np.pi, 2)
                frame_identifier = f"ColumnDensity_TopDown_{orbit_phase}"
                plt.savefig(f'output/{path}/plots/{frame_identifier}.png')
            if showfig:
                plt.show()
            plt.close()


            # RADIAL HISTOGRAM
            if not i ==0:
                fig, ax = plt.subplots(figsize=(8, 8))
                for k in range(Params.num_species):
                    counts, bin_edges = np.histogram(rdata[:,k][rdata[:,k] != 0], 100 , range=(0, 50))  #bins=int(np.sqrt(len(rdata[:,k])))
                    bin_width = bin_edges[1] - bin_edges[0]
                    if moon_exists:
                        weights = counts / (bin_width * ps["planet"].r) / 100
                    else:
                        weights = counts / (bin_width * ps[0].r) / 100

                    weights_phys = weights * Params.get_species(k+1).particles_per_superparticle(1e3)

                    bincenters = (bin_edges[1:] + bin_edges[:-1]) / 2

                    ax.plot(bincenters[weights != 0], weights_phys[weights != 0], '-', label=f"{Params.get_species(k+1).element}", alpha=1)
                    ax.scatter(bincenters[weights != 0], weights_phys[weights != 0], marker='x')

                ax.set_yscale("log")
                ax.set_xlabel("Distance from primary in primary radii")
                ax.set_ylabel("Number of particles per cm")
                plt.legend()
                plt.grid(True)
                if save:
                    frame_identifier = f"Radial_FullOrbit_Linear_Density_{i}"
                    plt.savefig(f'output/{path}/plots/{frame_identifier}.png')
                if showhist:
                    plt.show()
                plt.close()


            # COLUMN DENSITY
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.suptitle(r"Particle density in 1/cm$^2$", size='xx-large', y=.95)
            if moon_exists:
                moon_a = sim_instance.particles["moon"].calculate_orbit(primary=sim_instance.particles["planet"]).a
                yboundryC = Params.int_spec["r_max"] * moon_a + sim_instance.particles["moon"].y
                zboundryC = 10 * sim_instance.particles["planet"].r

                moon_patch = plt.Circle((ps["moon"].y, ps["moon"].z), ps["moon"].r, fc='y', alpha=.7)
                planet_patch = plt.Circle((ps["planet"].y, ps["planet"].z), ps["planet"].r, fc='sandybrown')

                ax.add_patch(moon_patch)
                ax.add_patch(planet_patch)

                lim = 15 * ps["planet"].r
                ax.set_xlim([-lim + ps["planet"].y, lim + ps["planet"].y])
                ax.set_ylim([-lim + ps["planet"].z, lim + ps["planet"].z])

                ax.set_xlabel("y-distance in planetary radii")
                ax.set_ylabel("z-distance in planetary radii")

            else:
                planet_a = sim_instance.particles["planet"].a
                yboundryC = Params.int_spec["r_max"] * planet_a
                zboundryC = 10 * sim_instance.particles[0].r

                planet_patch = plt.Circle((ps["planet"].y, ps["planet"].z), ps["planet"].r, fc='sandybrown')
                star_patch = plt.Circle((ps[0].x, ps[0].y), ps[0].r, fc='y', zorder=4)

                ax.add_patch(planet_patch)
                ax.add_patch(star_patch)

                lim = 15 * ps["planet"].r
                ax.set_xlim([-lim + ps[0].y, lim + ps[0].y])
                ax.set_ylim([-lim + ps[0].z, lim + ps[0].z])

                ax.set_xlabel("y-distance in stellar radii")
                ax.set_ylabel("z-distance in stellar radii")

            bins = 100
            HC, yedgesC, zedgesC = np.histogram2d(ydata[:, 0][ydata[:, 0] != 0], zdata[:, 0][zdata[:, 0] != 0], range=[[-yboundryC, yboundryC],[-zboundryC, zboundryC]], bins=bins)
            HC = HC.T

            bin_size = (yedgesC[1] - yedgesC[0]) * (zedgesC[1] - zedgesC[0])

            weight = Params.species1.particles_per_superparticle(1e3) / bin_size / 10000

            ax.set_facecolor('k')

            norm = colors.LogNorm() if not np.max(H) == 0 else colors.Normalize(vmin=0, vmax=0)  # Not needed if first sim_instance is already with particles.
            cmap = matplotlib.cm.afmhot
            #cmap.set_bad('k', 1.)
            im = ax.imshow(HC*weight, interpolation='gaussian', origin='lower', extent=[yedgesC[0], yedgesC[-1], zedgesC[0], zedgesC[-1]], cmap=cmap, norm=norm)
            if not np.max(H) == 0:
                # Colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            if save:
                frame_identifier = f"ColumnDensity_YZplane_{orbit_phase}"
                plt.savefig(f'output/{path}/plots/{frame_identifier}.png')
            if show_column_density:
                plt.show()
            plt.close()
            plt.show()
















