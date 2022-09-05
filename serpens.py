import rebound
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
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
savefig = False
showfig = True
plot_freq = 1  # Plot at each *plot_freq* advance

showhist = True
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
    imageio.mimsave('movie.gif', images, fps=4)


if __name__ == "__main__":

    # dat_final = np.loadtxt("particles.txt", skiprows=1, usecols=(0,1))
    # xdat_final = dat[:,0][3:]
    # ydat_final = dat[:,1][3:]

    sa = rebound.SimulationArchive("archive.bin", process_warnings=False)
    for i, sim_instance in enumerate(sa):
        ps = sim_instance.particles

        try:
            sim_instance.particles["moon"]
        except rebound.ParticleNotFound:
            moon_exists = False
        else:
            moon_exists = True

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
        rdata = np.zeros((sim_instance.N - sim_instance.N_active, Params.num_species))  # Distance from primary
        for k in range(sim_instance.N_active, sim_instance.N):
            ps_species = hashes_and_species[:,1][np.where(ps[k].hash.value == hashes_and_species[:,0])][0]

            xdata[int(k - sim_instance.N_active)][int(ps_species-1)] = ps[k].x
            ydata[int(k - sim_instance.N_active)][int(ps_species-1)] = ps[k].y

            if moon_exists:
                rdata[int(k - sim_instance.N_active)][int(ps_species-1)] = (np.sqrt((ps[k].x - ps["planet"].x) ** 2 + (ps[k].y - ps["planet"].y) ** 2)) / ps["planet"].r
            else:
                rdata[int(k - sim_instance.N_active)][int(ps_species-1)] = (np.sqrt((ps[k].x - ps[0].x) ** 2 + (ps[k].y - ps[0].y) ** 2)) / ps[0].r

        if i % plot_freq == 0:

            subplot_rows = int(np.ceil(Params.num_species/3))
            subplot_columns = Params.num_species if Params.num_species <= 3 else 3
            fig, axs = plt.subplots(subplot_rows, subplot_columns, figsize=(15, 8))
            for k in range(Params.num_species):
                H, xedges, yedges = getHistogram(sim_instance, xdata[:, k][xdata[:, k] != 0], ydata[:, k][ydata[:, k] != 0], bins=160)
                plotting(fig, axs[k], sim_instance, save=savefig, show=showfig, iter=i, histogram=H, xedges=xedges, yedges=yedges,
                         density=True)
                if not species_occurences.size == 0:
                    axs[k].set_title(f"{species_names[k]} \n Number of Particles: {species_occurences[k]}", y=1.08, c='k', size='x-large')
            if moon_exists:
                fig.suptitle(f"Particle Simulation around Planetary Body \n Number of particles {sim_instance.N}", size='xx-large', y=.95)
            else:
                fig.suptitle(f"Particle Simulation around Stellar Body \n Number of particles {sim_instance.N}", size='xx-large', y=.95)
            plt.tight_layout()
            plt.show()

            if i == 0 or not showhist:
                continue

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

            #plt.hist(bin_edges[:-1], bin_edges, weights=weights_phys)
            #plt.plot(bincenters[weights != 0], weights_phys[weights != 0], '-', c='black', alpha = 0.5)
            ax.set_yscale("log")
            ax.set_xlabel("Distance from primary in primary radii")
            ax.set_ylabel("Number of particles per cm")
            plt.legend()
            plt.grid(True)
            plt.show()
