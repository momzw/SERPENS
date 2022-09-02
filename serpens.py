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
    Params = Parameters()
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
    imageio.mimsave('movie.gif', images, fps=3)


# dat_final = np.loadtxt("particles.txt", skiprows=1, usecols=(0,1))
# xdat_final = dat[:,0][3:]
# ydat_final = dat[:,1][3:]

sa = rebound.SimulationArchive("archive.bin")
for i, sim_instance in enumerate(sa):
    ps = sim_instance.particles

    try:
        sim_instance.particles["moon"]
    except rebound.ParticleNotFound:
        moon_exists = False
    else:
        moon_exists = True

    xdata = []
    ydata = []
    rdata = []  # Distance from primary
    rdata_per_cm = []
    for k in range(sim_instance.N_active, sim_instance.N):
        xdata.append(ps[k].x)
        ydata.append(ps[k].y)
        if moon_exists:
            rdata.append((np.sqrt((ps[k].x - ps["planet"].x) ** 2 + (ps[k].y - ps["planet"].y) ** 2)) / ps["planet"].r)
        else:
            rdata.append((np.sqrt((ps[k].x - ps[0].x) ** 2 + (ps[k].y - ps[0].y) ** 2)) / ps[0].r)

    # rdata = np.sqrt((xdata - ps["planet"].x)**2 + (ydat - ps["planet"].y)**2) / ps["planet"].r
    H, xedges, yedges = getHistogram(sim_instance, xdata, ydata, bins=160)

    if i % plot_freq == 0:
        plotting(sim_instance, save=savefig, show=showfig, iter=i, histogram=H, xedges=xedges, yedges=yedges,
                 density=True)

        if i == 0 or not showhist:
            continue

        log = True if rdata else False  # Not needed if first sim_instance is already with particles.

        counts, bin_edges = np.histogram(rdata, 100, range=(0, 50))
        bin_width = bin_edges[1] - bin_edges[0]
        weights = counts / (bin_width * ps[0].r) / 100
        bincenters = (bin_edges[1:] + bin_edges[:-1]) / 2

        plt.hist(bin_edges[:-1], bin_edges, weights=weights)
        plt.plot(bincenters[weights != 0], weights[weights != 0], '-', c='black')
        plt.yscale("log")
        plt.xlabel("Distance from primary in primary radii")
        plt.ylabel("Number of particles per cm")
        plt.grid(True)
        plt.show()
