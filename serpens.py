import rebound
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
from tqdm import tqdm
from plotting import plotting
from init import init3, Simulation_Parameters

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


def init3():
    """
    This function initializes the basic REBOUND simulation structure and adds the first 3 major objects.
    These three objects are the host star, planet and moon.
    :return: rebound simulation object
    """
    sim = rebound.Simulation()
    # sim.automateSimulationArchive("archive.bin", walltime=60)
    sim.integrator = "whfast" # Fast and unbiased symplectic Wisdom-Holman integrator. Suitability not yet assessed.
    sim.collision = "direct" # Brute force collision search and scales as O(N^2). It checks for instantaneous overlaps between every particle pair.
    sim.collision_resolve = "merge"

    # SI units:
    # ---------
    sim.units = ('m', 's', 'kg')
    sim.G = 6.6743e-11

    # CGS units:
    # ----------
    # sim.G = 6.67430e-8

    sim.dt = 500

    # labels = ["Sun", "Jupiter", "Io"]
    # sim.add(labels)      # Note: Takes current position in the solar system. Therefore more useful to define objects manually in the following.
    sim.add(m=1.988e30, hash="sun")
    sim.add(m=1.898e27, a=7.785e11, e=0.0489, inc=0.0227, primary=sim.particles[0], hash="planet")  # Omega=1.753, omega=4.78
    sim.add(m=8.932e22, a=4.217e8, e=0.0041, inc=0.00087, primary=sim.particles["planet"], hash="moon")
    #sim.add(m=4.799e22, a=6.709e8, e=0.009, inc=0.0082, primary=sim.particles["planet"], hash="moon2")
    sim.N_active = 3
    sim.move_to_com()  # Center of mass coordinate-system (Jacobi coordinates without this line)

    sim.particles["planet"].r = 69911000
    sim.particles["moon"].r = 1821600


    # IMPORTANT:
    # * This setting boosts WHFast's performance, but stops automatic synchronization and recalculation of Jacobi coordinates!
    # * If particle masses are changed or massive particles' position/velocity are changed manually you need to include
    #   sim.ri_whfast.recalculate_coordinates_this_timestep
    # * Synchronization is needed if simulation gets manipulated or particle states get printed.
    # Refer to https://rebound.readthedocs.io/en/latest/ipython_examples/AdvWHFast/
    # => sim.ri_whfast.safe_mode = 0

    return sim

#sim = init3()

#Io_P = sim.particles["moon"].calculate_orbit(primary=sim.particles["planet"]).P
#Io_a = sim.particles["moon"].calculate_orbit(primary=sim.particles["planet"]).a

Params = Simulation_Parameters()
gen_Params = Params.gen()
r_max = gen_Params["r_max"]

"""
    PARAMETER SETUP
    _______________
    all units in SI
"""

"""
# Integration specifics
# ---------------------
# NOTE: sim time step =/= sim advance => sim advance refers to number of sim time steps until integration is paused and actions are performed. !!!
sim_advance = Io_P / sim.dt / 12  # When simulation reaches multiples of this time step, new particles are generated and sim state gets plotted.
num_sim_advances = 60  # Number of times the simulation advances.
stop_at_steady_state = True
max_num_of_generation_advances = gen_max = None  # Define a maximum number of particle generation time steps. After this simulation advances without generating further particles.

# Generating particles
# ---------------------
num_thermal_per_advance = n_th = 0  # Number of particles created by thermal evap each integration advance.
num_sputter_per_advance = n_sp = 5000  # Number of particles created by sputtering each integration advance.
r_max = 1.8 * Io_a # Maximal radial distance. Particles beyond get removed from simulation.

# Thermal evaporation parameters
# ---------------------
Io_temp_max = 130
Io_temp_min = 90
spherical_symm_ejection = False
part_mass_in_amu = 23

# Sputtering model
# ---------------------
sput_model = "maxwell"  # Valid inputs: maxwell, wurz, smyth.

# Sputtering model shape parameters
# ---------------------
model_maxwell_mean = 3500
model_maxwell_std = 100

model_wurz_inc_part_speed = 5000
model_wurz_binding_en = 2.89 * 1.602e-19  # See table 1, in: Kudriavtsev Y., et al. 2004, "Calculation of the surface binding energy for ion sputtered particles".
model_wurz_inc_mass_in_amu = 23
model_wurz_ejected_mass_in_amu = 23

model_smyth_v_b = 1000       # "low cutoff" speed to prevent the slowest nonescaping atoms from dominating the distribution (see Wilson et al. 2002)
model_smyth_v_M = 40000     # Maximum velocity achievable. Proportional to plasma velocity (see Wilson et al. 2002)
model_smyth_a = 7 / 3       # Speed distribution shape parameter

# Particle emission position
# ---------------------
# Longitude and latitude distributions may be changed inside the 'create_particle' function.

"""
# Plotting
# ---------------------
savefig = False
showfig = True
plot_freq = 1 # Plot at each *plot_freq* advance

"""
    =====================================
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
    ps = sim.particles
    Io_a = sim.particles["moon"].calculate_orbit(primary=sim.particles["planet"]).a
    H, xedges, yedges = np.histogram2d(xdata, ydata, range=[[ps["planet"].x - r_max*Io_a, ps["planet"].x + r_max*Io_a], [ps["planet"].y - r_max*Io_a, ps["planet"].y + r_max*Io_a]], bins=bins)
    H = H.T
    return H, xedges, yedges


def pngToGif(max_PNG_index, step):
    """
    TOOL TO COMBINE PNG TO GIF
    """
    import imageio
    filenames = [f'plots/sim_{i}.png' for i in range(0,max_PNG_index,step)]
    #with imageio.get_writer('mygif.gif', mode='I') as writer:
    #    for filename in filenames:
    #        image = imageio.imread(filename)
    #        writer.append_data(image)

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('movie.gif', images, fps=1)


sa = rebound.SimulationArchive("archive.bin")
for i, sim_instance in enumerate(sa):
    ps = sim_instance.particles
    xdata = []
    ydata = []
    rdata = []
    for k in range(3, sim_instance.N):
        xdata.append(ps[k].x)
        ydata.append(ps[k].y)
        rdata.append((np.sqrt((ps[k].x - ps["planet"].x) ** 2 + (ps[k].y - ps["planet"].y) ** 2)) / ps["planet"].r)
    H, xedges, yedges = getHistogram(sim_instance, xdata, ydata, 160)

    if i % plot_freq == 0:
        plotting(sim_instance, save=savefig, show=showfig, iter=i, histogram=H, xedges=xedges, yedges=yedges)

        log = True if rdata else False  # Not needed if first sim_instance is already with particles.
        y, binEdges, patches = plt.hist(rdata, 100, log=log, range=(0, 50))
        bincenters = (binEdges[1:] + binEdges[:-1]) / 2

        plt.plot(bincenters[y != 0], y[y != 0], '-', c='black')
        plt.grid(True)
        plt.show()


















