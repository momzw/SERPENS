import rebound
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import random
from tqdm import tqdm
from plotting import plotting

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
sim.add(m=1.898e27, a=7.785e11, e=0.0489, inc=0.0227, primary=sim.particles[0], hash="jupiter")  # Omega=1.753, omega=4.78
sim.add(m=8.932e22, a=4.217e8, e=0.0041, inc=0.0386, primary=sim.particles[1], hash="io")
sim.N_active = 3
sim.move_to_com()  # Center of mass coordinate-system (Jacobi coordinates without this line)

sim.particles[1].r = 69911000
sim.particles[2].r = 1821600


# IMPORTANT:
# * This setting boosts WHFast's performance, but stops automatic synchronization and recalculation of Jacobi coordinates!
# * If particle masses are changed or massive particles' position/velocity are changed manually you need to include
#   sim.ri_whfast.recalculate_coordinates_this_timestep
# * Synchronization is needed if simulation gets manipulated or particle states get printed.
# Refer to https://rebound.readthedocs.io/en/latest/ipython_examples/AdvWHFast/
# => sim.ri_whfast.safe_mode = 0


Io_P = sim.particles[2].calculate_orbit(primary=sim.particles[1]).P
Io_a = sim.particles[2].calculate_orbit(primary=sim.particles[1]).a

"""
    PARAMETER SETUP
    _______________
    all units in SI
"""
# Integration specifics
# ---------------------
# NOTE: sim time step =/= sim advance => sim advance refers to number of sim time steps until integration is paused and actions are performed. !!!
sim_advance = Io_P / sim.dt / 12  # When simulation reaches multiples of this time step, new particles are generated and sim state gets plotted.
num_sim_advances = 6  # Number of times the simulation advances.
stop_at_steady_state = True
max_num_of_generation_advances = gen_max = None  # Define a maximum number of particle generation time steps. After this simulation advances without generating further particles.

# Generating particles
# ---------------------
num_thermal_per_advance = n_th = 0  # Number of particles created by thermal evap each integration advance.
num_sputter_per_advance = n_sp = 2000  # Number of particles created by sputtering each integration advance.
r_max =  1.8 * Io_a # Maximal radial distance. Particles beyond get removed from simulation.

# Thermal evaporation parameters
# ---------------------
Io_temp_max = 130
Io_temp_min = 90
spherical_symm_ejection = False
part_mass_in_amu = 23

# Sputtering model
# ---------------------
sput_model = "smyth"  # Valid inputs: maxwell, wurz, smyth.

# Sputtering model shape parameters
# ---------------------
model_maxwell_mean = 2500
model_maxwell_std = 300

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

# Plotting
# ---------------------
savefig = False
showfig = True
plot_freq = 2 # Plot at each *plot_freq* advance

"""
    =====================================
"""


def random_pos(lat_dist, long_dist, **kwargs):
    # Coordinates:
    # Inertial system Cartesian coordinates. x-axis points from star away, y in direction of orbit.
    valid_dist = {"truncnorm": 0, "uniform": 1}
    if lat_dist in valid_dist:
        if valid_dist[lat_dist] == 0:
            from scipy.stats import truncnorm
            lower = kwargs.get("a_lat", -np.pi / 2)
            upper = kwargs.get("b_lat", np.pi / 2)
            center = kwargs.get("loc_lat", 0)
            std = kwargs.get("std_lat", 1)
            a, b = (lower - center) / std, (upper - center) / std
            latitude = truncnorm.rvs(a, b, loc=center, size=1)[0]
        else:
            lower = kwargs.get("a_lat", -np.pi / 2)
            upper = kwargs.get("b_lat", np.pi / 2)
            latitude = np.random.default_rng().uniform(lower, upper)
    else:
        raise ValueError("Invalid latitude distribution encountered in positional calculation: " % lat_dist)

    if long_dist in valid_dist:
        if valid_dist[long_dist] == 0:
            from scipy.stats import truncnorm
            lower = kwargs.get("a_long", -np.pi)
            upper = kwargs.get("b_long", np.pi)
            center = kwargs.get("loc_long", 0)
            std = kwargs.get("std_long", 1)
            a, b = (lower - center) / std, (upper - center) / std
            longitude = truncnorm.rvs(a, b, loc=center, size=1)[0]
        else:
            lower = kwargs.get("a_long", -np.pi)
            upper = kwargs.get("b_long", np.pi)
            longitude = np.random.default_rng().uniform(lower, upper)
    else:
        raise ValueError("Invalid longitude distribution encountered in positional calculation: " % long_dist)

    # Spherical Coordinates. x towards Sun. Change y sign to preserve direction of rotation.
    # Regarding latitude: In spherical coordinates 0 = Northpole, pi = Southpole
    x = sim.particles[2].r * np.cos(longitude) * np.sin(np.pi / 2 - latitude)
    y = sim.particles[2].r * np.sin(longitude) * np.sin(np.pi / 2 - latitude)
    z = sim.particles[2].r * np.cos(np.pi / 2 - latitude)

    x = sim.particles[2].r * np.cos(longitude)  # 2D-TEST
    y = sim.particles[2].r * np.sin(longitude)  # 2D-TEST
    z = 0  # 2D-TEST

    pos = np.array([x, y, z])

    return pos, latitude, longitude


def random_temp(temp_min, temp_max, latitude, longitude):
    longitude_wrt_sun = longitude + np.arctan2(sim.particles[2].y, sim.particles[2].x)
    if not spherical_symm_ejection:
        # Coordinate system relevant. If x-axis away from star a longitude -np.pi / 2 < longitude_wrt_sun < np.pi / 2 points away from the star!
        # Need to change temperature-longitude dependence.
        # (refer Wurz, P., 2002, "Monte-Carlo simulation of Mercury's exosphere"; -np.pi / 2 < longitude_wrt_sun < np.pi / 2)
        if np.pi / 2 < longitude_wrt_sun < 3 * np.pi / 2:
            temp = temp_min + (temp_max - temp_min) * (np.abs(np.cos(longitude_wrt_sun)) * np.cos(latitude)) ** (
                    1 / 4)
        else:
            temp = temp_min
    else:
        temp = (temp_max + temp_min) / 2
    return temp


def random_vel_thermal(temp):
    from scipy.stats import maxwell, norm
    v1 = maxwell.rvs()
    v2 = norm.rvs(scale=0.1)  # Maxwellian only has positive values. For hemispheric coverage we need Gaussian (or other dist)
    v3 = norm.rvs(scale=0.1)  # Maxwellian only has positive values. For hemispheric coverage we need Gaussian (or other dist)
    v3 = 0  # 2D

    u = 1.660539e-27
    k_B = 1.380649e-23
    vel_Na = np.sqrt((k_B * temp) / (part_mass_in_amu * u)) * np.array([v1, v2, v3])

    return vel_Na


def random_vel_sputter(E_i, E_b):
    from scipy.stats import rv_continuous

    class _elevation_gen(rv_continuous):
        def _pdf(self, x):
            normalization = 4 / np.pi  # Inverse of integral of cos^2 from 0 to pi/2
            f_alpha = normalization * np.cos(x) ** 2
            return f_alpha

    elev_dist = _elevation_gen(a=0, b=np.pi / 2)
    ran_elev = elev_dist.rvs()
    ran_azi = np.random.default_rng().uniform(0, 2 * np.pi)

    v1 = np.cos(ran_azi) * np.sin(ran_elev)
    v2 = np.sin(ran_azi) * np.sin(ran_elev)
    v3 = np.cos(ran_elev)
    v1 = 0  # 2D-TEST

    # ___________________________________________________

    # MAXWELLIAN MODEL
    def model_maxwell():
        from scipy.stats import maxwell
        maxwell_ran_speed = maxwell.rvs(loc=model_maxwell_mean, scale=model_maxwell_std)
        ran_vel_sputter_maxwell = maxwell_ran_speed * np.array(
            [v3, v2, v1])  # Rotated, s.t. reference direction along x-axis.
        return ran_vel_sputter_maxwell

    # MODEL 1
    def model_wurz():
        class _sputter_gen(rv_continuous):
            def _pdf(self, x, E_inc, E_bin):
                normalization = (E_inc + E_bin) ** 2 / E_inc ** 2
                f_E = normalization * 2 * E_bin * x / ((x + E_bin) ** 3)
                return f_E

        energy_dist = _sputter_gen(a=0, b=E_i, shapes='E_inc, E_bin')
        ran_energy = energy_dist.rvs(E_i, E_b)
        u = 1.660539e-27
        m = model_wurz_ejected_mass_in_amu * u
        ran_speed = np.sqrt(2 * ran_energy / m)
        ran_vel_sputter_model1 = ran_speed * np.array([v3, v2, v1])  # Rotated, s.t. reference direction along x-axis.
        return ran_vel_sputter_model1

    # MODEL 2
    def model_smyth():
        class _sputter_gen2(rv_continuous):
            def _pdf(self, x, a, v_b, v_M):
                def model2func(x, a, v_b, v_M):
                    f_v = 1 / v_b * (x / v_b) ** 3 \
                          * (v_b ** 2 / (v_b ** 2 + x ** 2)) ** a \
                          * (1 - np.sqrt((x ** 2 + v_b ** 2) / v_M ** 2))
                    return f_v

                def model2func_int(x, a, v_b, v_M):
                    integral_bracket = 2 * (v_b ** 2 + x ** 2) ** (5 / 2) / ((2 * a - 5) * v_M) \
                                       - 2 * v_b ** 2 * (v_b ** 2 + x ** 2) ** (3 / 2) / ((2 * a - 3) * v_M) \
                                       - (v_b ** 2 + x ** 2) ** 2 / (a - 2) \
                                       + v_b ** 2 * (v_b ** 2 + x ** 2) / (a - 1)
                    f_v_integrated = 1 / 2 * v_b ** (2 * a - 4) * (v_b ** 2 + x ** 2) ** (-a) * integral_bracket
                    return f_v_integrated

                normalization = 1 / (model2func_int(v_M, a, v_b, v_M) - model2func_int(v_b, a, v_b, v_M))
                f_pdf = normalization * model2func(x, a, v_b, v_M)
                return f_pdf

        v_b = model_smyth_v_b
        v_M = model_smyth_v_M
        a = model_smyth_a
        model2_vel_dist = _sputter_gen2(a=v_b, b=v_M, shapes='a, v_b, v_M')
        model2_ran_speed = model2_vel_dist.rvs(a, v_b, v_M)

        ran_vel_sputter_model2 = model2_ran_speed * np.array(
            [v3, v2, v1])  # Rotated, s.t. reference direction along x-axis.
        return ran_vel_sputter_model2

    # ___________________________________________________

    valid_model = {"maxwell": 0, "wurz": 1, "smyth": 2}
    if sput_model in valid_model:
        if valid_model[sput_model] == 0:
            vel = model_maxwell()
        elif valid_model[sput_model] == 1:
            vel = model_wurz()
        else:
            vel = model_smyth()
    else:
        raise ValueError("Invalid sputtering model: " % sput_model)

    return vel


def create_particle(process, **kwargs):
    valid_process = {"thermal": 0, "sputter": 1}
    if process in valid_process:
        if valid_process[process] == 0:
            temp_min = kwargs.get("temp_midnight", 90)  # Default value corresponds to Io
            temp_max = kwargs.get("temp_noon", 130)  # Default value corresponds to Io
            ran_pos, ran_lat, ran_long = random_pos(lat_dist="truncnorm", long_dist="uniform", a_long=0,
                                                    b_long=2 * np.pi)
            ran_temp = random_temp(temp_min, temp_max, ran_lat, ran_long)
            ran_vel_not_rotated_in_place = random_vel_thermal(ran_temp)

        else:
            angle_correction = np.arctan2((sim.particles[2].y - sim.particles[1].y),
                                          (sim.particles[2].x - sim.particles[1].x))
            ran_pos, ran_lat, ran_long = random_pos(lat_dist="uniform", long_dist="truncnorm", a_lat=-np.pi / 2,
                                                    b_lat=np.pi / 2, a_long=-np.pi / 2 + angle_correction,
                                                    b_long=3 * np.pi / 2 + angle_correction,
                                                    loc_long=np.pi / 2 + angle_correction)
            # ran_pos, ran_lat, ran_long = random_pos(lat_dist="uniform", long_dist="uniform")
            u = 1.660539e-27
            E_inc_def = 1 / 2 * model_wurz_inc_mass_in_amu * u * (model_wurz_inc_part_speed ** 2)
            E_inc = kwargs.get("E_incoming", E_inc_def)
            E_bind = kwargs.get("E_bind",
                                model_wurz_binding_en)  # Default: Binding energy of Na, taken from table 1, in: Kudriavtsev, Y., et al. 2004, "Calculation of the surface binding energy for ion sputtered particles".

            ran_vel_not_rotated_in_place = random_vel_sputter(E_inc, E_bind)
    else:
        raise ValueError("Invalid escaping mechanism encountered in particle creation: " % process)

    # Rotation matrix in order to get velocity vector aligned with surface-normal.
    # Counterclockwise along z-axis (local longitude). Clockwise along y-axis (local latitude).
    rot_z = np.array([[np.cos(ran_long), -np.sin(ran_long), 0], [np.sin(ran_long), np.cos(ran_long), 0], [0, 0, 1]])
    rot_y = np.array([[np.cos(ran_lat), 0, -np.sin(ran_lat)], [0, 1, 0], [np.sin(ran_lat), 0, np.cos(ran_lat)]])
    rot = rot_y @ rot_z

    rot = rot_z  # 2D-TEST

    # NOTE: Escape velocity 2550 m/s at surface.
    ran_vel = rot @ ran_vel_not_rotated_in_place

    # Io position and velocity:
    Io_x, Io_y, Io_z = sim.particles[2].x, sim.particles[2].y, sim.particles[2].z
    Io_vx, Io_vy, Io_vz = sim.particles[2].vx, sim.particles[2].vy, sim.particles[2].vz

    p = rebound.Particle()
    p.x, p.y, p.z = ran_pos[0] + Io_x, ran_pos[1] + Io_y, ran_pos[2] + Io_z
    p.vx, p.vy, p.vz = ran_vel[0] + Io_vx, ran_vel[1] + Io_vy, ran_vel[2] + Io_vz

    return p


def getHistogram(sim, xdata, ydata, bins):
    ps = sim.particles
    H, xedges, yedges = np.histogram2d(xdata, ydata, range=[[ps[1].x - r_max, ps[1].x + r_max], [ps[1].y - r_max, ps[1].y + r_max]], bins=bins)
    H = H.T
    return H, xedges, yedges


def particle_lifetime():
    tau = 2 * 60 * 60
    return tau


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


def run_simulation():
    sim.simulationarchive_snapshot("archive.bin", deletefile=True)
    for i in range(num_sim_advances):

        sim_N_before = sim.N

        # Add particles
        # -------------
        if gen_max is None or i <= gen_max:
            for j1 in tqdm(range(n_th), desc="Adding thermal particles"):
                p = create_particle("thermal", temp_midnight=Io_temp_min, temp_noon=Io_temp_max)
                identifier = f"{i}_{j1}"
                p.hash = identifier
                sim.add(p)

            for j2 in tqdm(range(n_sp), desc="Adding sputter particles"):
                p = create_particle("sputter")
                identifier = f"{i}_{j2 + n_th}"
                p.hash = identifier
                sim.add(p)

        # Remove particles beyond specified number of Io semi-major axes
        # --------------------------------------------------------------
        N = sim.N
        k = 3
        while k < N:
            if np.linalg.norm(np.asarray(sim.particles[k].xyz) - np.asarray(sim.particles[1].xyz)) > r_max * Io_a:
                sim.remove(k)
                N += -1
            else:
                k += 1

        # Remove particles through loss function
        # --------------------------------------
        num_lost = 0
        for j in range(i):
            dt = sim.t - j * sim_advance
            identifiers = [f"{j}_{x}" for x in range(n_th + n_sp)]
            hashes = [rebound.hash(x).value for x in identifiers]
            for particle in sim.particles[3:]:
                if particle.hash.value in hashes:
                    tau = particle_lifetime()
                    prob_to_exist = np.exp(-dt/tau)
                    if random.random() > prob_to_exist:
                        sim.remove(hash=particle.hash)
                        num_lost += 1
        print(f"{num_lost} particles lost.")

        """
        # Get various particle data
        # -----------------
        ps = sim.particles
        xdata = []
        ydata = []
        rdata = []
        for k in range(3, sim.N):
            xdata.append(ps[k].x)
            ydata.append(ps[k].y)
            rdata.append((np.sqrt((ps[k].x - ps[1].x)**2 + (ps[k].y-ps[1].y)**2))/ps[1].r)
        H, xedges, yedges = getHistogram(sim, xdata, ydata, 160)
        
        # Plotting
        # --------
        if i % plot_freq == 0:  # Adjust '1' to plot every 'x' integration advance. Here: Plot at every advance.
            plotting(sim, save=savefig, show=showfig, iter=i, histogram=H, xedges=xedges, yedges=yedges)

            y, binEdges, patches = plt.hist(rdata, 100, log=True, range=(0,50))
            bincenters = (binEdges[1:] + binEdges[:-1]) / 2

            plt.plot(bincenters[y!=0], y[y!=0], '-', c='black')
            plt.grid(True)
            plt.show()
        """

        # ADVANCE INTEGRATION
        # ===================
        print("Starting advance {0} ... ".format(i+1))
        # sim.integrate(sim.t + Io_P/4)
        sim.steps(int(sim_advance))  # Only reliable with specific integrators that leave sim.dt constant (not the default one!)
        print("Advance done! ")
        print("Number of particles: {0}".format(sim.N))

        sim.simulationarchive_snapshot("archive.bin")

        print("------------------------------------------------")

        # SAVE PARTICLES
        # ==============
        particle_positions = np.zeros((sim.N, 3), dtype="float64")
        particle_velocities = np.zeros((sim.N, 3), dtype="float64")
        sim.serialize_particle_data(xyz=particle_positions)
        sim.serialize_particle_data(vxvyvz=particle_velocities)

        header = np.array(["x", "y", "z", "vx", "vy", "vz"])
        data = np.vstack((header, np.concatenate((particle_positions, particle_velocities), axis=1)))
        np.savetxt("particles.txt", data, delimiter="\t", fmt="%-20s")

        # Stop if steady state
        # --------------------
        if stop_at_steady_state and np.abs(sim_N_before - sim.N) < 0.001:
            print("Reached steady state!")
            sim.simulationarchive_snapshot("archive.bin")
            break
    print("Simulation completed successfully!")
    return

#run_simulation()

sa = rebound.SimulationArchive("archive.bin")
for i, sim_instance in enumerate(sa):
    ps = sim_instance.particles
    xdata = []
    ydata = []
    rdata = []
    for k in range(3, sim_instance.N):
        xdata.append(ps[k].x)
        ydata.append(ps[k].y)
        rdata.append((np.sqrt((ps[k].x - ps[1].x) ** 2 + (ps[k].y - ps[1].y) ** 2)) / ps[1].r)
    H, xedges, yedges = getHistogram(sim_instance, xdata, ydata, 160)

    if i % plot_freq == 0:
        plotting(sim_instance, save=savefig, show=showfig, iter=i, histogram=H, xedges=xedges, yedges=yedges)

        log = True if rdata else False  # Not needed if first sim_instance is already with particles.
        y, binEdges, patches = plt.hist(rdata, 100, log=log, range=(0, 50))
        bincenters = (binEdges[1:] + binEdges[:-1]) / 2

        plt.plot(bincenters[y != 0], y[y != 0], '-', c='black')
        plt.grid(True)
        plt.show()


















