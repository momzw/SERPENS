import rebound
import numpy as np
import random
from tqdm import tqdm
from create_particle import create_particle

sim = rebound.Simulation("archive.bin")

Io_P = sim.particles["moon"].calculate_orbit(primary=sim.particles["planet"]).P
Io_a = sim.particles["moon"].calculate_orbit(primary=sim.particles["planet"]).a


# Integration specifics
# ---------------------
# NOTE: sim time step =/= sim advance => sim advance refers to number of sim time steps until integration is paused and actions are performed. !!!
sim_advance = Io_P / sim.dt / 12  # When simulation reaches multiples of this time step, new particles are generated and sim state gets plotted.
num_sim_advances = 20  # Number of times the simulation advances.
stop_at_steady_state = True
max_num_of_generation_advances = gen_max = None  # Define a maximum number of particle generation time steps. After this simulation advances without generating further particles.

# Generating particles
# ---------------------
num_thermal_per_advance = n_th = 0  # Number of particles created by thermal evap each integration advance.
num_sputter_per_advance = n_sp = 2000  # Number of particles created by sputtering each integration advance.
r_max =  1.8 * Io_a # Maximal radial distance. Particles beyond get removed from simulation.



def particle_lifetime():
    """
    Calculates a particle's lifetime.
    :return: tau: float.
    """
    tau = 4 * 60 * 60
    return tau


def run_simulation():
    """
    Runs a REBOUND simulation given the at the beginning defined setup.
    Simulation stati after each advance get appended to the "archive.bin" file. These can be loaded at any later point.
    NOTE: Any "archive.bin" file in the folder gets deleted and overwritten!

    Saves a "particles.txt" file with every particles' position and velocity components. File gets overwritten at each advance.
    :return:
    """
    sim.simulationarchive_snapshot("archive.bin", deletefile=True)
    for i in range(num_sim_advances):

        sim_N_before = sim.N

        # Add particles
        # -------------
        if gen_max is None or i <= gen_max:
            for j1 in tqdm(range(n_th), desc="Adding thermal particles"):
                #p = create_particle("thermal", temp_midnight=90, temp_noon=130)
                p = create_particle("thermal")
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
            if np.linalg.norm(np.asarray(sim.particles[k].xyz) - np.asarray(sim.particles["planet"].xyz)) > r_max * Io_a:
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
                    prob_to_exist = np.exp(-dt / tau)
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
            rdata.append((np.sqrt((ps[k].x - ps["planet"].x)**2 + (ps[k].y-ps["planet"].y)**2))/ps["planet"].r)
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
        print("Starting advance {0} ... ".format(i + 1))
        # sim.integrate(sim.t + Io_P/4)
        sim.steps(
            int(sim_advance))  # Only reliable with specific integrators that leave sim.dt constant (not the default one!)
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