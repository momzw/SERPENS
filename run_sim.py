import rebound
import numpy as np
import random
from tqdm import tqdm
from create_particle import create_particle
from init import init3, Simulation_Parameters



def particle_lifetime():
    """
    Calculates a particle's lifetime.
    :return: tau: float.
    """
    tau = 1000 * 60 * 60
    return tau


def run_simulation():
    """
    Runs a REBOUND simulation given the at the beginning defined setup.
    Simulation stati after each advance get appended to the "archive.bin" file. These can be loaded at any later point.
    NOTE: Any "archive.bin" file in the folder gets deleted and overwritten!

    Saves a "particles.txt" file with every particles' position and velocity components. File gets overwritten at each advance.
    :return:
    """
    sim = rebound.Simulation("archive.bin")

    Io_P = sim.particles["moon"].calculate_orbit(primary=sim.particles["planet"]).P
    Io_a = sim.particles["moon"].calculate_orbit(primary=sim.particles["planet"]).a

    Params = Simulation_Parameters()
    int_Params = Params.int()
    gen_Params = Params.gen()

    #sim.simulationarchive_snapshot("archive.bin", deletefile=True)
    for i in range(int_Params["num_sim_advances"]):

        sim_N_before = sim.N

        # Add particles
        # -------------
        if int_Params["gen_max"] is None or i <= int_Params["gen_max"]:
            for j1 in tqdm(range(gen_Params["n_th"]), desc="Adding thermal particles"):
                #p = create_particle("thermal", temp_midnight=90, temp_noon=130)
                p = create_particle("thermal")
                identifier = f"{i}_{j1}"
                p.hash = identifier
                sim.add(p)

            for j2 in tqdm(range(gen_Params["n_sp"]), desc="Adding sputter particles"):
                p = create_particle("sputter")
                identifier = f"{i}_{j2 + gen_Params['n_th']}"
                p.hash = identifier
                sim.add(p)

        # Remove particles beyond specified number of Io semi-major axes
        # --------------------------------------------------------------
        N = sim.N
        k = 3
        while k < N:
            if np.linalg.norm(np.asarray(sim.particles[k].xyz) - np.asarray(sim.particles["planet"].xyz)) > gen_Params["r_max"] * Io_a:
                sim.remove(k)
                N += -1
            else:
                k += 1

        # Remove particles through loss function
        # --------------------------------------
        num_lost = 0
        for j in range(i):
            dt = sim.t - j * int_Params["sim_advance"]
            identifiers = [f"{j}_{x}" for x in range(gen_Params["n_th"] + gen_Params["n_sp"])]
            hashes = [rebound.hash(x).value for x in identifiers]
            for particle in sim.particles[3:]:
                if particle.hash.value in hashes:
                    tau = particle_lifetime()
                    prob_to_exist = np.exp(-dt / tau)
                    if random.random() > prob_to_exist:
                        sim.remove(hash=particle.hash)
                        num_lost += 1
        print(f"{num_lost} particles lost.")

        # ADVANCE INTEGRATION
        # ===================
        print("Starting advance {0} ... ".format(i + 1))
        # sim.integrate(sim.t + Io_P/4)
        advance = Io_P / sim.dt * int_Params["sim_advance"]
        sim.steps(int(advance))  # Only reliable with specific integrators that leave sim.dt constant (not the default one!)
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
        if int_Params["stop_at_steady_state"] and np.abs(sim_N_before - sim.N) < 0.001:
            print("Reached steady state!")
            break
    print("Simulation completed successfully!")
    return


if __name__ == "__main__":
    init3()
    run_simulation()
