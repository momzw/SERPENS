import rebound
import numpy as np
import random
from tqdm import tqdm
from create_particle import create_particle
from init import init3, Parameters


def run_simulation():
    """
    Runs a REBOUND simulation given the at the beginning defined setup.
    Simulation stati after each advance get appended to the "archive.bin" file. These can be loaded at any later point.
    NOTE: Any "archive.bin" file in the folder gets deleted and overwritten!

    Saves a "particles.txt" file with every particles' position and velocity components. File gets overwritten at each advance.
    :return:
    """
    sim = rebound.Simulation("archive.bin")

    Params = Parameters()
    num_species = Params.num_species

    try:
        moon_P = sim.particles["moon"].calculate_orbit(primary=sim.particles["planet"]).P
        moon_a = sim.particles["moon"].calculate_orbit(primary=sim.particles["planet"]).a
        moon_exists = True
    except rebound.ParticleNotFound:
        planet_P = sim.particles["planet"].P
        planet_a = sim.particles["planet"].a
        moon_exists = False

    for i in range(Params.int_spec["num_sim_advances"]):

        sim_N_before = sim.N

        for ns in range(num_species):
            species = Params.get_species(ns+1)

            if (species.n_th == 0 or None) and (species.n_sp == 0 or None):
                continue

            # Add particles of given species
            # ------------------------------
            if Params.int_spec["gen_max"] is None or i <= Params.int_spec["gen_max"]:
                if not (species.n_th == 0 or None):
                    for j1 in tqdm(range(species.n_th), desc=f"Adding {species.element} particles thermally"):
                        #p = create_particle("thermal", temp_midnight=90, temp_noon=130)
                        p = create_particle(species, "thermal")
                        identifier = f"{species.id}_{i}_{j1}"
                        p.hash = identifier
                        sim.add(p)
                if not (species.n_sp == 0 or None):
                    for j2 in tqdm(range(species.n_sp), desc=f"Adding {species.element} particles via sputtering"):
                        p = create_particle(species,"sputter")
                        identifier = f"{species.id}_{i}_{j2 + species.n_th}"
                        p.hash = identifier
                        sim.add(p)

            # Remove particles through loss function
            # --------------------------------------
            num_lost = 0
            for j in range(i):
                dt = sim.t - j * Params.int_spec["sim_advance"]
                identifiers = [f"{species.id}_{j}_{x}" for x in range(species.n_th + species.n_sp)]
                hashes = [rebound.hash(x).value for x in identifiers]
                for particle in sim.particles[sim.N_active:]:
                    if particle.hash.value in hashes:
                        tau = species.lifetime
                        prob_to_exist = np.exp(-dt / tau)
                        if random.random() > prob_to_exist:
                            sim.remove(hash=particle.hash)
                            num_lost += 1
            print(f"{num_lost} {species.element} particles lost.")


        # Remove particles beyond specified number of semi-major axes
        # -----------------------------------------------------------
        boundary = Params.int_spec["r_max"] * moon_a if moon_exists else Params.int_spec["r_max"] * planet_a
        N = sim.N
        k = sim.N_active
        while k < N:
            if np.linalg.norm(np.asarray(sim.particles[k].xyz) - np.asarray(sim.particles["planet"].xyz)) > boundary:
                sim.remove(k)
                N += -1
            else:
                k += 1

        # ADVANCE INTEGRATION
        # ===================
        print("------------------------------------------------")
        print("Starting advance {0} ... ".format(i + 1))
        # sim.integrate(sim.t + Io_P/4)
        advance = moon_P / sim.dt * Params.int_spec["sim_advance"] if moon_exists else planet_P / sim.dt * Params.int_spec["sim_advance"]
        sim.steps(int(advance))  # Only reliable with specific integrators that leave sim.dt constant (not the default one!)
        print("Advance done! ")
        print("Number of particles: {0}".format(sim.N))

        sim.simulationarchive_snapshot("archive.bin")

        print("------------------------------------------------")

        # SAVE PARTICLES
        # ==============
        particle_positions = np.zeros((sim.N, 3), dtype="float64")
        particle_velocities = np.zeros((sim.N, 3), dtype="float64")
        sim.serialize_particle_data(xyz=particle_positions, vxvyvz=particle_velocities)

        header = np.array(["x", "y", "z", "vx", "vy", "vz"])
        data = np.vstack((header, np.concatenate((particle_positions, particle_velocities), axis=1)))
        np.savetxt("particles.txt", data, delimiter="\t", fmt="%-20s")

        # Stop if steady state
        # --------------------
        if Params.int_spec["stop_at_steady_state"] and np.abs(sim_N_before - sim.N) < 0.001:
            print("Reached steady state!")
            break
    print("Simulation completed successfully!")
    return


if __name__ == "__main__":
    init3(moon=True)
    run_simulation()
