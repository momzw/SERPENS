import rebound
import numpy as np
import random
from tqdm import tqdm
from create_particle import create_particle
from init import init3, Parameters, Species


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
    moon_exists = Params.int_spec["moon"]

    hash_dict = {}

    if moon_exists:
        moon_P = sim.particles["moon"].calculate_orbit(primary=sim.particles["planet"]).P
        moon_a = sim.particles["moon"].calculate_orbit(primary=sim.particles["planet"]).a
    else:
        planet_P = sim.particles["planet"].P
        planet_a = sim.particles["planet"].a

    for i in range(Params.int_spec["num_sim_advances"]):

        sim_N_before = sim.N

        # CREATE PARTICLES
        # ================
        for ns in range(num_species):
            species = Params.get_species(ns+1)

            if (species.n_th == 0 or None) and (species.n_sp == 0 or None):
                continue

            # Add particles of given species
            # ------------------------------
            if Params.int_spec["gen_max"] is None or i <= Params.int_spec["gen_max"]:
                if not (species.n_th == 0 or None):
                    for j1 in tqdm(range(species.n_th), desc=f"Adding {species.name} particles thermally"):
                        p = create_particle(species, "thermal")
                        identifier = f"{species.id}_{i}_{j1}"
                        p.hash = identifier
                        sim.add(p)

                        hash_dict[str(p.hash.value)] = {"identifier": identifier, "i": i, "id": species.id}

                if not (species.n_sp == 0 or None):
                    for j2 in tqdm(range(species.n_sp), desc=f"Adding {species.name} particles via sputtering"):
                        p = create_particle(species, "sputter")
                        identifier = f"{species.id}_{i}_{j2 + species.n_th}"
                        p.hash = identifier
                        sim.add(p)

                        hash_dict[str(p.hash.value)] = {"identifier": identifier, "i": i, "id": species.id}

        # LOSS FUNCTION
        # =============
        boundary = Params.int_spec["r_max"] * moon_a if moon_exists else Params.int_spec["r_max"] * planet_a
        num_lost = 0
        rng = np.random.default_rng()

        # Go through all previous advances:
        for j in range(i):
            if moon_exists:
                dt = sim.t - j * Params.int_spec["sim_advance"] * moon_P
            else:
                dt = sim.t - j * Params.int_spec["sim_advance"] * planet_P

            # Check all particles
            for particle in sim.particles[sim.N_active:]:

                particle_iter = hash_dict[f"{particle.hash.value}"]["i"]
                species_id = hash_dict[f"{particle.hash.value}"]["id"]
                species = Params.get_species_by_id(species_id)

                # Take particles created in iteration j (corresponding to dt):
                if particle_iter == j:

                    # Remove if too far away:
                    if moon_exists:
                        particle_distance = np.linalg.norm(np.asarray(particle.xyz) - np.asarray(sim.particles["planet"].xyz))
                    else:
                        particle_distance = np.linalg.norm(np.asarray(particle.xyz) - np.asarray(sim.particles[0].xyz))
                    if particle_distance > boundary:
                        sim.remove(hash=particle.hash)
                        continue

                    # Remove if chemical reaction happens:
                    chem_network = species.network()     # tau (float), educts (str), products (str)
                    if not isinstance(chem_network, int):

                        rng.shuffle(chem_network)   # Mitigate ordering bias

                        # Go through all reactions/lifetimes
                        for i in range(np.size(chem_network[:,0])):
                            tau = float(chem_network[:,0][i])
                            prob_to_exist = np.exp(-dt / tau)
                            if random.random() > prob_to_exist:

                                # Check all products if they have been implemented.
                                for i2 in chem_network[:,2][i].split():

                                    # Convert species if a product has been implemented.
                                    if any([True for k, v in species.implementedSpecies.items() if k == i2]):

                                        to_species = Params.get_species_by_name(i2)

                                        # Take all species ids that are in iteration j:
                                        temp = "id"
                                        ids = [val[temp] for key, val in hash_dict.items() if temp in val and val["i"] == j]

                                        # Count number of product-species particles:
                                        to_species_total = np.count_nonzero(np.asarray(ids) == to_species.id)

                                        # Change particle hash
                                        new_hash = f"{to_species.id}_{j}_{to_species_total+1}"
                                        sim.particles[particle.hash].hash = new_hash

                                        # Update library
                                        hash_dict[f"{particle.hash.value}"] = {"identifier": new_hash, "i": j, "id": to_species.id}

                                    else:
                                        sim.remove(hash=particle.hash)
                                num_lost += 1
                                break
                    else:
                        tau = chem_network
                        prob_to_exist = np.exp(-dt / tau)
                        if random.random() > prob_to_exist:
                            sim.remove(hash=particle.hash)
                            num_lost += 1

        print(f"{num_lost} {species.name} particles lost or transformed.")

        # Remove particles beyond specified number of semi-major axes
        # -----------------------------------------------------------
        #boundary = Params.int_spec["r_max"] * moon_a if moon_exists else Params.int_spec["r_max"] * planet_a
        #N = sim.N
        #k = sim.N_active
        #while k < N:
        #    if np.linalg.norm(np.asarray(sim.particles[k].xyz) - np.asarray(sim.particles["planet"].xyz)) > boundary:
        #        sim.remove(k)
        #        N += -1
        #    else:
        #        k += 1


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
    Params = Parameters()
    init3(moon = Params.int_spec["moon"])
    run_simulation()
