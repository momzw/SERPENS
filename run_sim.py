import concurrent.futures

import rebound
import reboundx
import numpy as np
import random
import json
import warnings

import pymp
import multiprocess as mp
import multiprocessing
from queue import Empty

from tqdm import tqdm
from create_particle import create_particle
from init import init3, Parameters, Species

import time

def run_simulation():
    """
    Runs a REBOUND simulation given the at the beginning defined setup.
    Simulation stati after each advance get appended to the "archive.bin" file. These can be loaded at any later point.
    NOTE: Any "archive.bin" file in the folder gets deleted and overwritten!

    Saves a "particles.txt" file with every particles' position and velocity components. File gets overwritten at each advance.
    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = rebound.Simulation("archive.bin")
        #sim.automateSimulationArchive("archive.bin", walltime=120)

        sim.collision = "direct"  # Brute force collision search and scales as O(N^2). It checks for instantaneous overlaps between every particle pair.
        sim.collision_resolve = "merge"

        deep_copies = []
        for proc in range(mp.cpu_count()):
            dc = sim.copy()
            dc.collision = "direct"
            dc.collision_resolve = "merge"

            rebxdc = reboundx.Extras(dc)
            rf = rebxdc.load_force("radiation_forces")
            rebxdc.add_force(rf)
            rf.params["c"] = 3.e8

            deep_copies.append(dc)

    Params = Parameters()
    num_species = Params.num_species
    moon_exists = Params.int_spec["moon"]

    hash_supdict = {}
    hash_dict = {}

    if moon_exists:
        moon_P = sim.particles["moon"].calculate_orbit(primary=sim.particles["planet"]).P
        moon_a = sim.particles["moon"].calculate_orbit(primary=sim.particles["planet"]).a
    else:
        planet_P = sim.particles["planet"].P
        planet_a = sim.particles["planet"].a


    # REBOUNDX ADDITIONAL FORCES
    # ==========================
    rebx = reboundx.Extras(sim)
    rf = rebx.load_force("radiation_forces")
    rebx.add_force(rf)
    rf.params["c"] = 3.e8

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

            if Params.int_spec["gen_max"] is None or i < Params.int_spec["gen_max"]:
                if not (species.n_th == 0 or None):
                    for j1 in tqdm(range(species.n_th), desc=f"Adding {species.name} particles thermally"):
                        p = create_particle(species, "thermal")
                        identifier = f"{species.id}_{i}_{j1}"
                        p.hash = identifier
                        sim.add(p)

                        hash_dict[str(p.hash.value)] = {"identifier": identifier, "i": i, "id": species.id}

                if not (species.n_sp == 0 or None):

                    #def mp_addsput(num):
                    #    p = create_particle(species, "sputter")
                    #    #identifier = f"{species.id}_{i}_{num + species.n_th}"
                    #    #p.hash = identifier
                    #    #hash_dict[str(p.hash.value)] = {"identifier": identifier, "i": i, "id": species.id}
                    #    return p.xyz + p.vxyz
                    #
                    #with mp.Pool(multiprocessing.cpu_count() - 1) as p:
                    #    r = list(tqdm(p.imap(mp_addsput, range(species.n_sp)), total=species.n_sp))
                    #
                    #for index, coord in enumerate(r):
                    #    identifier = f"{species.id}_{i}_{index + species.n_th}"
                    #    sim.add(x=coord[0], y=coord[1], z=coord[2], vx=coord[3], vy=coord[4], vz=coord[5], hash=identifier)
                    #
                    #    #sim.particles[identifier].params["kappa"] = 1.0e-6 / species.mass_num
                    #    sim.particles[identifier].params["beta"] = species.beta
                    #
                    #    hash_dict[str(sim.particles[identifier].hash.value)] = {"identifier": identifier, "i": i, "id": species.id}


                    #pymp.config.nested = True
                    ##print(f"Adding {species.name}...")
                    #with pymp.Parallel(4) as p:
                    #   for j2 in p.range(species.n_sp):
                    #        p = create_particle(species, "sputter")
                    #        identifier = f"{species.id}_{i}_{j2 + species.n_th}"
                    #        p.hash = identifier
                    #        sim.add(p)
                    #        hash_dict[str(p.hash.value)] = {"identifier": identifier, "i": i, "id": species.id}
                    #
                    #        if Params.int_spec["random_walk"]:
                    #            sim.particles[identifier].params["kappa"] = 1.0e-6 / species.mass_num

                    for j2 in tqdm(range(species.n_sp), desc=f"Adding {species.name} particles via sputtering"):
                        p = create_particle(species, "sputter")
                        identifier = f"{species.id}_{i}_{j2 + species.n_th}"
                        p.hash = identifier
                        sim.add(p)
                        #sim.particles[identifier].params["kappa"] = 1.0e-6 / species.mass_num
                        sim.particles[identifier].params["beta"] = species.beta
                        hash_dict[str(p.hash.value)] = {"identifier": identifier, "i": i, "id": species.id}


        # LOSS FUNCTION & CHEMICAL NETWORK
        # ================================
        boundary = Params.int_spec["r_max"] * moon_a if moon_exists else Params.int_spec["r_max"] * planet_a
        num_lost = 0
        num_converted = 0
        rng = np.random.default_rng()

        # Go through all previous advances:
        for j in range(i):
            if moon_exists:
                dt = sim.t - j * Params.int_spec["sim_advance"] * moon_P
            else:
                dt = sim.t - j * Params.int_spec["sim_advance"] * planet_P

            # Check all particles
            toberemoved = []
            for particle in sim.particles[sim.N_active:]:

                particle_iter = hash_dict[f"{particle.hash.value}"]["i"]
                species_id = hash_dict[f"{particle.hash.value}"]["id"]
                species = Params.get_species_by_id(species_id)

                if species.duplicate is not None:
                    species_id = int(str(species_id)[0])
                    species = Params.get_species_by_id(species_id)

                # Take particles created in iteration j (corresponding to dt):
                if particle_iter == j:

                    # Remove if too far away:
                    if moon_exists:
                        particle_distance = np.linalg.norm(np.asarray(particle.xyz) - np.asarray(sim.particles["planet"].xyz))
                    else:
                        particle_distance = np.linalg.norm(np.asarray(particle.xyz) - np.asarray(sim.particles[0].xyz))
                    if particle_distance > boundary:
                        toberemoved.append(particle.hash)
                        #sim.remove(hash=particle.hash)
                        #del hash_dict[f"{particle.hash.value}"]
                        num_lost += 1
                        #print(f"Particle {particle.hash.value} lost")
                        continue


                    # Remove if chemical reaction happens:
                    chem_network = species.network     # tau (float), educts (str), products (str), velocities (float)
                    if not isinstance(chem_network, int):

                        rng.shuffle(chem_network)   # Mitigate ordering bias

                        # Go through all reactions/lifetimes
                        for l in range(np.size(chem_network[:,0])):
                            tau = float(chem_network[:,0][l])
                            prob_to_exist = np.exp(-dt / tau)
                            if random.random() > prob_to_exist:

                                # Check all products if they have been implemented.
                                for i2 in chem_network[:,2][l].split():

                                    # Convert species if a product has been implemented.
                                    if any([True for k, v in species.implementedSpecies.items() if k == i2]):

                                        to_species = Params.get_species_by_name(i2)

                                        if to_species == None:
                                            sim.remove(hash=hash_dict[f"{particle.hash.value}"]["identifier"])
                                            # del hash_dict[f"{particle.hash.value}"]
                                            num_lost += 1
                                            continue

                                        # Change particle velocity if velocity delta has been implemented:
                                        if chem_network.shape[1] == 4:
                                            if not float(chem_network[:,3][l]) == 0:
                                                delv = float(chem_network[:,3][l])
                                                orbit_vel_vec = sim.particles["moon"].vxyz if moon_exists else sim.particles["planet"].vxyz
                                                orbit_vel_vec_normalized = 1/np.linalg.norm(orbit_vel_vec) * np.asarray(orbit_vel_vec)
                                                particle.vxyz(delv * orbit_vel_vec_normalized)

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

                                        num_converted += 1

                                    else:
                                        sim.remove(hash=hash_dict[f"{particle.hash.value}"]["identifier"])
                                        #del hash_dict[f"{particle.hash.value}"]
                                        num_lost += 1
                                        break
                                break
                    else:
                        tau = chem_network
                        prob_to_exist = np.exp(-dt / tau)
                        if random.random() > prob_to_exist:
                            sim.remove(hash=hash_dict[f"{particle.hash.value}"]["identifier"])
                            #del hash_dict[f"{particle.hash.value}"]
                            num_lost += 1

            for r in range(len(toberemoved)):
                sim.remove(hash=toberemoved[r])
                del hash_dict[f"{toberemoved[r].value}"]

        print(f"{num_lost} particles lost.")
        print(f"{num_converted} particles were converted.")


        # SAVE HASH_DICT
        # ==============
        hash_supdict[str(i+1)] = hash_dict.copy()


        # ADVANCE INTEGRATION
        # ===================

        print("------------------------------------------------")
        print(f"Starting advance {i+1} ... ")
        #advance = moon_P / sim.dt * Params.int_spec["sim_advance"] if moon_exists else planet_P / sim.dt * Params.int_spec["sim_advance"]
        #sim.steps(int(advance))  # Only reliable with specific integrators that leave sim.dt constant (not the default one!)

        #sim.ri_whfast.recalculate_coordinates_this_timestep = 1

        start_time = time.time()

        advance = moon_P * Params.int_spec["sim_advance"] if moon_exists else planet_P * Params.int_spec["sim_advance"]
        sim.integrate(int(advance * (i+1)), exact_finish_time=0)

        print(f"Calculation time for advance: {time.time() - start_time}")

        print("Advance done! ")
        print("Number of particles: {0}".format(sim.N))

        #sim.integrator_synchronize()

        sim.simulationarchive_snapshot("archive.bin")

        print("------------------------------------------------")


        #print("------------------------------------------------")
        #print(f"Starting advance {i} ... ")
        #
        #start_time = time.time()
        #cpus = multiprocessing.cpu_count()
        #
        #def advance_sim(dc_index, q):
        #    adv = moon_P * Params.int_spec["sim_advance"] if moon_exists else planet_P * Params.int_spec["sim_advance"]
        #    dc = deep_copies[dc_index]
        #    print(f"{dc.particles[1].hash.value}: {dc.particles[1].xyz}")
        #    dc.integrate(int(adv*i))
        #    print(f"{dc.particles[1].hash.value}: {dc.particles[1].xyz}")
        #    q.put("Process done")
        #
        #lst = range(sim_N_before, sim.N)
        #split = np.array_split(lst, cpus)
        #processes = []
        #q = multiprocessing.Queue()
        #for proc in range(cpus):
        #    dc = deep_copies[proc]
        #    dc.add([sim.particles[i] for i in np.ndarray.tolist(split[proc])])
        #    p = multiprocessing.Process(target=advance_sim, args=(proc,q))
        #    p.start()
        #    processes.append(p)
        #
        #def yield_from_process(q, p):
        #    while p.is_alive():
        #        p.join(timeout=60)
        #        while True:
        #            try:
        #                yield q.get(block=False)
        #            except Empty:
        #                break
        #
        #for process in processes:
        #    yield_from_process(q, process)
        #
        #del sim.particles
        #
        #sim.add([deep_copies[0].particles[i] for i in range(deep_copies[0].N_active)])
        #sim.N_active = deep_copies[0].N_active
        #for proc in range(len(deep_copies)):
        #    dc = deep_copies[proc]
        #    sim.add([dc.particles[i] for i in range(dc.N_active, dc.N)])
        #
        #sim.simulationarchive_snapshot("archive.bin")
        #
        #print("Advance done! ")
        #print(f"Simulation runtime: {time.time() - start_time}")
        #print(f"Number of particles: {sim.N}")
        #print("------------------------------------------------")


        # SAVE HASH
        # ==============

        with open("hash_library.json", 'w') as f:
            json.dump(hash_supdict, f)

        # Stop if steady state
        # --------------------
        if Params.int_spec["stop_at_steady_state"] and np.abs(sim_N_before - sim.N) < 0.001:
            print("Reached steady state!")
            break

    print("Simulation completed successfully!")
    return


if __name__ == "__main__":
    Params = Parameters()
    init3(moon = Params.int_spec["moon"], additional_majors=False)
    run_simulation()
