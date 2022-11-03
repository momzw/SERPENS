import rebound
import reboundx
import numpy as np
import random
import json
import warnings
import os

import multiprocess as mp
import multiprocessing

from tqdm import tqdm

from create_particle import create_particle
from init import init3, Parameters, Species

import time

def set_pointers(sim_copy):
    sim_copy.collision = "direct"   # Brute force collision search and scales as O(N^2). It checks for instantaneous overlaps between every particle pair.
    sim_copy.collision_resolve = "merge"

    # REBOUNDX ADDITIONAL FORCES
    # ==========================
    rebxdc = reboundx.Extras(sim_copy)
    rf = rebxdc.load_force("radiation_forces")
    rebxdc.add_force(rf)
    rf.params["c"] = 3.e8

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
        set_pointers(sim)

        deep_copies = []
        for _ in range(mp.cpu_count()):
            dc = sim.copy()
            set_pointers(dc)
            deep_copies.append(dc)

    if not os.path.exists('proc/'):
        os.makedirs('proc/')

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


    #def add_particle(id, physprocess, advance):
    #    species = Params.get_species_by_id(id)
    #
    #    if (species.n_th == 0 or None) and (species.n_sp == 0 or None):
    #        return
    #    elif Params.int_spec["gen_max"] is not None or advance > Params.int_spec["gen_max"]:
    #        return
    #    elif physprocess not in ['thermal', 'sputter']:
    #        return
    #
    #    p = create_particle(species, physprocess)
    #    return p.xyz + p.vxyz


    for i in range(Params.int_spec["num_sim_advances"]):

        sim_N_before = sim.N

        # CREATE PARTICLES
        # ================

        #species_ids = [Params.get_species(s+1).id for s in range(Params.num_species)]
        #species_n = [Params.get_species(s+1).n_th + Params.get_species(s+1).n_sp for s in range(Params.num_species)]
        #
        #use_pool = any([n>1000 for n in species_n])
        #
        #if use_pool:
        #    inputs = [()]
        #    with mp.Pool() as p:
        #        r = p.starmap(add_particle, range(species.n_sp)), total=species.n_sp))

        addst = time.time()
        print("Creating particles: ")
        for ns in range(num_species):
            species = Params.get_species(ns+1)

            if (species.n_th == 0 or None) and (species.n_sp == 0 or None):
                continue

            # Add particles of given species
            # ------------------------------
            if Params.int_spec["gen_max"] is None or i < Params.int_spec["gen_max"]:
                if not (species.n_th == 0 or None):
                    for j1 in tqdm(range(species.n_th), desc=f"Adding {species.name} particles thermally"):

                        print("Thermal creation is not up-to-date!")

                        p = create_particle(species, "thermal")
                        identifier = f"{species.id}_{i}_{j1}"
                        p.hash = identifier
                        sim.add(p)

                        hash_dict[str(p.hash.value)] = {"identifier": identifier, "i": i, "id": species.id}

                if not (species.n_sp == 0 or None):

                    print(f"\t Adding sputtered {species.name}---{species.description}:")

                    def mp_addsput(_):
                        p = create_particle(species, "sputter")
                        return p.xyz + p.vxyz

                    with mp.Pool() as p:
                        r = p.map(mp_addsput, range(species.n_sp))

                    for index, coord in enumerate(r):
                        identifier = f"{species.id}_{i}_{index + species.n_th}"
                        sim.add(x=coord[0], y=coord[1], z=coord[2], vx=coord[3], vy=coord[4], vz=coord[5], hash=identifier)

                        #sim.particles[identifier].params["kappa"] = 1.0e-6 / species.mass_num
                        sim.particles[identifier].params["beta"] = species.beta

                        hash_dict[str(sim.particles[identifier].hash.value)] = {"identifier": identifier, "i": i, "id": species.id}

                    #for j2 in tqdm(range(species.n_sp), desc=f"Adding {species.name} particles via sputtering"):
                    #    p = create_particle(species, "sputter")
                    #    identifier = f"{species.id}_{i}_{j2 + species.n_th}"
                    #    p.hash = identifier
                    #    sim.add(p)
                    #    #sim.particles[identifier].params["kappa"] = 1.0e-6 / species.mass_num
                    #    sim.particles[identifier].params["beta"] = species.beta
                    #    hash_dict[str(p.hash.value)] = {"identifier": identifier, "i": i, "id": species.id}

        print(f"Time needed for adding particles: {time.time() - addst}")
        print("------------------------------------------------")


        # LOSS FUNCTION & CHEMICAL NETWORK
        # ================================
        print("Checking losses")
        boundary = Params.int_spec["r_max"] * moon_a if moon_exists else Params.int_spec["r_max"] * planet_a
        num_lost = 0
        num_converted = 0
        rng = np.random.default_rng()

        # Check all particles
        toberemoved = []
        for particle in sim.particles[sim.N_active:]:

            particle_iter = hash_dict[f"{particle.hash.value}"]["i"]
            species_id = hash_dict[f"{particle.hash.value}"]["id"]
            species = Params.get_species_by_id(species_id)

            if moon_exists:
                dt = deep_copies[0].t - int(particle_iter * Params.int_spec["sim_advance"] * moon_P)
            else:
                dt = deep_copies[0].t - int(particle_iter * Params.int_spec["sim_advance"] * planet_P)

            if species.duplicate is not None:
                species_id = int(str(species_id)[0])
                species = Params.get_species_by_id(species_id)

            # Remove if too far away:
            if moon_exists:
                particle_distance = np.linalg.norm(np.asarray(particle.xyz) - np.asarray(sim.particles["planet"].xyz))
            else:
                particle_distance = np.linalg.norm(np.asarray(particle.xyz) - np.asarray(sim.particles[0].xyz))
            if particle_distance > boundary:
                toberemoved.append(particle.hash)
                continue

            # Remove if chemical reaction happens:
            chem_network = species.network     # tau (float), educts (str), products (str), velocities (float)
            if not isinstance(chem_network, (int,float)):

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
                                    toberemoved.append(particle.hash)
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
                                ids = [val[temp] for key, val in hash_dict.items() if temp in val and val["i"] == particle_iter]

                                # Count number of product-species particles:
                                to_species_total = np.count_nonzero(np.asarray(ids) == to_species.id)

                                # Change particle hash
                                new_hash = f"{to_species.id}_{particle_iter}_{to_species_total+1}"
                                sim.particles[particle.hash].hash = new_hash

                                # Update library
                                hash_dict[f"{particle.hash.value}"] = {"identifier": new_hash, "i": particle_iter, "id": to_species.id}

                                num_converted += 1

                            else:
                                toberemoved.append(particle.hash)
                                break
                        break
            else:
                tau = chem_network
                prob_to_exist = np.exp(-dt / tau)
                if random.random() > prob_to_exist:
                    toberemoved.append(particle.hash)

        for r in range(len(toberemoved)):
            try:
                sim.remove(hash=toberemoved[r])
                #del hash_dict[f"{toberemoved[r].value}"]           WHY DOESN'T THIS WORK????
                num_lost += 1
            except:
                pass

        print(f"{num_lost} particles lost.")
        print(f"{num_converted} particles were converted.")
        print("------------------------------------------------")


        # WIP PARTICLE INTERPOLATION
        # ================================

        if not i < 2 and Params.int_spec["particle_interpolation"]:
            print("Particle interpolation...")
            counter = 0
            for iter in range(2, 3):
                lfkey = "identifier"
                new_part_idf = [val[lfkey] for key, val in hash_dict.items() if lfkey in val and val["i"] == iter - 1]
                adv_part_idf = [val[lfkey] for key, val in hash_dict.items() if lfkey in val and val["i"] == iter - 2]

                particle_hashes = []
                new_part = []
                adv_part = []
                new_part_species = []
                adv_part_species = []
                for x in range(sim.N_active, sim.N):
                    particle_hashes.append(sim.particles[x].hash.value)
                for idf in adv_part_idf:
                    if rebound.hash(idf).value in particle_hashes:
                        adv_part.append(sim.particles[idf])
                        adv_part_species.append(idf[0])
                for idf in new_part_idf:
                    if rebound.hash(idf).value in particle_hashes:
                        new_part.append(sim.particles[idf])
                        new_part_species.append(idf[0])

                #new_part = [sim.particles[idf] for idf in new_part_idf if rebound.hash(idf).value in particle_hashes]
                #adv_part = [sim.particles[idf] for idf in adv_part_idf if rebound.hash(idf).value in particle_hashes]
                #
                #new_part_species = [idf[0] for idf in new_part_idf if rebound.hash(idf).value in particle_hashes]
                #adv_part_species = [idf[0] for idf in adv_part_idf if rebound.hash(idf).value in particle_hashes]

                species_ids = np.unique(adv_part_species)

                for id in species_ids:
                    new_part_by_species = []
                    adv_part_by_species = []
                    for ind, spec in enumerate(new_part_species):
                        if spec == id:
                            new_part_by_species.append(new_part[ind])
                    for ind, spec in enumerate(adv_part_species):
                        if spec == id:
                            adv_part_by_species.append(adv_part[ind])

                    #new_part_by_species = [part for ind, part in enumerate(new_part) if new_part_species[ind] == id]
                    #adv_part_by_species = [part for ind, part in enumerate(adv_part) if adv_part_species[ind] == id]

                    inter_num = (len(new_part_by_species) + len(adv_part_by_species)) / 2

                    for k in range(int(inter_num)):
                        # REBOUND Operators bugged...
                        # Manually combining positions and velocities for new particles.
                        choice_new = random.choice(range(len(new_part_by_species)))
                        choice_adv = random.choice(range(len(adv_part_by_species)))
                        rand_new_xyz = np.asarray(new_part_by_species[choice_new].xyz)
                        rand_new_vxyz = np.asarray(new_part_by_species[choice_new].vxyz)
                        rand_adv_xyz = np.asarray(adv_part_by_species[choice_adv].xyz)
                        rand_adv_vxyz = np.asarray(adv_part_by_species[choice_adv].vxyz)
                        inter_xyz = (rand_new_xyz + rand_adv_xyz) / 2
                        inter_vxyz = (rand_new_vxyz + rand_adv_vxyz) / 2

                        inter_part = rebound.Particle(x=inter_xyz[0],y=inter_xyz[1],z=inter_xyz[2],vx=inter_vxyz[0], vy=inter_vxyz[1], vz=inter_vxyz[2])

                        if np.linalg.norm(np.asarray(inter_part.xyz) - np.asarray(sim.particles[1].xyz)) < np.linalg.norm(rand_new_xyz - np.asarray(sim.particles[1].xyz)):
                            continue

                        identifier = f"{int(id)}_{iter / 2}_{k}"
                        inter_part.hash = identifier

                        sim.add(inter_part)

                        # sim.particles[identifier].params["kappa"] = 1.0e-6 / species.mass_num
                        sim.particles[identifier].params["beta"] = Params.get_species_by_id(int(id)).beta

                        hash_dict[str(sim.particles[identifier].hash.value)] = {"identifier": identifier, "i": iter / 2,
                                                                                "id": int(id)}
                        counter += 1

            print(f"{counter} particles added through interpolation")


        # SAVE HASH_DICT
        # ==============
        hash_supdict[str(i+1)] = hash_dict.copy()


        # ADVANCE INTEGRATION
        # ===================

        #print("#######################################################")
        #print(f"Starting advance {i+1} ... ")
        ##advance = moon_P / sim.dt * Params.int_spec["sim_advance"] if moon_exists else planet_P / sim.dt * Params.int_spec["sim_advance"]
        ##sim.steps(int(advance))  # Only reliable with specific integrators that leave sim.dt constant (not the default one!)
        #
        ##sim.ri_whfast.recalculate_coordinates_this_timestep = 1
        #
        #start_time = time.time()
        #
        #advance = moon_P * Params.int_spec["sim_advance"] if moon_exists else planet_P * Params.int_spec["sim_advance"]
        #sim.dt = advance / 20
        #sim.integrate(int(advance * (i+1)), exact_finish_time=0)
        #
        #print(f"Calculation time for advance: {time.time() - start_time}")
        #
        #print("Advance done! ")
        #print("Number of particles: {0}".format(sim.N))
        #
        ##sim.integrator_synchronize()
        #
        #sim.simulationarchive_snapshot("archive.bin")
        #
        #print("#######################################################")

        print("#######################################################")
        print(f"Starting advance {i} ... ")
        start_time = time.time()
        cpus = multiprocessing.cpu_count()

        def advance_sim(dc_index):
            adv = moon_P * Params.int_spec["sim_advance"] if moon_exists else planet_P * Params.int_spec["sim_advance"]
            dc = deep_copies[dc_index]
            dc.dt = adv/10
            dc.integrate(int(adv*(i+1)))
            dc.simulationarchive_snapshot(f"proc/archiveProcess{dc_index}.bin", deletefile=True)

        lst = list(range(sim_N_before, sim.N))
        split = np.array_split(lst, cpus)
        processes = []
        for proc in range(cpus):
            dc = deep_copies[proc]
            for x in split[proc]:
                dc.add(sim.particles[int(x)])
            p = multiprocessing.Process(target=advance_sim, args=(proc,))
            p.start()
            processes.append(p)

        for ind, process in enumerate(processes):
            process.join()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                deep_copies[ind] = rebound.Simulation(f"proc/archiveProcess{ind}.bin")
                set_pointers(deep_copies[ind])

        print("\t MP Processes joined.")
        del sim.particles

        print("\t Transfering particle data...")
        for act in range(deep_copies[0].N_active):
            sim.add(deep_copies[0].particles[act])
        sim.N_active = deep_copies[0].N_active
        for proc in range(len(deep_copies)):
            dc = deep_copies[proc]
            for particle in dc.particles[dc.N_active:]:
                sim.add(particle)

        sim.simulationarchive_snapshot("archive.bin")

        print("Advance done! ")
        print(f"Simulation runtime: {time.time() - start_time}")
        print(f"Number of particles: {sim.N}")
        print("#######################################################")


        # SAVE HASH
        # ==============
        print("Saving hashes...")
        with open("hash_library.json", 'w') as f:
            json.dump(hash_supdict, f)
        print("\t ... done!")

        # Stop if steady state
        # --------------------
        ss_counter = 0
        if Params.int_spec["stop_at_steady_state"]:
            if np.abs(sim_N_before - sim.N) < 0.001:
                ss_counter += 1
                if ss_counter == 5:
                    print("Reached steady state!")
                    break
            else:
                ss_counter = 0


    print("Simulation completed successfully!")
    return


if __name__ == "__main__":
    Params = Parameters()
    init3(moon = Params.int_spec["moon"], additional_majors=False)
    run_simulation()
