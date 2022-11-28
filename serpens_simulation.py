import rebound
import reboundx
import numpy as np
import random
import json
import warnings
import os

import multiprocess as mp
import multiprocessing

from itertools import repeat

import pickle

from create_particle import create_particle
from init import init3, Parameters
import time


params = Parameters()


def reb_setup():
    print("=======================================")
    print("Initializing new simulation instance...")

    reb_sim = rebound.Simulation()
    reb_sim.integrator = "whfast"  # Fast and unbiased symplectic Wisdom-Holman integrator. Suitability not yet assessed.
    reb_sim.ri_whfast.kernel = "lazy"
    reb_sim.collision = "direct"  # Brute force collision search and scales as O(N^2). It checks for instantaneous overlaps between every particle pair.
    reb_sim.collision_resolve = "merge"

    # SI units:
    reb_sim.units = ('m', 's', 'kg')
    reb_sim.G = 6.6743e-11

    reb_sim.dt = 500

    # PRELIMINARY: moon defines which objects to use!
    # ----------------------------------------------
    if params.int_spec["moon"]:
        # labels = ["Sun", "Jupiter", "Io"]
        # sim.add(labels)      # Note: Takes current position in the solar system. Therefore more useful to define objects manually in the following.
        reb_sim.add(m=1.988e30, hash="sun")
        reb_sim.add(m=1.898e27, a=7.785e11, e=0.0489, inc=0.0227, primary=reb_sim.particles["sun"],
                    hash="planet")  # Omega=1.753, omega=4.78

        reb_sim.particles["sun"].r = 696340000
        reb_sim.particles["planet"].r = 69911000
    else:
        # 55 Cancri e
        # -----------
        reb_sim.add(m=1.799e30, hash="sun")
        reb_sim.add(m=4.77179e25, a=2.244e9, e=0.05, inc=0.00288, primary=reb_sim.particles["sun"], hash="planet")

        reb_sim.particles["sun"].r = 6.56e8
        reb_sim.particles["planet"].r = 1.196e7
    # ----------------------------------------------

    if params.int_spec["moon"]:
        # sim.add(m=8.932e22, a=4.217e8, e=0.0041, inc=0.0386, primary=sim.particles["planet"], hash="moon")
        # sim.particles["moon"].r = 1821600

        reb_sim.add(m=4.799e22, a=6.709e8, e=0.009, inc=0.0082, primary=reb_sim.particles["planet"], hash="moon")
        reb_sim.particles["moon"].r = 1560800

        reb_sim.N_active = 3
    else:
        reb_sim.N_active = 2

    reb_sim.move_to_com()  # Center of mass coordinate-system (Jacobi coordinates without this line)

    # IMPORTANT:
    # * This setting boosts WHFast's performance, but stops automatic synchronization and recalculation of Jacobi coordinates!
    # * If particle masses are changed or massive particles' position/velocity are changed manually you need to include
    #   sim.ri_whfast.recalculate_coordinates_this_timestep
    # * Synchronization is needed if simulation gets manipulated or particle states get printed.
    # Refer to https://rebound.readthedocs.io/en/latest/ipython_examples/AdvWHFast/
    # => sim.ri_whfast.safe_mode = 0

    reb_sim.simulationarchive_snapshot("archive.bin", deletefile=True)
    with open(f"Parameters.txt", "w") as text_file:
        text_file.write(f"{params.__str__()}")

    print("\t \t ... done!")
    print("=======================================")

    return reb_sim


def set_pointers(reb_sim):
    reb_sim.collision = "direct"   # Brute force collision search and scales as O(N^2). It checks for instantaneous overlaps between every particle pair.
    reb_sim.collision_resolve = "merge"

    # REBOUNDX ADDITIONAL FORCES
    # ==========================
    rebxdc = reboundx.Extras(reb_sim)
    rf = rebxdc.load_force("radiation_forces")
    rebxdc.add_force(rf)
    rf.params["c"] = 3.e8


def create(source_state, source_r, process, species):

    if process == "thermal":
        n = species.n_th
    elif process == "sputter":
        n = species.n_sp
    else:
        raise ValueError("Invalid process in particle creation.")

    if n == 0 or n is None:
        return np.array([])

    def mp_add(_):
        part_state = create_particle(species.id, process=process, source=source_state, source_r=source_r, num=per_create)
        return part_state

    per_create = int(n / multiprocessing.cpu_count())

    with mp.Pool(multiprocessing.cpu_count()) as p:
        r = p.map(mp_add, range(multiprocessing.cpu_count()))
        r = np.asarray(r).reshape(np.shape(r)[0] * np.shape(r)[1], 6)

    return r


class SerpensSimulation:

    __instance = None

    __sim = None
    __sim_deepcopies = []
    hash_supdict = {}
    hash_dict = {}
    params = Parameters()
    var = {"iter": 0}

    def __new__(cls, *args, **kw):
        if cls.__instance is None:
            # Handle arguments
            filename = None
            if len(args) > 0:
                filename = args[0]
            if "filename" in kw:
                filename = kw["filename"]
            snapshot = -1
            if len(args) > 1:
                snapshot = args[1]
            if "snapshot" in kw:
                snapshot = kw["snapshot"]

            # Create simulation
            if filename is None:
                # Create a new simulation
                reb_sim = reb_setup()
                cls.__sim = reb_sim
            else:
                cls.__sim = rebound.SimulationArchive(filename,process_warnings=False)[snapshot]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(multiprocessing.cpu_count()):
                    cls.__sim_deepcopies.append(cls.__sim.copy())

            cls.var["moon"] = cls.params.int_spec["moon"]
            if cls.var["moon"]:
                cls.var["source_a"] = cls.__sim.particles["moon"].calculate_orbit(primary=cls.__sim.particles["planet"]).a
                cls.var["source_P"] = cls.__sim.particles["moon"].calculate_orbit(primary=cls.__sim.particles["planet"]).P
            else:
                cls.var["source_a"] = cls.__sim.particles["planet"].a
                cls.var["source_P"] = cls.__sim.particles["planet"].P
            cls.var["boundary"] = cls.params.int_spec["r_max"] * cls.var["source_a"]

            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __init__(self):
        set_pointers(self.__sim)
        for dc in self.__sim_deepcopies:
            set_pointers(dc)
        print("Serpens run class called.")

    def __add_celest(self, name, radius = 0, primary_hash = "planet", **kw):
        self.__sim.add(primary=self.__sim.particles[primary_hash], **kw)
        self.__sim.particles[f"{name}"].r = radius

    def __add_particles(self):

        if self.params.int_spec["moon"]:
            source = self.__sim.particles["moon"]
        else:
            source = self.__sim.particles["planet"]
        source_state = np.array([source.xyz, source.vxyz])

        if params.int_spec["gen_max"] is None or self.var['iter'] < params.int_spec["gen_max"]:
            for s in range(params.num_species):
                species = params.get_species(s + 1)

                rth = create(source_state, source.r, "thermal", species)
                rsp = create(source_state, source.r, "sputter", species)

                r = np.vstack((rth.reshape(len(rth), 6), rsp.reshape(len(rsp), 6)))

                for index, coord in enumerate(r):
                    identifier = f"{species.id}_{self.var['iter']}_{index}"
                    self.__sim.add(x=coord[0], y=coord[1], z=coord[2], vx=coord[3], vy=coord[4], vz=coord[5],
                                   hash=identifier)

                    # sim.particles[identifier].params["kappa"] = 1.0e-6 / species.mass_num
                    self.__sim.particles[identifier].params["beta"] = species.beta

                    self.hash_dict[str(self.__sim.particles[identifier].hash.value)] = {"identifier": identifier,
                                                                                        "i": self.var['iter'],
                                                                                        "id": species.id,
                                                                                        "weight": 1,
                                                                                        "products_weight": np.zeros(params.num_species)}

    def __loss_per_advance(self):

        ## Check all particles
        for particle in self.__sim.particles[self.__sim.N_active:]:

            particle_weight = self.hash_dict[f"{particle.hash.value}"]["weight"]
            species_id = self.hash_dict[f"{particle.hash.value}"]["id"]
            species = self.params.get_species_by_id(species_id)

            if species.duplicate is not None:
                species_id = int(str(species_id)[0])
                species = self.params.get_species_by_id(species_id)

            # Remove if too far away:
            if self.var["moon"]:
                particle_distance = np.linalg.norm(np.asarray(particle.xyz) - np.asarray(self.__sim.particles["planet"].xyz))
            else:
                particle_distance = np.linalg.norm(np.asarray(particle.xyz) - np.asarray(self.__sim.particles[0].xyz))
            if particle_distance > self.var["boundary"]:
                self.__sim.remove(particle.hash)
                continue

            # Remove if chemical reaction happens:
            chem_network = species.network  # tau (float), educts (str), products (str), velocities (float)

            dt = self.params.int_spec["sim_advance"] * self.var["source_P"]
            mass_inject_per_advance = species.mass_per_sec * dt

            pps = species.particles_per_superparticle(mass_inject_per_advance)

            if not isinstance(chem_network, (int, float)):
                # Go through all reactions/lifetimes
                for l in range(np.size(chem_network[:, 0])):
                    tau = float(chem_network[:, 0][l])
                    particle_weight = particle_weight * np.exp(-dt / tau)
            else:
                tau = chem_network
                particle_weight = particle_weight * np.exp(-dt / tau)

            self.hash_dict[f"{particle.hash.value}"].update({'weight': particle_weight})

    def __advance_integration(self, dc_index):
        adv = self.var["source_P"] * self.params.int_spec["sim_advance"]
        dc = self.__sim_deepcopies[dc_index]
        dc.dt = adv / 10
        dc.integrate(int(adv * (self.var["iter"] + 1)), exact_finish_time=0)
        dc.simulationarchive_snapshot(f"proc/archiveProcess{dc_index}.bin", deletefile=True)

    def advance(self, num):

        start_time = time.time()
        cpus = multiprocessing.cpu_count()

        for _ in range(num):

            print(f"Starting advance {self.var['iter']} ... ")
            n_before = self.__sim.N

            self.__add_particles()
            self.__loss_per_advance()

            lst = list(range(n_before, self.__sim.N))
            split = np.array_split(lst, cpus)
            processes = []
            for proc in range(cpus):
                dc = self.__sim_deepcopies[proc]
                for x in split[proc]:
                    dc.add(self.__sim.particles[int(x)])
                p = multiprocessing.Process(target=self.__advance_integration, args=(proc,))
                p.start()
                processes.append(p)

            for ind, process in enumerate(processes):
                process.join()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.__sim_deepcopies[ind] = rebound.Simulation(f"proc/archiveProcess{ind}.bin")
                    set_pointers(self.__sim_deepcopies[ind])

            print("\t MP Processes joined.")
            del self.__sim.particles

            print("\t Transfering particle data...")
            for act in range(self.__sim_deepcopies[0].N_active):
                self.__sim.add(self.__sim_deepcopies[0].particles[act])
            self.__sim.N_active = self.__sim_deepcopies[0].N_active

            for proc in range(len(self.__sim_deepcopies)):

                dc = self.__sim_deepcopies[proc]

                for particle in dc.particles[dc.N_active:]:

                    w = self.hash_dict[f"{particle.hash.value}"]['weight']
                    species_id = self.hash_dict[f"{particle.hash.value}"]['id']
                    species = self.params.get_species_by_id(species_id)
                    mass_inject_per_advance = species.mass_per_sec * self.params.int_spec["sim_advance"] * self.var["source_P"]
                    pps = species.particles_per_superparticle(mass_inject_per_advance)


                    if self.var["moon"]:
                        particle_distance = np.linalg.norm(np.asarray(particle.xyz) - np.asarray(self.__sim.particles["planet"].xyz))
                    else:
                        particle_distance = np.linalg.norm(np.asarray(particle.xyz) - np.asarray(self.__sim.particles[0].xyz))


                    if particle_distance > self.var["boundary"]:
                        dc.remove(hash=particle.hash)
                        #del hash_dict[f"{particle.hash.value}"]
                    elif w * pps < 1:
                        dc.remove(hash=particle.hash)
                        #del hash_dict[f"{particle.hash.value}"]
                    else:
                        self.__sim.add(particle)

            self.var["iter"] += 1
            self.__sim.simulationarchive_snapshot("archive.bin")

            print("Advance done! ")
            print(f"Simulation runtime: {time.time() - start_time}")
            print(f"Number of particles: {self.__sim.N}")

            print("Saving hash dict...")
            self.hash_supdict[str(self.var["iter"] + 1)] = self.hash_dict
            with open("hash_library.pickle", 'wb') as f:
                pickle.dump(self.hash_supdict, f, pickle.HIGHEST_PROTOCOL)
            print("\t ... done!")
            print("#######################################################")












