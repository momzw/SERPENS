import rebound
import reboundx
import numpy as np
import warnings
import multiprocess
import multiprocessing
import dill
import copy
import os as os
import sys
from src.create_particle import create_particle
from parameters import Parameters
import time


def heartbeat(sim_pointer):
    sim = sim_pointer.contents
    par = Parameters.celest["moon"].copy()
    par.pop('hash', None)

    p0 = rebound.Particle(simulation=sim, **par, primary=sim.particles["planet"])
    o = p0.calculate_orbit(primary=sim.particles["planet"], G=sim.G)
    mean_anomaly = o.n * sim.t
    p1 = rebound.Particle(simulation=sim, M=mean_anomaly, **par, primary=sim.particles["planet"])

    sim.particles["moon"].xyz = p1.xyz
    sim.particles["moon"].vxyz = p1.vxyz


#def rebx_setup(reb_sim):
#    # REBOUNDX ADDITIONAL FORCES
#    # ==========================
#    rebx = reboundx.Extras(reb_sim)
#    rf = rebx.load_force("radiation_forces")
#    rebx.add_force(rf)
#    rf.params["c"] = 3.e8
#    reb_sim.particles["star"].params["radiation_source"] = 1
#    rebx.register_param('serpens_species', 'REBX_TYPE_INT')
#    rebx.register_param('serpens_weight', 'REBX_TYPE_DOUBLE')
#    return rebx


def reb_setup(params):
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

    for k, v in Parameters.celest.items():
        if not type(v) == dict:
            continue
        else:
            primary = v.pop("primary", 0)
            if reb_sim.N == 0:
                reb_sim.add(**v)
            else:
                reb_sim.add(**v, primary=reb_sim.particles[primary])
    reb_sim.N_active = len(Parameters.celest) - 1

    reb_sim.move_to_com()  # Center of mass coordinate-system (Jacobi coordinates without this line)

    # IMPORTANT:
    # * This setting boosts WHFast's performance, but stops automatic synchronization and recalculation of Jacobi coordinates!
    # * If particle masses are changed or massive particles' position/velocity are changed manually you need to include
    #   sim.ri_whfast.recalculate_coordinates_this_timestep
    # * Synchronization is needed if simulation gets manipulated or particle states get printed.
    # Refer to https://rebound.readthedocs.io/en/latest/ipython_examples/AdvWHFast/
    # => sim.ri_whfast.safe_mode = 0

    reb_sim.simulationarchive_snapshot("archive.bin", deletefile=True)
    with open(f"Parameters.txt", "w") as f:
        f.write(f"{params.__str__()}")

    print("\t \t ... done!")
    print("=======================================")

    return reb_sim  # REBX: , rebx


def set_pointers(reb_sim):
    reb_sim.collision = "direct"  # Brute force collision search and scales as O(N^2). It checks for instantaneous overlaps between every particle pair.
    reb_sim.collision_resolve = "merge"
    reb_sim.heartbeat = heartbeat

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
        part_state = create_particle(species.id, process=process, source=source_state, source_r=source_r,
                                     num=per_create)
        return part_state

    per_create = int(n / multiprocessing.cpu_count())

    with multiprocess.Pool(10) as p:
        r = p.map(mp_add, range(multiprocessing.cpu_count()))
        r = np.asarray(r).reshape(np.shape(r)[0] * np.shape(r)[1], 6)
        p.close()

    return r


class SerpensSimulation:

    def __init__(self, *args, **kw):

        print("=====================================")
        print("SERPENS simulation has been created.")
        print("=====================================")

        self.hash_supdict = {}
        self.hash_dict = {}

        # Handle arguments
        filename = None
        if len(args) > 0:
            filename = args[0]
        if "filename" in kw:
            filename = kw["filename"]
        if filename is not None:
            with open('hash_library.pickle', 'rb') as handle:
                self.hash_supdict = dill.load(handle)
            with open('Parameters.pickle', 'rb') as handle:
                params_load = dill.load(handle)
                params_load()
        self.params = Parameters()

        snapshot = -1
        if len(args) > 1:
            snapshot = args[1]
        if "snapshot" in kw:
            snapshot = kw["snapshot"]

        # Create simulation
        if filename is None:
            # Create a new simulation
            reb_sim = reb_setup(self.params)

            # REBX: reb_sim, rebx = reb_setup(self.params)
            # REBX: self.__rebx = rebx

            self.__sim = reb_sim
            iter = 0

            if os.path.exists("hash_library.pickle"):
                os.remove("hash_library.pickle")
        else:
            arch = rebound.SimulationArchive(filename, process_warnings=False)
            self.__sim = arch[snapshot]

            # REBX: arch, rebx = reboundx.SimulationArchive(filename, rebxfilename="rebx.bin")
            # REBX: self.__sim, self.__rebx = arch[snapshot]

            iter = len(arch) - 1 if snapshot == -1 else snapshot
            self.hash_dict = self.hash_supdict[f"{iter+1}"]

        self.__sim_deepcopies = []
        # REBX: self.__rebx_deepcopies = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(multiprocessing.cpu_count()):
                self.__sim_deepcopies.append(self.__sim.copy())

        self.var = {"iter": iter, "moon": self.params.int_spec["moon"]}
        if self.var["moon"]:
            self.var["source_a"] = self.__sim.particles["moon"].calculate_orbit(
                primary=self.__sim.particles["planet"]).a
            self.var["source_P"] = self.__sim.particles["moon"].calculate_orbit(
                primary=self.__sim.particles["planet"]).P
        else:
            self.var["source_a"] = self.__sim.particles["planet"].a
            self.var["source_P"] = self.__sim.particles["planet"].P
        self.var["boundary"] = self.params.int_spec["r_max"] * self.var["source_a"]

        set_pointers(self.__sim)
        for dc in self.__sim_deepcopies:
            set_pointers(dc)
            dc.t = self.var["iter"] * self.var["source_P"] * self.params.int_spec["sim_advance"]

    def __add_particles(self):

        if self.params.int_spec["moon"]:
            source = self.__sim.particles["moon"]
        else:
            source = self.__sim.particles["planet"]
        source_state = np.array([source.xyz, source.vxyz])

        if self.params.int_spec["gen_max"] is None or self.var['iter'] < self.params.int_spec["gen_max"]:
            for s in range(self.params.num_species):
                species = self.params.get_species(num=s + 1)

                rth = create(source_state, source.r, "thermal", species)
                rsp = create(source_state, source.r, "sputter", species)

                r = np.vstack((rth.reshape(len(rth), 6), rsp.reshape(len(rsp), 6)))

                for index, coord in enumerate(r):
                    identifier = f"{species.id}_{self.var['iter']}_{index}"
                    self.__sim.add(x=coord[0], y=coord[1], z=coord[2], vx=coord[3], vy=coord[4], vz=coord[5],
                                   hash=identifier)

                    # sim.particles[identifier].params["kappa"] = 1.0e-6 / species.mass_num
                    self.__sim.particles[identifier].params["beta"] = species.beta
                    # REBX: self.__sim.particles[identifier].params["serpens_species"] = species.id
                    # REBX: self.__sim.particles[identifier].params["serpens_weight"] = 1.

                    self.hash_dict[str(self.__sim.particles[identifier].hash.value)] = {"id": species.id,
                                                                                        "weight": 1}

    def __check_shadow(self, particle_pos):
        planet_radius = self.__sim.particles["planet"].r
        star_radius = self.__sim.particles["star"].r
        shadow_apex = np.asarray(self.__sim.particles["planet"].xyz) * (
                    1 + planet_radius / (star_radius - planet_radius))

        h = np.linalg.norm(self.__sim.particles["planet"].xyz) * planet_radius / (star_radius - planet_radius)
        axis_normal_vec = - np.asarray(self.__sim.particles["planet"].xyz) / np.linalg.norm(
            self.__sim.particles["planet"].xyz)

        cone_constant = planet_radius ** 2 / h

        Y0 = np.dot(axis_normal_vec, shadow_apex)

        coneY = np.dot(particle_pos, axis_normal_vec) - Y0
        if (coneY < 0) or (coneY > h):
            in_cone = False
        else:
            X = particle_pos - shadow_apex - coneY * axis_normal_vec
            X_square = np.dot(X, X)
            if X_square > cone_constant * coneY:
                in_cone = False
            else:
                in_cone = True

        return in_cone

    def __loss_per_advance(self):

        ## Check all particles
        exception_counter = 0
        for particle in self.__sim.particles[self.__sim.N_active:]:

            try:
                particle_weight = self.hash_dict[f"{particle.hash.value}"]["weight"]
                species_id = self.hash_dict[f"{particle.hash.value}"]["id"]
            except:
                print("Particle not found.")
                exception_counter += 1
                if exception_counter == 20:
                    raise Exception("Something went wrong..")
                else:
                    continue

            # REBX: particle_weight = particle.params["serpens_weight"]
            # REBX: species_id = particle.params["serpens_species"]

            species = self.params.get_species(id=species_id)

            if species.duplicate is not None:
                species_id = int(str(species_id)[0])
                species = self.params.get_species(id=species_id)

            if isinstance(species.tau_shielded, (float, int)) or (
                    self.params.int_spec["radiation_pressure_shield"] and species.beta > 0):
                if self.__check_shadow(particle.xyz):
                    if isinstance(species.tau_shielded, (float, int)):
                        chem_network = species.tau_shielded
                    else:
                        chem_network = species.network

                    if self.params.int_spec["radiation_pressure_shield"]:
                        particle.params["beta"] = 0
                else:
                    chem_network = species.network
                    particle.params["beta"] = species.beta
            else:
                chem_network = species.network  # tau (str), educts (str), products (str), velocities (str)

            dt = self.params.int_spec["sim_advance"] * self.var["source_P"]

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
        dc.integrate(adv * (self.var["iter"] + 1), exact_finish_time=0)
        dc.simulationarchive_snapshot(f"proc/archiveProcess{dc_index}.bin", deletefile=True)

        # REBX: dc_rebx = self.__rebx_deepcopies[dc_index]
        # REBX: dc_rebx.save(f"proc/archiveRebx{dc_index}.bin")

    def advance(self, num, save_freq=1):

        start_time = time.time()
        cpus = multiprocessing.cpu_count()
        steady_state_counter = 0
        steady_state_breaker = None

        for _ in range(num):

            print(f"Starting advance {self.var['iter']} ... ")
            n_before = self.__sim.N

            # ADD+REMOVE PARTICLES
            self.__add_particles()
            self.__loss_per_advance()

            # ADVANCE SIMULATION
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

                    # REBX: self.__rebx_deepcopies[ind] = reboundx.Extras(self.__sim_deepcopies[ind],
                    # REBX:                                             f"proc/archiveRebx{ind}.bin")

            print("\t MP Processes joined.")
            del self.__sim.particles

            print("\t Transfering particle data...")

            # Copy active objects from first simulation copy:
            for act in range(self.__sim_deepcopies[0].N_active):

                try:
                    if self.var["moon"]:
                        _ = self.__sim_deepcopies[0].particles["moon"]
                    _ = self.__sim_deepcopies[0].particles["planet"]
                except:
                    print("ERROR:")
                    print("moon or planet collided with the planet (if moon exists) or star!")
                    print("aborting simulation...")
                    return

                self.__sim.add(self.__sim_deepcopies[0].particles[act])

            self.__sim.N_active = self.__sim_deepcopies[0].N_active

            # Transfer superparticles
            num_lost = 0
            for proc in range(len(self.__sim_deepcopies)):

                dc = self.__sim_deepcopies[proc]

                dc_remove = []
                for particle in dc.particles[dc.N_active:]:

                    try:
                        w = self.hash_dict[f"{particle.hash.value}"]['weight']
                        species_id = self.hash_dict[f"{particle.hash.value}"]['id']
                        # REBX: w = particle.params["serpens_weight"]
                        # REBX: species_id = particle.params["serpens_species"]
                    except:
                        print("Particle not found.")
                        dc_remove.append(particle.hash)
                        continue

                    species = self.params.get_species(id=species_id)
                    mass_inject_per_advance = species.mass_per_sec * self.params.int_spec["sim_advance"] * self.var[
                        "source_P"]
                    pps = species.particles_per_superparticle(mass_inject_per_advance)

                    if self.var["moon"]:
                        particle_distance = np.linalg.norm(
                            np.asarray(particle.xyz) - np.asarray(self.__sim.particles["planet"].xyz))
                    else:
                        particle_distance = np.linalg.norm(
                            np.asarray(particle.xyz) - np.asarray(self.__sim.particles[0].xyz))

                    if particle_distance > self.var["boundary"]:
                        try:
                            dc_remove.append(particle.hash)
                        except RuntimeError:
                            print("Removal error occurred.")
                            pass
                    elif w * pps < 1e10:
                        try:
                            dc_remove.append(particle.hash)
                        except RuntimeError:
                            print("Removal error occurred.")
                            pass
                    else:
                        self.__sim.add(particle)

                for hash in dc_remove:
                    dc.remove(hash=hash)
                    try:
                        del self.hash_dict[f"{hash.value}"]
                    except:
                        continue
                    num_lost += 1

            t = self.__sim_deepcopies[0].t
            self.__sim.simulationarchive_snapshot("archive.bin")
            # REBX: self.__rebx.save("rebx.bin")

            print("Advance done! ")
            print(f"Simulation time [h]: {np.around(t / 3600, 2)}")
            print(f"Simulation runtime [s]: {np.around(time.time() - start_time, 2)}")
            print(f"Number of particles removed: {num_lost}")
            print(f"Number of particles: {self.__sim.N}")

            if self.var["iter"] % save_freq == 0:
                print("Saving hash dict...")
                # self.hash_supdict[str(self.var["iter"] + 1)] = copy.deepcopy(self.hash_dict)
                dict_saver = {f"{str(self.var['iter'] + 1)}": copy.deepcopy(self.hash_dict)}
                with open("hash_library.pickle", 'ab') as f:
                    dill.dump(dict_saver, f, dill.HIGHEST_PROTOCOL)

            self.var["iter"] += 1

            if np.abs(self.__sim.N - n_before) < 50:
                steady_state_counter += 1
                if steady_state_counter == 5 and self.params.int_spec["stop_at_steady_state"] == True:
                    print("Steady state reached!")
                    print("Stopping after another successful revolution...")
                    steady_state_breaker = 1
            else:
                steady_state_counter = 0

            if steady_state_breaker is not None:
                print(f"Advances left: {1/self.params.int_spec['sim_advance'] - steady_state_breaker}")
                if steady_state_breaker == 5/4 * 1/self.params.int_spec["sim_advance"]:
                    break
                else:
                    steady_state_breaker += 1

            print("\t ... done!")
            print("#######################################################")


if __name__ == "__main__":
    params = Parameters()
    with open("Parameters.pickle", 'wb') as f:
        dill.dump(params, f, protocol=dill.HIGHEST_PROTOCOL)

    ssim = SerpensSimulation()
    ssim.advance(Parameters.int_spec["num_sim_advances"])
