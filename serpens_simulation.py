import rebound
import reboundx
import numpy as np
import warnings
import multiprocess
import multiprocessing
import dill
from src.create_particle import create_particle
from parameters import Parameters
import time


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
        primary = v.pop("primary", 0)
        if reb_sim.N == 0:
            reb_sim.add(**v)
        else:
            reb_sim.add(**v, primary=reb_sim.particles[primary])
    reb_sim.N_active = len(Parameters.celest)

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

    return reb_sim


def set_pointers(reb_sim):
    reb_sim.collision = "direct"  # Brute force collision search and scales as O(N^2). It checks for instantaneous overlaps between every particle pair.
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
            self.__sim = reb_sim
            iter = 0
        else:
            arch = rebound.SimulationArchive(filename, process_warnings=False)
            self.__sim = arch[snapshot]
            iter = len(arch) - 1 if snapshot == -1 else snapshot
            self.hash_dict = self.hash_supdict[f"{iter+1}"]

        self.__sim_deepcopies = []
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

                    self.hash_dict[str(self.__sim.particles[identifier].hash.value)] = {"identifier": identifier,
                                                                                        "i": self.var['iter'],
                                                                                        "id": species.id,
                                                                                        "weight": 1,
                                                                                        "products_weight": np.zeros(
                                                                                            self.params.num_species)}

    def __loss_per_advance(self):

        ## Check all particles
        for particle in self.__sim.particles[self.__sim.N_active:]:

            particle_weight = self.hash_dict[f"{particle.hash.value}"]["weight"]
            species_id = self.hash_dict[f"{particle.hash.value}"]["id"]
            species = self.params.get_species(id=species_id)

            if species.duplicate is not None:
                species_id = int(str(species_id)[0])
                species = self.params.get_species(id=species_id)

            # Check if chemical reaction happens:
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

                    w = self.hash_dict[f"{particle.hash.value}"]['weight']
                    species_id = self.hash_dict[f"{particle.hash.value}"]['id']
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
                            # dc.remove(hash=particle.hash)
                            # del hash_dict[f"{particle.hash.value}"]
                        except:
                            print("Removal error occurred.")
                            pass
                    elif w * pps < 1e10:
                        try:
                            dc_remove.append(particle.hash)
                            # dc.remove(hash=particle.hash)
                            # del hash_dict[f"{particle.hash.value}"]
                        except:
                            print("Removal error occurred.")
                            pass
                    else:
                        self.__sim.add(particle)

                for hash in dc_remove:
                    dc.remove(hash=hash)
                    # del self.hash_dict[f"{particle.hash.value}"]
                    num_lost += 1

            t = self.__sim_deepcopies[0].t
            self.__sim.simulationarchive_snapshot("archive.bin")

            print("Advance done! ")
            print(f"Simulation time [h]: {np.around(t / 3600, 2)}")
            print(f"Simulation runtime [s]: {np.around(time.time() - start_time, 2)}")
            print(f"Number of particles removed: {num_lost}")
            print(f"Number of particles: {self.__sim.N}")

            print("Saving hash dict...")
            self.hash_supdict[str(self.var["iter"] + 1)] = self.hash_dict
            with open("hash_library.pickle", 'wb') as f:
                dill.dump(self.hash_supdict, f, dill.HIGHEST_PROTOCOL)

            self.var["iter"] += 1

            print("\t ... done!")
            print("#######################################################")


if __name__ == "__main__":
    params = Parameters()
    with open("Parameters.pickle", 'wb') as f:
        dill.dump(params, f, protocol=dill.HIGHEST_PROTOCOL)

    ssim = SerpensSimulation()
    ssim.advance(Parameters.int_spec["num_sim_advances"])
