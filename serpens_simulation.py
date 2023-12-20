import rebound
import reboundx
import numpy as np
import warnings
import multiprocessing
import concurrent.futures
import dill
import copy
import os as os
import functools
from src.create_particle import create_particle
from src.parameters import Parameters
from tqdm import tqdm
import time


def heartbeat(sim_pointer):
    """
    Not meant for external use.
    REBOUND heartbeat function to fix a source's orbit to be circular if the source is a moon.
    """
    sim = sim_pointer.contents

    if Parameters.int_spec["source_index"] > 2:
        for key, subdict in Parameters.celest.items():
            if "source" in subdict:
                par = subdict.copy()
        par.pop('primary', None)
        par.pop('source', None)

        p0 = rebound.Particle(simulation=sim, **par, primary=sim.particles["source_primary"])
        o = p0.orbit(primary=sim.particles["source_primary"], G=sim.G)
        mean_anomaly = o.n * sim.t
        p1 = rebound.Particle(simulation=sim, M=mean_anomaly, **par, primary=sim.particles["source_primary"])

        sim.particles["source"].xyz = p1.xyz
        sim.particles["source"].vxyz = p1.vxyz


def rebx_setup(reb_sim):
    # REBOUNDX ADDITIONAL FORCES
    # This is a way of saving the particle information with REBX, but it appears to be slower.
    # ==========================
    rebx = reboundx.Extras(reb_sim)
    rf = rebx.load_force("radiation_forces")
    rebx.add_force(rf)
    rf.params["c"] = 3.e8
    reb_sim.particles["star"].params["radiation_source"] = 1
    rebx.register_param('serpens_species', 'REBX_TYPE_INT')
    rebx.register_param('serpens_weight', 'REBX_TYPE_DOUBLE')
    return rebx


def rebound_setup(params):
    """
    Not meant for external use.
    REBOUND simulation set up. Sets integrator and collision algorithm.
    Adds the gravitationally acting bodies to the simulation.
    Saves used parameters to drive.
    """
    print("Initializing new simulation instance...")

    reb_sim = rebound.Simulation()
    reb_sim.integrator = "whfast"  # Fast and unbiased symplectic Wisdom-Holman integrator.
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
            v_copy = v.copy()
            v_copy.pop("primary", 0)
            v_copy.pop("source", 0)
            source_index = Parameters.int_spec["source_index"]
            source_is_planet = source_index == 2
            if reb_sim.N == 0:
                if source_is_planet:
                    reb_sim.add(**v_copy, hash="source_primary")
                else:
                    reb_sim.add(**v_copy, hash="star")
                continue
            if reb_sim.N == 1:
                if source_is_planet:
                    reb_sim.add(**v_copy, primary=reb_sim.particles[0], hash="source")
                else:
                    reb_sim.add(**v_copy, primary=reb_sim.particles[0], hash="source_primary")
                continue
            else:
                if not source_is_planet:
                    if reb_sim.N == source_index - 1:
                        reb_sim.add(**v_copy, primary=reb_sim.particles["source_primary"], hash='source')
                    else:
                        reb_sim.add(**v_copy, primary=reb_sim.particles["source_primary"])
                else:
                    reb_sim.add(**v_copy, primary=reb_sim.particles["source_primary"])
    reb_sim.N_active = len(Parameters.celest) - 1

    reb_sim.move_to_com()  # Center of mass coordinate-system (Jacobi coordinates without this line)

    # IMPORTANT:
    # * This setting boosts WHFast's performance, but stops automatic synchronization and recalculation of Jacobi coordinates!
    # * If particle masses are changed or massive particles' position/velocity are changed manually you need to include
    #   sim.ri_whfast.recalculate_coordinates_this_timestep
    # * Synchronization is needed if simulation gets manipulated or particle states get printed.
    # Refer to https://rebound.readthedocs.io/en/latest/ipython_examples/AdvWHFast/
    # => sim.ri_whfast.safe_mode = 0

    rebx = rebx_setup(reb_sim)

    reb_sim.save_to_file("archive.bin", delete_file=True)
    with open(f"Parameters.txt", "w") as f:
        f.write(f"{params.__str__()}")

    with open("Parameters.pickle", 'wb') as f:
        dill.dump(params, f, protocol=dill.HIGHEST_PROTOCOL)

    print("\t \t ... done!")

    return reb_sim, rebx


def set_pointers(reb_sim):
    """
    Not meant for external use.
    Sets the pointers for handling simulation sub-instances in the integration process.
    Responsible for including REBOUNDX additional forces.
    """
    # Brute force collision search, scales as O(N^2). It checks for instantaneous overlaps between every particle pair.
    reb_sim.collision = "direct"
    reb_sim.collision_resolve = "merge"
    if Parameters.int_spec["fix_source_circular_orbit"]:
        reb_sim.heartbeat = heartbeat

    # REBOUNDX ADDITIONAL FORCES
    # ==========================
    rebxdc = reboundx.Extras(reb_sim)
    rf = rebxdc.load_force("radiation_forces")
    rebxdc.add_force(rf)
    rf.params["c"] = 3.e8
    reb_sim.particles["star"].params["radiation_source"] = 1
    rebxdc.register_param('serpens_species', 'REBX_TYPE_INT')
    rebxdc.register_param('serpens_weight', 'REBX_TYPE_DOUBLE')


def create(source_state, source_r, phys_process, species):
    """
    Creates a batch of particles to be added to a SERPENS simulation.
    Utilizes multiprocessing and returns an array of particle state vectors.

    Arguments
    ---------
    source_state : array-like
        Coordinates and velocity components of the source to be added to created particles' states.
    source_r : float
        Radius of the source. Needed for correct anchoring of new particle vectors.
    phys_process : str
        Physical process responsible for the creation of particles.
        Currently implemented are thermal evaporation and sputtering.
    species : Species class instance
        Species to be created.
    """
    if phys_process == "thermal":
        n = species.n_th
    elif phys_process == "sputter":
        n = species.n_sp
    else:
        raise ValueError("Invalid process in particle creation.")

    if n == 0 or n is None:
        return np.array([])

    # Use the number of available CPU cores
    num_processes = min([multiprocessing.cpu_count(), n])

    def add_with_multiprocessing():
        per_create = int(n / num_processes)
        part_state = create_particle(species.id, process=phys_process, source=source_state, source_r=source_r,
                                     num=per_create)
        return part_state

    # Create a ThreadPoolExecutor with the desired number of threads/processes
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        # Submit function multiple times for parallel execution
        future_to_result = {executor.submit(add_with_multiprocessing): None for _ in range(num_processes)}

        # Collect the results from the futures
        results = []
        for future in concurrent.futures.as_completed(future_to_result):
            result = future.result()
            results.append(result)

    # Results as ndarray and reshape before returning
    results = np.asarray(results).reshape(np.shape(results)[0] * np.shape(results)[1], 6)

    return results


class SerpensSimulation:
    """
    Main class responsible for the Monte Carlo process of SERPENS.
    (Simulating the Evolution of Ring Particles Emergent from Natural Satellites)
    """

    def __init__(self, system='default', *args, **kw):
        """
        Initializes a REBOUND simulation instance, as well as used class instance variables.

        Arguments
        ---------
        system : str    (default: 'default')
            Name of the celestial system to be simulated.
            Valid are all names that have been set up in the src/objects.txt.
        """
        print("=======================================")
        print("SERPENS simulation has been created.")
        print("=======================================")

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
        if not system == 'default':
            try:
                Parameters.modify_objects(celestial_name=system)
            except Exception:
                print("Cannot change the celestial objects. Are you sure you have implemented this system?")
                print("Exiting...")
                exit()

        snapshot = -1
        if len(args) > 1:
            snapshot = args[1]
        if "snapshot" in kw:
            snapshot = kw["snapshot"]

        # Create simulation
        if filename is None:
            # Create a new simulation
            # reb_sim = rebound_setup(self.params)

            reb_sim, rebx = rebound_setup(self.params)
            self._rebx = rebx

            self._sim = reb_sim
            iter = 0

            if os.path.exists("hash_library.pickle"):
                os.remove("hash_library.pickle")
        else:
            #arch = rebound.Simulationarchive(filename, process_warnings=False)
            #self._sim = arch[snapshot]

            arch, rebx = reboundx.Simulationarchive(filename, rebxfilename="rebx.bin")
            self._sim, self._rebx = arch[snapshot]

            iter = len(arch) - 1 if snapshot == -1 else snapshot
            self.hash_dict = self.hash_supdict[f"{iter + 1}"]

        self._sim_partial_processes = []
        # REBX: self.__rebx_deepcopies = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(multiprocessing.cpu_count()):
                self._sim_partial_processes.append(self._sim.copy())

        self.var = {"iter": iter,
                    "source_a": self._sim.particles["source"].orbit(
                        primary=self._sim.particles["source_primary"]).a,
                    "source_P": self._sim.particles["source"].orbit(
                        primary=self._sim.particles["source_primary"]).P}

        self.var["boundary"] = self.params.int_spec["r_max"] * self.var["source_a"]

        set_pointers(self._sim)
        for dc in self._sim_partial_processes:
            set_pointers(dc)
            dc.t = self.var["iter"] * self.var["source_P"] * self.params.int_spec["sim_advance"]

    def _add_particles(self):
        """
        Internal use only.
        Calls particle creation and adds the created particles to the REBOUND simulation instance.
        Saves particle hashes to a dictionary which contains information about a particle's species and weight factor.
        """
        source = self._sim.particles["source"]
        source_state = np.array([source.xyz, source.vxyz])

        if self.params.int_spec["gen_max"] is None or self.var['iter'] < self.params.int_spec["gen_max"]:
            for s in range(self.params.num_species):
                species = self.params.get_species(num=s + 1)

                rth = create(source_state, source.r, "thermal", species)
                rsp = create(source_state, source.r, "sputter", species)

                r = np.vstack((rth.reshape(len(rth), 6), rsp.reshape(len(rsp), 6)))

                for index, coord in enumerate(r):
                    identifier = f"{species.id}_{self.var['iter']}_{index}"
                    self._sim.add(x=coord[0], y=coord[1], z=coord[2], vx=coord[3], vy=coord[4], vz=coord[5],
                                  hash=identifier)

                    # sim.particles[identifier].params["kappa"] = 1.0e-6 / species.mass_num
                    self._sim.particles[identifier].params["beta"] = species.beta
                    self._sim.particles[identifier].params["serpens_species"] = species.id
                    self._sim.particles[identifier].params["serpens_weight"] = 1.

                    self.hash_dict[str(self._sim.particles[identifier].hash.value)] = {"id": species.id,
                                                                                       "weight": 1}

    def _check_shadow(self, particle_pos):
        """
        Internal use only.
        Checks if a particle resides inside the shadow of the planet.
        Returns a boolean value.
        """
        planet = self._sim.particles[1]
        star = self._sim.particles[0]
        shadow_apex = np.asarray(planet.xyz) * (1 + planet.r / (star.r - planet.r))

        h = np.linalg.norm(planet.xyz) * planet.r / (star.r - planet.r)
        axis_normal_vec = - np.asarray(planet.xyz) / np.linalg.norm(planet.xyz)

        cone_constant = planet.r ** 2 / h

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

    def _loss_per_advance(self):
        """
        Internal use only.
        Reads the hash dictionary and updates the weight of particles according to the species lifetime and time that
        the particles have been in the simulation. The weight is important for density calculations.
        """
        # Check all particles
        exception_counter = 0
        for particle in self._sim.particles[self._sim.N_active:]:

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

            particle_weight = particle.params["serpens_weight"]
            species_id = particle.params["serpens_species"]

            species = self.params.get_species(id=species_id)

            if species.duplicate is not None:
                species_id = int(str(species_id)[0])
                species = self.params.get_species(id=species_id)

            if isinstance(species.tau_shielded, (float, int)) or (
                    self.params.int_spec["radiation_pressure_shield"] and species.beta > 0):
                if self._check_shadow(particle.xyz):
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
                particle.params["serpens_weight"] = particle_weight

            self.hash_dict[f"{particle.hash.value}"].update({'weight': particle_weight})

    def _advance_integration(self, dc_index):
        """
        Internal use only.
        Function to integrate a REBOUND simulation instance that has been split from the main simulation.
        Multiprocessing allows for the integration of multiple sub-simulations at the same time.
        Saves the partial simulation to a process archive file in order to recombine after multiprocessing.
        """
        adv = self.var["source_P"] * self.params.int_spec["sim_advance"]
        dc = self._sim_partial_processes[dc_index]
        dc.dt = adv / 10
        dc.integrate(adv * (self.var["iter"] + 1), exact_finish_time=0)

        #dc.save_to_file(f"proc/archiveProcess{dc_index}.bin", delete_file=True)

        # REBX: dc_rebx = self.__rebx_deepcopies[dc_index]
        # REBX: dc_rebx.save(f"proc/archiveRebx{dc_index}.bin")

    def _advance_integration_wrapper(self, proc, split):
        """
        Internal use only.
        Wrapper function to assign sub-simulations a subset of all particles before integration.
        """
        dc = self._sim_partial_processes[proc]
        for x in split[proc]:
            dc.add(self._sim.particles[int(x)])
        self._advance_integration(proc)

        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        #    self._sim_partial_processes[proc] = rebound.Simulation(f"proc/archiveProcess{proc}.bin")
        #    set_pointers(self._sim_partial_processes[proc])
        #    # w/REBX: self.__rebx_deepcopies[ind] = reboundx.Extras(self.__sim_deepcopies[ind],
        #    # w/REBX:                                             f"proc/archiveRebx{ind}.bin")

    def single_advance(self, verbose=False):
        cpus = multiprocessing.cpu_count()

        n_before = self._sim.N

        # ADD & REMOVE PARTICLES
        self._add_particles()
        self._loss_per_advance()

        # ADVANCE SIMULATION
        lst = list(range(n_before, self._sim.N))
        split = np.array_split(lst, cpus)
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpus) as executor:
            wrapped_fn = functools.partial(self._advance_integration_wrapper, split=split)
            executor.map(wrapped_fn, range(cpus))

        if verbose:
            print("\t MP Processes joined.")
            print("Transferring particle data...")

        del self._sim.particles

        # Copy active objects from first simulation copy:
        for act in range(self._sim_partial_processes[0].N_active):
            try:
                _ = self._sim_partial_processes[0].particles["source"]
                _ = self._sim_partial_processes[0].particles["source_primary"]
            except rebound.ParticleNotFound:
                print("ERROR:")
                print("Source collided with primary or other object!")
                print("aborting simulation...")
                return
            self._sim.add(self._sim_partial_processes[0].particles[act])

        self._sim.N_active = self._sim_partial_processes[0].N_active

        # Transfer particles
        num_lost = 0
        for proc in range(len(self._sim_partial_processes)):

            dc = self._sim_partial_processes[proc]

            dc_remove = []
            for particle in dc.particles[dc.N_active:]:

                try:
                    w = self.hash_dict[f"{particle.hash.value}"]['weight']
                    species_id = self.hash_dict[f"{particle.hash.value}"]['id']
                    w = particle.params["serpens_weight"]
                    species_id = particle.params["serpens_species"]
                except:
                    print("Particle not found.")
                    dc_remove.append(particle.hash)
                    continue

                species = self.params.get_species(id=species_id)
                mass_inject_per_advance = (species.mass_per_sec * self.params.int_spec["sim_advance"] *
                                           self.var["source_P"])
                pps = species.particles_per_superparticle(mass_inject_per_advance)

                particle_distance = np.linalg.norm(
                    np.asarray(particle.xyz) - np.asarray(self._sim.particles["source_primary"].xyz))

                if particle_distance > self.var["boundary"] or w * pps < 1e10:
                    try:
                        dc_remove.append(particle.hash)
                    except RuntimeError:
                        print("Removal error occurred.")
                        pass
                else:
                    self._sim.add(particle)

            for hash in dc_remove:
                dc.remove(hash=hash)
                try:
                    del self.hash_dict[f"{hash.value}"]
                except:
                    continue
                num_lost += 1

    def advance(self, num_sim_advances, save_freq=1, verbose=False):
        """
        Main function to be called for advancing the SERPENS simulation.
        Uses internal function to add particles, include loss for super-particles, and integrate in time using
        multiprocessing. Saves resulting REBOUND simulation state to disk.

        Arguments
        ---------
        num_sim_advances : int
            Number of advances to simulate.
        save_freq : int     (default: 1)
            Number of advances after which SERPENS saves the simulation instance.
        verbose : bool      (default: False)
            Enable printing of logs.
        """
        start_time = time.time()
        steady_state_counter = 0
        steady_state_breaker = None

        for _ in tqdm(range(num_sim_advances), disable=verbose):
            if verbose:
                print(f"Starting advance {self.var['iter']} ... ")

            n_before = self._sim.N
            self.single_advance(verbose=verbose)
            self._sim.save_to_file("archive.bin")
            self._rebx.save("rebx.bin")

            if verbose:
                t = self._sim_partial_processes[0].t
                print(f"Advance done! \n"
                      f"Simulation time [h]: {np.around(t / 3600, 2)} \n"
                      f"Simulation runtime [s]: {np.around(time.time() - start_time, 2)} \n"
                      f"Number of particles: {self._sim.N}")

            # Handle saves
            if self.var["iter"] % save_freq == 0:
                if verbose: print("Saving hash dict...")
                dict_saver = {f"{str(self.var['iter'] + 1)}": copy.deepcopy(self.hash_dict)}
                with open("hash_library.pickle", 'ab') as f:
                    dill.dump(dict_saver, f, dill.HIGHEST_PROTOCOL)

            # Handle steady state (1/2)
            if np.abs(self._sim.N - n_before) < 50:
                steady_state_counter += 1
                if steady_state_counter == 10 and self.params.int_spec["stop_at_steady_state"] is True:
                    print("Steady state reached!")
                    print("Stopping after another successful revolution...")
                    steady_state_breaker = 1
            else:
                steady_state_counter = 0

            # Handle steady state (2/2)
            if steady_state_breaker is not None:
                print(f"Advances left: {1 / self.params.int_spec['sim_advance'] - steady_state_breaker}")
                if steady_state_breaker == 1 / self.params.int_spec["sim_advance"]:
                    break
                else:
                    steady_state_breaker += 1

            # End iteration
            self.var["iter"] += 1
            if verbose:
                print("\t ... done!\n============================================")

        print('                 .                     .        .   *          .            \n . '
              '         \          .            .       .           .      .            \n      .      \   ,          '
              '          ,                ,    ,               \n   .          o     .           SIMULATION FINISHED! '
              '               .       \n     .         \                 ,                       .                . '
              '\n               #\##\#      .                              .        .        \n             #  '
              '#O##\###                .                        .          \n   .        #*#  #\##\###                '
              '       .                     ,     \n        .   ##*#  #\##\##               .                     .   '
              '          \n      .      ##*#  #o##\#         .                             ,       .   \n          .  '
              '   *#  #\#     .                    .             .          , \n                      \          .    '
              '                     .                '
              '\n____^/\___^--____/\____O______________/\/\---/\___________---______________ \n   /\^   ^  ^    ^     '
              '             ^^ ^  \'\ ^          ^       ---        \n         --           -            --  -      - '
              '        ---  __       ^     \n(~ ASCII Art by Robert Casey)  ___--  ^  ^                         --  '
              '__   ')

if __name__ == "__main__":
    params = Parameters()
    with open("Parameters.pickle", 'wb') as f:
        dill.dump(params, f, protocol=dill.HIGHEST_PROTOCOL)

    ssim = SerpensSimulation(system="Jupiter (Europa-Source)")
    ssim.advance(Parameters.int_spec["num_sim_advances"])
